# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Device-resident surface integrals for the PCM family (PCM.esp_cache).

Every SCF cycle of a PCM calculation evaluates two contractions of the same
three-centre integrals T[g, mu, nu] = (mu nu | g), with g a Gaussian charge on
the cavity surface:

    v[g]       = sum_{mu nu} D[mu, nu] T[g, mu, nu]      (PCM._get_v)
    V[mu, nu]  = sum_g q[g] T[g, mu, nu]                  (PCM._get_vmat)

int1e_grids recomputes T with the Rys kernels for each. T depends only on the
geometry, so this module computes it once per PCM.build with the same fill
kernel (GINTfill_int3c1e), keeps it on the device packed over mu >= nu in the
sorted AO basis, and turns each contraction into one pass over it.

Tiers (PCM.esp_cache):
    'fp64': T stored in FP64; both contractions are cuBLAS GEMV/GEMM, exact to
            roundoff against int1e_grids.
    'fp32': T stored in FP32 (half the memory); the contractions read it and
            accumulate in FP64 (the kernels below). Only T's own rounding
            (relative 2**-24 per integral) is FP32.
If T does not fit PCM.esp_cache_memory_fraction of the free device memory the
cache is not built, int1e_grids is used, and the reason is recorded in
pcmobj._intermediates['esp_cache_info'].
'''

import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.lib import c_null_ptr
from gpu4pyscf.lib.cupy_helper import cart2sph, get_avail_mem
from gpu4pyscf.gto.int3c1e import libgint

ESP_CACHE_TIERS = ('fp64', 'fp32')
# Peak size of the dense per-batch block built before packing.
BUILD_BATCH_BYTES = 512 * 1024**2

_ESP32_SRC = r'''
extern "C" __global__
void esp_from_packed_fp32(const float* __restrict__ T, const double* __restrict__ w,
                          double* __restrict__ out, const long long npair, const int ngrids)
{
    // one block per surface point g: out[g] = sum_k T[g, k] w[k], FP64 sums
    __shared__ double red[256];
    const int g = blockIdx.x;
    if (g >= ngrids) return;
    const float* row = T + (long long)g * npair;
    double s = 0.0;
    for (long long k = threadIdx.x; k < npair; k += blockDim.x) {
        s += (double)row[k] * w[k];
    }
    red[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[g] = red[0];
}

extern "C" __global__
void vmat_from_packed_fp32(const float* __restrict__ T, const double* __restrict__ q,
                           double* __restrict__ out, const long long npair, const int ngrids)
{
    // one thread per packed pair k: out[k] = sum_g q[g] T[g, k], FP64 sums;
    // neighbouring threads read neighbouring k, so every row read is coalesced
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= npair) return;
    double s = 0.0;
    for (int g = 0; g < ngrids; g++) {
        s += q[g] * (double)T[(long long)g * npair + k];
    }
    out[k] = s;
}
'''
_kernels = {}

def _kernel(name):
    if name not in _kernels:
        _kernels[name] = cp.RawModule(code=_ESP32_SRC).get_function(name)
    return _kernels[name]

def cache_nbytes(nao, ngrids, tier):
    itemsize = 8 if tier == 'fp64' else 4
    return ngrids * (nao * (nao + 1) // 2) * itemsize

def build_esp_cache(mol, grid_coords, charge_exponents, intopt, tier, memory_fraction):
    '''Returns ({'T', 'tier', 'rows', 'cols', ...}, info) or (None, info) when the
    packed tensor does not fit memory_fraction of the free device memory.'''
    assert tier in ESP_CACHE_TIERS
    omega = mol.omega
    assert omega >= 0.0
    nao = mol.nao
    ngrids = grid_coords.shape[0]
    need = cache_nbytes(nao, ngrids, tier)
    cp.get_default_memory_pool().free_all_blocks()
    avail = get_avail_mem()
    budget = int(avail * memory_fraction)
    info = {'tier': tier, 'nbytes': need, 'budget': budget, 'ngrids': ngrids, 'nao': nao,
            'built': False}
    if need + BUILD_BATCH_BYTES > budget:
        info['reason'] = (f'packed tensor {need/1e9:.2f} GB + build batch does not fit '
                          f'{budget/1e9:.2f} GB')
        return None, info

    dtype = np.float64 if tier == 'fp64' else np.float32
    npair = nao * (nao + 1) // 2
    rows, cols = np.tril_indices(nao)
    rows = cp.asarray(rows)
    cols = cp.asarray(cols)
    T = cp.empty((ngrids, npair), dtype=dtype)

    grids = cp.asarray(grid_coords, order='C')
    exps = None
    if charge_exponents is not None:
        exps = cp.asarray(charge_exponents, order='C', dtype=np.float64)
        if exps.size == 1:
            exps = cp.zeros(ngrids) + exps
    batch = max(1, min(ngrids, BUILD_BATCH_BYTES // (8 * nao * nao)))
    stream = cp.cuda.get_current_stream()
    for p0, p1 in lib.prange(0, ngrids, batch):
        # the loop of int3c1e.get_int3c1e, kept on the device in the sorted AO basis
        dense = cp.zeros([p1-p0, nao, nao], order='C')
        for cp_ij_id, log_q_ij in enumerate(intopt.log_qs):
            if len(log_q_ij) == 0:
                continue
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]
            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
            i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
            j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]
            ni = i1 - i0
            nj = j1 - j0
            ao_offsets = np.array([i0, j0], dtype=np.int32)
            strides = np.array([ni, ni*nj], dtype=np.int32)
            block = cp.zeros([p1-p0, nj, ni], order='C')
            exps_ptr = c_null_ptr() if exps is None else exps[p0:p1].data.ptr
            grids_slice = grids[p0:p1]
            err = libgint.GINTfill_int3c1e(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(grids_slice.data.ptr, ctypes.c_void_p),
                ctypes.cast(exps_ptr, ctypes.c_void_p),
                ctypes.c_int(p1-p0),
                ctypes.cast(block.data.ptr, ctypes.c_void_p),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(1),
                ctypes.c_int(cp_ij_id),
                ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError('GINTfill_int3c1e failed')
            i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
            j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
            if not mol.cart:
                block = cart2sph(block, axis=1, ang=lj)
                block = cart2sph(block, axis=2, ang=li)
            dense[:, j0:j1, i0:i1] = block
        # the filled blocks hold mu <= nu (upper); packed storage keeps mu >= nu
        T[p0:p1] = dense[:, cols, rows].astype(dtype)
        dense = None
    info['built'] = True
    wscale = cp.where(rows == cols, 0.5, 1.0)
    return {'T': T, 'tier': tier, 'rows': rows, 'cols': cols, 'wscale': wscale,
            'nao': nao}, info

def _packed_weights(cache, dms_sorted):
    '''w[k, i] = D_i[mu, nu] + D_i[nu, mu] for mu > nu, D_i[mu, mu] on the diagonal.'''
    rows, cols = cache['rows'], cache['cols']
    w = dms_sorted[:, rows, cols] + dms_sorted[:, cols, rows]
    w *= cache['wscale']
    return w

def esp(cache, intopt, dms):
    '''v[i, g] = sum_{mu nu} D_i[mu, nu] (mu nu | g); dms (n_dm, nao, nao), original AO order.'''
    dms_sorted = intopt.sort_orbitals(cp.asarray(dms), axis=[1, 2])
    w = _packed_weights(cache, dms_sorted)                     # (n_dm, npair)
    T = cache['T']
    if cache['tier'] == 'fp64':
        return (T @ w.T).T
    ngrids, npair = T.shape
    out = cp.empty((w.shape[0], ngrids))
    fn = _kernel('esp_from_packed_fp32')
    for i in range(w.shape[0]):
        wi = cp.ascontiguousarray(w[i])
        fn((ngrids,), (256,), (T, wi, out[i], np.int64(npair), np.int32(ngrids)))
    return out

def vmat(cache, intopt, q):
    '''V[i, mu, nu] = sum_g q_i[g] (mu nu | g); q (n_dm, ngrids); original AO order.'''
    T = cache['T']
    ngrids, npair = T.shape
    q = cp.asarray(q, dtype=np.float64)
    if cache['tier'] == 'fp64':
        packed = q @ T                                         # (n_dm, npair)
    else:
        packed = cp.empty((q.shape[0], npair))
        fn = _kernel('vmat_from_packed_fp32')
        nblk = (npair + 255) // 256
        for i in range(q.shape[0]):
            qi = cp.ascontiguousarray(q[i])
            fn((nblk,), (256,), (T, qi, packed[i], np.int64(npair), np.int32(ngrids)))
    nao = cache['nao']
    rows, cols = cache['rows'], cache['cols']
    out = cp.empty((q.shape[0], nao, nao))
    out[:, rows, cols] = packed
    out[:, cols, rows] = packed
    return intopt.unsort_orbitals(out, axis=[1, 2])
