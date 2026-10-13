/*
 * Copyright 2026 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Mixed-precision variants of the two derivative kernels the PCM gradient's grad_qv calls
 * (solvent/grad/pcm.py; opt-in through PCM.grad_qv_fp32):
 *
 *   GINTfill_int3c1e_ip1_charge_contracted_fp32   sum_C q_C (d/dA mu | 1/|r-C| | nu)
 *   GINTfill_int3c1e_ip2_density_contracted_fp32  sum_{mu nu} D_{mu nu} (mu | d/dC 1/|r-C| | nu)
 *
 * Same tasks, launch shapes, outputs and accumulation as GINTfill_int3c1e_ip1_charge_contracted and
 * GINTfill_int3c1e_ip2_density_contracted (nr_fill_ao_int3c1e_ip.cu / g3c1e_ip.cu). What changes:
 *   - FP64: the Rys roots and weights (GINTrys_root), the pair and charge geometry, the prefactor,
 *     and every sum that leaves a thread (the per-thread caches, the Hermite-density products, the
 *     atomicAdd into the output);
 *   - FP32: the 1D Rys recursion and horizontal transfer (g), and the root sums of the derivative
 *     products, i.e. the arithmetic that grows as (l+1)^2 x roots per primitive pair and charge.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gint.h"
#include "gint1e.h"
#include "cuda_alloc.cuh"
#include "cint2e.cuh"

#include "rys_roots.cu"

// The 1D integrals of GINT_g1e (g1e.cu) with the recursion in FP32. g holds gx, gy, gz, each
// NROOTS * (i_l + 1) * (j_l + 1) floats, in GINT_g1e's layout.
template <int NROOTS>
__device__
static void g1e_fp32(float* __restrict__ g, const double* __restrict__ grid_point,
                     const int ish, const int jsh, const int prim_ij,
                     const int i_l, const int j_l, const double charge_exponent, const double omega)
{
    const double aij = c_bpcache.a12[prim_ij];
    const double eij = c_bpcache.e12[prim_ij];
    const double Px  = c_bpcache.x12[prim_ij];
    const double Py  = c_bpcache.y12[prim_ij];
    const double Pz  = c_bpcache.z12[prim_ij];
    const double PCx = Px - grid_point[0];
    const double PCy = Py - grid_point[1];
    const double PCz = Pz - grid_point[2];

    double a0 = aij;
    const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
    const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
    a0 *= q_over_p_plus_q;
    const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
    const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
    a0 *= theta;

    const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
    const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
    double uw[NROOTS * 2];
    GINTrys_root<NROOTS>(boys_input, uw);
    GINTscale_u<NROOTS>(uw, theta);
    const double* __restrict__ u = uw;
    const double* __restrict__ w = u + NROOTS;

    const int g_size = NROOTS * (i_l + 1) * (j_l + 1);
    float* __restrict__ gx = g;
    float* __restrict__ gy = g + g_size;
    float* __restrict__ gz = g + g_size * 2;

    const int nbas = c_bpcache.nbas;
    const double* __restrict__ bas_x = c_bpcache.bas_coords;
    const double* __restrict__ bas_y = bas_x + nbas;
    const double* __restrict__ bas_z = bas_y + nbas;
    const double Ax = bas_x[ish];
    const double Ay = bas_y[ish];
    const double Az = bas_z[ish];
    const double PAx = Px - Ax;
    const double PAy = Py - Ay;
    const double PAz = Pz - Az;

#pragma unroll
    for (int i_root = 0; i_root < NROOTS; i_root++) {
        gx[i_root] = 1.0f;
        gy[i_root] = (float)prefactor;
        gz[i_root] = (float)w[i_root];

        const double u2 = a0 * u[i_root];
        const double qt2_over_p_plus_q = u2 / (u2 + aij * q_over_p_plus_q) * q_over_p_plus_q;
        const float b10 = (float)(0.5 / aij * (1.0 - qt2_over_p_plus_q));
        const float c00x = (float)(PAx - qt2_over_p_plus_q * PCx);
        const float c00y = (float)(PAy - qt2_over_p_plus_q * PCy);
        const float c00z = (float)(PAz - qt2_over_p_plus_q * PCz);

        if (i_l + j_l > 0) {
            float s0x = gx[i_root];
            float s0y = gy[i_root];
            float s0z = gz[i_root];
            float s1x = c00x * s0x;
            float s1y = c00y * s0y;
            float s1z = c00z * s0z;
            gx[i_root + NROOTS] = s1x;
            gy[i_root + NROOTS] = s1y;
            gz[i_root + NROOTS] = s1z;
            for (int i_rys = 1; i_rys < i_l + j_l; i_rys++) {
                const float s2x = c00x * s1x + i_rys * b10 * s0x;
                const float s2y = c00y * s1y + i_rys * b10 * s0y;
                const float s2z = c00z * s1z + i_rys * b10 * s0z;
                gx[i_root + (i_rys+1) * NROOTS] = s2x;
                gy[i_root + (i_rys+1) * NROOTS] = s2y;
                gz[i_root + (i_rys+1) * NROOTS] = s2z;
                s0x = s1x; s0y = s1y; s0z = s1z;
                s1x = s2x; s1y = s2y; s1z = s2z;
            }
        }
    }

    if (j_l > 0) {
        const float ABx = (float)(Ax - bas_x[jsh]);
        const float ABy = (float)(Ay - bas_y[jsh]);
        const float ABz = (float)(Az - bas_z[jsh]);
        for (int j_rys = 0; j_rys < j_l; j_rys++) {
            for (int i_rys = i_l + j_l - j_rys - 1; i_rys >= 0; i_rys--) {
                for (int i_root = 0; i_root < NROOTS; i_root++) {
                    const int dst = i_root + (i_rys + (j_rys+1) * (i_l+1)) * NROOTS;
                    const int up  = i_root + (i_rys+1 + j_rys * (i_l+1)) * NROOTS;
                    const int src = i_root + (i_rys + j_rys * (i_l+1)) * NROOTS;
                    gx[dst] = gx[up] + ABx * gx[src];
                    gy[dst] = gy[up] + ABy * gy[src];
                    gz[dst] = gz[up] + ABz * gz[src];
                }
            }
        }
    }
}

// The 1D integrals of GINT_g1e_without_hrr_save_u2 (g1e.cu), recursion in FP32; u2 saved in FP32.
template <int L_SUM>
__device__
static void g1e_without_hrr_save_u2_fp32(float* __restrict__ g, float* __restrict__ u2_save,
                                         const double grid_x, const double grid_y, const double grid_z,
                                         const int ish, const int prim_ij, const double charge_exponent,
                                         const double omega)
{
    constexpr int NROOTS = L_SUM / 2 + 1;
    const double aij = c_bpcache.a12[prim_ij];
    const double eij = c_bpcache.e12[prim_ij];
    const double Px  = c_bpcache.x12[prim_ij];
    const double Py  = c_bpcache.y12[prim_ij];
    const double Pz  = c_bpcache.z12[prim_ij];
    const double PCx = Px - grid_x;
    const double PCy = Py - grid_y;
    const double PCz = Pz - grid_z;

    double a0 = aij;
    const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
    const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
    a0 *= q_over_p_plus_q;
    const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
    const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
    a0 *= theta;

    const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q;
    const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
    double uw[NROOTS * 2];
    GINTrys_root<NROOTS>(boys_input, uw);
    GINTscale_u<NROOTS>(uw, theta);
    const double* __restrict__ u = uw;
    const double* __restrict__ w = u + NROOTS;

    constexpr int g_size = NROOTS * (L_SUM + 1);
    float* __restrict__ gx = g;
    float* __restrict__ gy = g + g_size;
    float* __restrict__ gz = g + g_size * 2;

    const int nbas = c_bpcache.nbas;
    const double* __restrict__ bas_x = c_bpcache.bas_coords;
    const double* __restrict__ bas_y = bas_x + nbas;
    const double* __restrict__ bas_z = bas_y + nbas;
    const double PAx = Px - bas_x[ish];
    const double PAy = Py - bas_y[ish];
    const double PAz = Pz - bas_z[ish];

#pragma unroll
    for (int i_root = 0; i_root < NROOTS; i_root++) {
        gx[i_root] = 1.0f;
        gy[i_root] = (float)prefactor;
        gz[i_root] = (float)w[i_root];

        const double u2 = a0 * u[i_root];
        u2_save[i_root] = (float)(charge_exponent > 0.0 ? u2 * charge_exponent / (u2 + charge_exponent) : u2);
        const double qt2_over_p_plus_q = u2 / (u2 + aij * q_over_p_plus_q) * q_over_p_plus_q;
        const float b10 = (float)(0.5 / aij * (1.0 - qt2_over_p_plus_q));
        const float c00x = (float)(PAx - qt2_over_p_plus_q * PCx);
        const float c00y = (float)(PAy - qt2_over_p_plus_q * PCy);
        const float c00z = (float)(PAz - qt2_over_p_plus_q * PCz);

        if constexpr (L_SUM > 0) {
            float s0x = gx[i_root];
            float s0y = gy[i_root];
            float s0z = gz[i_root];
            float s1x = c00x * s0x;
            float s1y = c00y * s0y;
            float s1z = c00z * s0z;
            gx[i_root + NROOTS] = s1x;
            gy[i_root + NROOTS] = s1y;
            gz[i_root + NROOTS] = s1z;
#pragma unroll
            for (int i_rys = 1; i_rys < L_SUM; i_rys++) {
                const float s2x = c00x * s1x + i_rys * b10 * s0x;
                const float s2y = c00y * s1y + i_rys * b10 * s0y;
                const float s2z = c00z * s1z + i_rys * b10 * s0z;
                gx[i_root + (i_rys+1) * NROOTS] = s2x;
                gy[i_root + (i_rys+1) * NROOTS] = s2y;
                gz[i_root + (i_rys+1) * NROOTS] = s2z;
                s0x = s1x; s0y = s1y; s0z = s1z;
                s1x = s2x; s1y = s2y; s1z = s2z;
            }
        }
    }
}

// ip1, charge contracted: GINTfill_int3c1e_ip1_charge_contracted_kernel_expanded with FP32 g.
template <int LI, int LJ>
__global__
static void ip1_charge_contracted_fp32_kernel(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                              const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                              const double omega, const double* grid_points, const double* charge_exponents)
{
    constexpr int NROOTS = (LI + LJ + 1) / 2 + 1;
    constexpr int NI = (LI + 1) * (LI + 2) / 2;
    constexpr int NJ = (LJ + 1) * (LJ + 2) / 2;
    constexpr int NIJ = NI * NJ;
    constexpr int G_SIZE = NROOTS * (LI + 2) * (LJ + 1);

    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_ij >= ntasks_ij) {
        return;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int ish = c_bpcache.bas_pair2bra[bas_ij];
    const int jsh = c_bpcache.bas_pair2ket[bas_ij];
    const double* __restrict__ a_exponents = c_bpcache.a1;

    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    double output_cache[NIJ * 3] { 0.0 };

    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double charge = grid_point[3];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        float g[3 * G_SIZE];
        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            g1e_fp32<NROOTS>(g, grid_point, ish, jsh, ij, LI + 1, LJ, charge_exponent, omega);
            const float minus_two_a = (float)(-2.0 * a_exponents[ij]);
            const float* __restrict__ gx = g;
            const float* __restrict__ gy = g + G_SIZE;
            const float* __restrict__ gz = g + G_SIZE * 2;
#pragma unroll
            for (int j = 0; j < NJ; j++) {
#pragma unroll
                for (int i = 0; i < NI; i++) {
                    const int loc_j = c_l_locs[LJ] + j;
                    const int loc_i = c_l_locs[LI] + i;
                    const int ix = idx[loc_i], iy = idy[loc_i], iz = idz[loc_i];
                    const int jx = idx[loc_j], jy = idy[loc_j], jz = idz[loc_j];
                    const int ox = ix + jx * (LI + 2);
                    const int oy = iy + jy * (LI + 2);
                    const int oz = iz + jz * (LI + 2);
                    float dAx = 0.0f, dAy = 0.0f, dAz = 0.0f;
#pragma unroll
                    for (int r = 0; r < NROOTS; r++) {
                        const float gx0 = gx[ox * NROOTS + r];
                        const float gy0 = gy[oy * NROOTS + r];
                        const float gz0 = gz[oz * NROOTS + r];
                        const float dgx = (ix > 0 ? ix * gx[(ox - 1) * NROOTS + r] : 0.0f) + minus_two_a * gx[(ox + 1) * NROOTS + r];
                        const float dgy = (iy > 0 ? iy * gy[(oy - 1) * NROOTS + r] : 0.0f) + minus_two_a * gy[(oy + 1) * NROOTS + r];
                        const float dgz = (iz > 0 ? iz * gz[(oz - 1) * NROOTS + r] : 0.0f) + minus_two_a * gz[(oz + 1) * NROOTS + r];
                        dAx += dgx * gy0 * gz0;
                        dAy += gx0 * dgy * gz0;
                        dAz += gx0 * gy0 * dgz;
                    }
                    output_cache[i + j * NI + 0 * NIJ] += (double)dAx * charge;
                    output_cache[i + j * NI + 1 * NIJ] += (double)dAy * charge;
                    output_cache[i + j * NI + 2 * NIJ] += (double)dAz * charge;
                }
            }
        }
    }

    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
#pragma unroll
    for (int j = 0; j < NJ; j++) {
#pragma unroll
        for (int i = 0; i < NI; i++) {
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 0 * stride_ij), output_cache[i + j * NI + 0 * NIJ]);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 1 * stride_ij), output_cache[i + j * NI + 1 * NIJ]);
            atomicAdd(output + ((i + i0) + (j + j0) * stride_j + 2 * stride_ij), output_cache[i + j * NI + 2 * NIJ]);
        }
    }
}

// ip2, density contracted: GINTfill_int3c1e_ip2_density_contracted_kernel_general with FP32 g.
template <int L_SUM>
__global__
static void ip2_density_contracted_fp32_kernel(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                               const BasisProdOffsets offsets, const int nprim_ij,
                                               const double omega, const double* grid_points, const double* charge_exponents)
{
    constexpr int NROOTS = (L_SUM + 1) / 2 + 1;
    constexpr int NHERMITE = (L_SUM + 1) * (L_SUM + 2) * (L_SUM + 3) / 6;
    constexpr int G_SIZE = NROOTS * (L_SUM + 2);

    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_grid >= ngrids) {
        return;
    }
    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    for (int task_ij = blockIdx.x * blockDim.x + threadIdx.x; task_ij < ntasks_ij; task_ij += gridDim.x * blockDim.x) {
        const int bas_ij = offsets.bas_ij + task_ij;
        const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
        const int ish = c_bpcache.bas_pair2bra[bas_ij];
        const int nbas = c_bpcache.nbas;
        const double* __restrict__ bas_x = c_bpcache.bas_coords;
        const float ACx = (float)(bas_x[ish] - Cx);
        const float ACy = (float)(bas_x[ish + nbas] - Cy);
        const float ACz = (float)(bas_x[ish + 2 * nbas] - Cz);

        double D_hermite[NHERMITE];
#pragma unroll
        for (int t = 0; t < NHERMITE; t++) {
            D_hermite[t] = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair
                                   + hermite_density_offsets.density_offset_of_angular_pair
                                   + t * hermite_density_offsets.n_pair_of_angular_pair];
        }

        for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
            float g[3 * G_SIZE];
            float u2[NROOTS];
            g1e_without_hrr_save_u2_fp32<L_SUM + 1>(g, u2, Cx, Cy, Cz, ish, ij, charge_exponent, omega);
            const float* __restrict__ gx = g;
            const float* __restrict__ gy = g + G_SIZE;
            const float* __restrict__ gz = g + G_SIZE * 2;
#pragma unroll
            for (int i_x = 0, t = 0; i_x <= L_SUM; i_x++) {
#pragma unroll
                for (int i_y = 0; i_x + i_y <= L_SUM; i_y++) {
#pragma unroll
                    for (int i_z = 0; i_x + i_y + i_z <= L_SUM; i_z++, t++) {
                        float dCx = 0.0f, dCy = 0.0f, dCz = 0.0f;
#pragma unroll
                        for (int r = 0; r < NROOTS; r++) {
                            const float gx0 = gx[r + NROOTS * i_x];
                            const float gy0 = gy[r + NROOTS * i_y];
                            const float gz0 = gz[r + NROOTS * i_z];
                            const float m2u2 = -2.0f * u2[r];
                            dCx += m2u2 * (gx[r + NROOTS * (i_x + 1)] + ACx * gx0) * gy0 * gz0;
                            dCy += gx0 * m2u2 * (gy[r + NROOTS * (i_y + 1)] + ACy * gy0) * gz0;
                            dCz += gx0 * gy0 * m2u2 * (gz[r + NROOTS * (i_z + 1)] + ACz * gz0);
                        }
                        const double Dt = D_hermite[t];
                        sum_x += (double)dCx * Dt;
                        sum_y += (double)dCy * Dt;
                        sum_z += (double)dCz * Dt;
                    }
                }
            }
        }
    }
    atomicAdd(output + task_grid + ngrids * 0, sum_x);
    atomicAdd(output + task_grid + ngrids * 1, sum_y);
    atomicAdd(output + task_grid + ngrids * 2, sum_z);
}

#define IP1_CASE(LI, LJ) \
    case (LI * 10 + LJ): ip1_charge_contracted_fp32_kernel<LI, LJ> <<<blocks, threads, 0, stream>>>( \
        output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;

static int ip1_charge_contracted_fp32_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                            const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                            const double omega, const double* grid_points, const double* charge_exponents,
                                            const int n_charge_sum_per_thread, const cudaStream_t stream)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = (offsets.ntasks_kl + n_charge_sum_per_thread - 1) / n_charge_sum_per_thread;
    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    switch (i_l * 10 + j_l) {
    IP1_CASE(0, 0) IP1_CASE(0, 1) IP1_CASE(0, 2) IP1_CASE(0, 3) IP1_CASE(0, 4)
    IP1_CASE(1, 0) IP1_CASE(1, 1) IP1_CASE(1, 2) IP1_CASE(1, 3) IP1_CASE(1, 4)
    IP1_CASE(2, 0) IP1_CASE(2, 1) IP1_CASE(2, 2) IP1_CASE(2, 3) IP1_CASE(2, 4)
    IP1_CASE(3, 0) IP1_CASE(3, 1) IP1_CASE(3, 2) IP1_CASE(3, 3) IP1_CASE(3, 4)
    IP1_CASE(4, 0) IP1_CASE(4, 1) IP1_CASE(4, 2) IP1_CASE(4, 3) IP1_CASE(4, 4)
    default:
        fprintf(stderr, "ip1 fp32: (i_l, j_l) = (%d, %d) out of range\n", i_l, j_l);
        return 1;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

#define IP2_CASE(L) \
    case L: ip2_density_contracted_fp32_kernel<L> <<<blocks, threads, 0, stream>>>( \
        output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;

static int ip2_density_contracted_fp32_tasks(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                             const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                             const double omega, const double* grid_points, const double* charge_exponents,
                                             const int n_pair_sum_per_thread, const cudaStream_t stream)
{
    const int ntasks_ij = (offsets.ntasks_ij + n_pair_sum_per_thread - 1) / n_pair_sum_per_thread;
    const int ngrids = offsets.ntasks_kl;
    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    switch (i_l + j_l) {
    IP2_CASE(0) IP2_CASE(1) IP2_CASE(2) IP2_CASE(3) IP2_CASE(4)
    IP2_CASE(5) IP2_CASE(6) IP2_CASE(7) IP2_CASE(8)
    default:
        fprintf(stderr, "ip2 fp32: i_l + j_l = %d out of range\n", i_l + j_l);
        return 1;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
int GINTfill_int3c1e_ip1_charge_contracted_fp32(const cudaStream_t stream, const BasisProdCache* bpcache,
                                                const double* grid_points, const double* charge_exponents, const int ngrids,
                                                double* integral_charge_contracted,
                                                const int* strides, const int* ao_offsets,
                                                const int* bins_locs_ij, const int nbins,
                                                const int cp_ij_id, const double omega, const int n_charge_sum_per_thread)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;
    if (nrys_roots > MAX_NROOTS_INT3C_1E + 1) {
        fprintf(stderr, "nrys_roots = %d too high\n", nrys_roots);
        return 2;
    }
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }
        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;
        const int err = ip1_charge_contracted_fp32_tasks(integral_charge_contracted, offsets, i_l, j_l, nprim_ij,
                                                         strides[0], strides[1], ao_offsets[0], ao_offsets[1],
                                                         omega, grid_points, charge_exponents, n_charge_sum_per_thread, stream);
        if (err != 0) {
            return err;
        }
    }
    return 0;
}

int GINTfill_int3c1e_ip2_density_contracted_fp32(const cudaStream_t stream, const BasisProdCache* bpcache,
                                                 const double* grid_points, const double* charge_exponents, const int ngrids,
                                                 const double* dm_pair_ordered, const int* density_offset,
                                                 double* integral_density_contracted,
                                                 const int* bins_locs_ij, const int nbins,
                                                 const int cp_ij_id, const double omega, const int n_pair_sum_per_thread)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l + 1) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;
    if (nrys_roots > MAX_NROOTS_INT3C_1E + 1) {
        fprintf(stderr, "nrys_roots = %d too high\n", nrys_roots);
        return 2;
    }
    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }
        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;
        HermiteDensityOffsets hermite_density_offsets;
        hermite_density_offsets.density_offset_of_angular_pair = density_offset[cp_ij_id];
        hermite_density_offsets.pair_offset_of_angular_pair = bas_pairs_locs[cp_ij_id];
        hermite_density_offsets.n_pair_of_angular_pair = bas_pairs_locs[cp_ij_id + 1] - bas_pairs_locs[cp_ij_id];
        const int err = ip2_density_contracted_fp32_tasks(integral_density_contracted, dm_pair_ordered, hermite_density_offsets,
                                                          offsets, i_l, j_l, nprim_ij,
                                                          omega, grid_points, charge_exponents, n_pair_sum_per_thread, stream);
        if (err != 0) {
            return err;
        }
    }
    return 0;
}
}
