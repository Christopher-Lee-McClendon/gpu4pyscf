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

'''The FP32 derivative kernels of grad_qv (lib/gint/nr_fill_ao_int3c1e_ip_fp32.cu,
PCM.grad_qv_fp32) against the FP64 kernels they shadow: every contraction the
option serves, on bases with d, f and g shells, and the PCM gradient on a fixed
density. The FP32 results must also DIFFER from FP64 (at FP32 rounding), so a
silent fall-through to the FP64 kernel cannot pass.'''

import unittest
import numpy
import cupy
import pyscf
from gpu4pyscf import scf
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2
from gpu4pyscf.solvent import pcm

ATOMS = '''
C      0.000000    0.000000    0.000000
O      0.000000    0.000000    1.205000
H      0.000000    0.940000   -0.587000
H      0.000000   -0.940000   -0.587000
O      3.000000    0.000000    0.117400
H      3.000000    0.757000   -0.469600
H      3.000000   -0.757000   -0.469600
'''
WATER = '''
O      0.000000    0.000000    0.117400
H      0.000000    0.757000   -0.469600
H      0.000000   -0.757000   -0.469600
'''
# d (def2-svp), f (def2-tzvp) and g (def2-qzvp on water) shells: every kernel template family
CASES = (('def2-svp', ATOMS), ('def2-tzvp', ATOMS), ('def2-qzvp', WATER))
REL_TOL = 1e-5          # elementwise, relative to the largest element of the FP64 result
GRAD_TOL = 1e-7         # Ha/Bohr, the PCM gradient on a fixed density

def _close_and_different(test, got, ref, what):
    got, ref = cupy.asarray(got), cupy.asarray(ref)
    test.assertEqual(got.shape, ref.shape, what)
    scale = float(abs(ref).max())
    err = float(abs(got - ref).max())
    test.assertGreater(scale, 0.0, what)
    test.assertLessEqual(err, REL_TOL * scale, f'{what}: {err:.3e} vs {REL_TOL:g} x {scale:.3e}')
    test.assertGreater(err, 1e-13 * scale, f'{what}: FP32 identical to FP64, the FP32 kernel did not run')

def _surface(mol):
    cm = pcm.PCM(mol)
    cm.build()
    return cm.surface['grid_coords'], cm.surface['charge_exp']**2

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):

    def test_default_off(self):
        mol = pyscf.M(atom=WATER, basis='sto3g', verbose=0)
        self.assertFalse(pcm.PCM(mol).grad_qv_fp32)

    def test_contractions_against_fp64(self):
        for basis, atoms in CASES:
            mol = pyscf.M(atom=atoms, basis=basis, verbose=0)
            coords, exps = _surface(mol)
            rng = cupy.random.default_rng(5)
            a = rng.standard_normal((mol.nao, mol.nao))
            dm = a + a.T
            q = rng.standard_normal(coords.shape[0])
            intopt = int3c1e.VHFOpt(mol)
            intopt.build(1e-14, aosym=False)
            kw = dict(charge_exponents=exps, intopt=intopt)
            for what, fn, args in (
                    ('ip1 dm+q', int1e_grids_ip1, dict(dm=dm, charges=q)),
                    ('ip1 q', int1e_grids_ip1, dict(charges=q)),
                    ('ip2 dm+q', int1e_grids_ip2, dict(dm=dm, charges=q)),
                    ('ip2 dm', int1e_grids_ip2, dict(dm=dm))):
                ref = fn(mol, coords, **args, **kw)
                got = fn(mol, coords, **args, **kw, fp32=True)
                _close_and_different(self, got, ref, f'{basis} {what}')

    def test_refusals(self):
        mol = pyscf.M(atom=WATER, basis='sto3g', verbose=0)
        coords, exps = _surface(mol)
        dm = cupy.eye(mol.nao)
        q = cupy.ones(coords.shape[0])
        self.assertRaises(NotImplementedError, int1e_grids_ip1, mol, coords,
                          charge_exponents=exps, dm=dm, fp32=True)
        self.assertRaises(NotImplementedError, int1e_grids_ip1, mol, coords,
                          charge_exponents=exps, fp32=True)
        self.assertRaises(NotImplementedError, int1e_grids_ip2, mol, coords,
                          charge_exponents=exps, charges=q, fp32=True)

    def test_pcm_gradient_on_a_fixed_density(self):
        mol = pyscf.M(atom=ATOMS, basis='def2-svp', verbose=0)
        for method in ('C-PCM', 'IEF-PCM'):
            cm = pcm.PCM(mol)
            cm.method = method
            mf = scf.RHF(mol).PCM(cm)
            mf.conv_tol = 1e-10
            mf.max_cycle = 100
            mf.kernel()
            self.assertTrue(mf.converged, method)
            dm = mf.make_rdm1()
            grads = {}
            for fp32 in (False, True):
                c = pcm.PCM(mol)
                c.method = method
                c.grad_qv_fp32 = fp32
                c.kernel(dm)
                grads[fp32] = c.grad(dm)
            err = float(abs(grads[True] - grads[False]).max())
            self.assertLess(err, GRAD_TOL, method)
            self.assertGreater(err, 0.0, f'{method}: identical gradients, FP32 path not taken')

    def test_end_to_end_gradient(self):
        mol = pyscf.M(atom=ATOMS, basis='def2-svp', verbose=0)
        out = {}
        for fp32 in (False, True):
            cm = pcm.PCM(mol)
            cm.grad_qv_fp32 = fp32
            mf = scf.RHF(mol).PCM(cm)
            mf.conv_tol = 1e-10
            mf.max_cycle = 100
            mf.kernel()
            self.assertTrue(mf.converged)
            out[fp32] = (mf.e_tot, mf.nuc_grad_method().kernel())
        # the SCF does not use grad_qv: the same energy up to atomicAdd ordering
        self.assertAlmostEqual(out[True][0], out[False][0], delta=1e-10)
        self.assertLess(float(numpy.abs(out[True][1] - out[False][1]).max()), GRAD_TOL)

if __name__ == "__main__":
    print("Full Tests for the FP32 grad_qv kernels")
    unittest.main()
