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

'''C-PCM / COSMO solver options: the skipped transposed solve, Cholesky and
FP32-Cholesky-with-refinement factorisations, the one-call gradient identity
and the FP32 dS gradient kernel, each against the arithmetic it replaces.'''

import unittest
import numpy
import cupy
import pyscf
from gpu4pyscf import scf
from gpu4pyscf.solvent import pcm
from gpu4pyscf.solvent.grad import pcm as pcm_grad
from cupyx.scipy.linalg import lu_solve

def setUpModule():
    global mol
    mol = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
C       0.0000000000     2.5000000000     0.3000000000
O       0.0000000000     3.6000000000     0.6000000000
H       0.9300000000     1.9400000000     0.2000000000
H      -0.9300000000     1.9400000000     0.2000000000
''', basis='def2-svp', verbose=0, output='/dev/null')

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _run(method='C-PCM', solver='lu', grad_fp32=False, cls=None):
    cm = (cls or pcm.PCM)(mol)
    cm.method = method
    cm.cpcm_solver = solver
    cm.grad_ds_fp32 = grad_fp32
    mf = scf.RHF(mol).PCM(cm)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf

class _TwoSolvePCM(pcm.PCM):
    '''The arithmetic before the change: both solves for every model.'''
    def _symmetrize_q(self, q, v_grids):
        vK_1 = self.left_solve_K(v_grids.T, K_transpose = True)
        qt = self.left_multiply_R(vK_1, R_transpose = True).T
        return (q + qt)/2.0

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ref = _run()
        cls.ref_grad = cls.ref.nuc_grad_method().kernel()

    def test_defaults_unchanged(self):
        cm = pcm.PCM(mol)
        self.assertEqual(cm.cpcm_solver, 'lu')
        self.assertFalse(cm.grad_ds_fp32)
        self.assertIn('K_LU', self.ref.with_solvent._intermediates)

    def test_skipped_transposed_solve_is_the_same_answer(self):
        two = _run(cls=_TwoSolvePCM)
        self.assertAlmostEqual(two.e_tot, self.ref.e_tot, delta=1e-10)
        q_new = self.ref.with_solvent._intermediates['q_sym']
        q_old = two.with_solvent._intermediates['q_sym']
        self.assertLess(float(abs(q_new - q_old).max()), 1e-10 * float(abs(q_old).max()))
        g_two = two.nuc_grad_method().kernel()
        self.assertLess(abs(g_two - self.ref_grad).max(), 1e-9)

    def test_symmetrize_skips_only_for_cpcm_category(self):
        for method, skip in (('C-PCM', True), ('COSMO', True), ('IEF-PCM', False),
                             ('SS(V)PE', False)):
            cm = pcm.PCM(mol)
            cm.method = method
            cm.build()
            calls = []
            orig = cm.left_solve_K
            cm.left_solve_K = lambda *a, **k: calls.append(k) or orig(*a, **k)
            v = cupy.ones((1, cm.surface['grid_coords'].shape[0]))
            q = cm.left_solve_K(cm.left_multiply_R(v.T)).T
            calls.clear()
            cm._symmetrize_q(q, v)
            self.assertEqual(len(calls), 0 if skip else 1, method)

    def test_one_call_gradient_identity(self):
        cm = self.ref.with_solvent
        dm = self.ref.make_rdm1()
        it = cm._intermediates
        fast = pcm_grad.grad_solver(cm, dm)
        # explicit copies switch the identity off: the general two-call path
        slow = pcm_grad.grad_solver(cm, dm, v_grids=it['v_grids'].copy(),
                                    v_grids_l=it['v_grids'].copy(), q=it['q'].copy())
        self.assertLess(float(abs(fast - slow).max()), 1e-11)
        self.assertGreater(float(abs(slow).max()), 1e-4)

    def test_cholesky_solvers_match_lu(self):
        for solver in ('cholesky', 'cholesky_ir'):
            mf = _run(solver=solver)
            self.assertAlmostEqual(mf.e_tot, self.ref.e_tot, delta=1e-10, msg=solver)
            g = mf.nuc_grad_method().kernel()
            self.assertLess(abs(g - self.ref_grad).max(), 1e-9, solver)
            it = mf.with_solvent._intermediates
            self.assertNotIn('K_LU', it)
            if solver == 'cholesky_ir':
                self.assertIn('K_chol32', it)
                self.assertNotIn('K_chol', it)
                st = it['ir_stats']
                self.assertGreater(st['solves'], 0)
                self.assertGreater(st['iters'], 0)          # FP32 alone is not FP64
                self.assertIsNone(st['fp64_fallback'])

    def test_cpcm_solve_against_lu_solve(self):
        cm = pcm.PCM(mol)
        cm.build()
        K_LU, piv = cm._intermediates['K_LU'], cm._intermediates['K_LU_pivot']
        n = K_LU.shape[0]
        _, K = pcm.get_D_S(cm.surface, with_S=True, with_D=False)
        rng = cupy.random.default_rng(7)
        for b in (rng.standard_normal(n), rng.standard_normal((n, 3))):
            ref = lu_solve((K_LU, piv), b)
            for solver in ('cholesky', 'cholesky_ir'):
                it = pcm.cpcm_factorize(K.copy(), solver)
                x = pcm.cpcm_solve(it, b)
                self.assertEqual(x.shape, b.shape)
                err = float(abs(x - ref).max()) / float(abs(ref).max())
                self.assertLess(err, 1e-11, solver)

    def test_refinement_fallback_when_not_converged(self):
        cm = pcm.PCM(mol)
        cm.build()
        _, K = pcm.get_D_S(cm.surface, with_S=True, with_D=False)
        b = cupy.arange(K.shape[0], dtype=numpy.float64)
        ref = pcm.cpcm_solve(pcm.cpcm_factorize(K.copy(), 'cholesky'), b)
        orig = pcm.CPCM_IR_MAX_ITER
        try:
            pcm.CPCM_IR_MAX_ITER = 0
            it = pcm.cpcm_factorize(K.copy(), 'cholesky_ir')
            x = pcm.cpcm_solve(it, b)
        finally:
            pcm.CPCM_IR_MAX_ITER = orig
        self.assertIn('not converged', it['ir_stats']['fp64_fallback'])
        self.assertIn('K_chol', it)
        self.assertLess(float(abs(x - ref).max()), 1e-13 * float(abs(ref).max()))

    def test_fp32_cholesky_failure_falls_back(self):
        cm = pcm.PCM(mol)
        cm.build()
        _, K = pcm.get_D_S(cm.surface, with_S=True, with_D=False)
        real = cupy.linalg.cholesky
        def fake(a):
            out = real(a)
            if a.dtype == numpy.float32:
                out[0, 0] = numpy.nan
            return out
        cupy.linalg.cholesky = fake
        try:
            it = pcm.cpcm_factorize(K.copy(), 'cholesky_ir')
        finally:
            cupy.linalg.cholesky = real
        self.assertEqual(it['ir_stats']['fp64_fallback'], 'FP32 Cholesky failed')
        self.assertIn('K_chol', it)
        self.assertNotIn('K_chol32', it)

    def test_refusals(self):
        cm = pcm.PCM(mol)
        cm.cpcm_solver = 'qr'
        self.assertRaises(ValueError, cm.build)
        for method in ('IEF-PCM', 'SS(V)PE'):
            cm = pcm.PCM(mol)
            cm.method = method
            cm.cpcm_solver = 'cholesky'
            self.assertRaises(NotImplementedError, cm.build)
        cm = pcm.PCM(mol)
        cm.cpcm_solver = 'cholesky_ir'
        cm.lowmem_intermediate_storage = True
        self.assertRaises(NotImplementedError, cm.build)
        from gpu4pyscf.solvent import smd
        sm = smd.SMD(mol)
        sm.solvent = 'water'
        sm.build()
        sm.cpcm_solver = 'cholesky'
        self.assertRaises(NotImplementedError, sm.left_solve_K, cupy.ones(sm.surface['grid_coords'].shape[0]))
        mf = _run(method='IEF-PCM')
        mf.with_solvent.grad_ds_fp32 = True
        self.assertRaises(NotImplementedError, mf.nuc_grad_method().kernel)

    def test_fp32_ds_kernel(self):
        cm = self.ref.with_solvent
        n = cm.surface['grid_coords'].shape[0]
        v = cupy.random.default_rng(3).standard_normal(n)
        for transpose in (False, True):
            ref = pcm_grad.left_multiply_dS_offdiagonal(cm.surface, v, transpose=transpose)
            got = pcm_grad.left_multiply_dS_offdiagonal_fp32(cm.surface, v, transpose=transpose)
            self.assertEqual(got.shape, ref.shape)
            self.assertLess(float(abs(got - ref).max()), 1e-5 * float(abs(ref).max()))

    def test_fp32_ds_gradient(self):
        mf = _run(grad_fp32=True)
        self.assertAlmostEqual(mf.e_tot, self.ref.e_tot, delta=1e-12)
        g = mf.nuc_grad_method().kernel()
        self.assertLess(abs(g - self.ref_grad).max(), 1e-8)

if __name__ == "__main__":
    print("Full Tests for C-PCM solver options")
    unittest.main()
