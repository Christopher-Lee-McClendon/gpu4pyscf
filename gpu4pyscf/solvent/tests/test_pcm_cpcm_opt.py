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
FP32-Cholesky-with-refinement factorisations, the one-call gradient identity,
the FP32 dS gradient kernel and the device-resident surface integrals
(esp_cache), each against the arithmetic it replaces.

Every option is compared with the default path ON THE SAME DENSITY (the
converged reference density), so an exact path is checked to roundoff and
not to SCF convergence. One end-to-end SCF per option checks that it
converges to the reference energy.'''

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
    # formaldehyde and a water 3 Angstrom away
    mol = pyscf.M(atom='''
C      0.000000    0.000000    0.000000
O      0.000000    0.000000    1.205000
H      0.000000    0.940000   -0.587000
H      0.000000   -0.940000   -0.587000
O      3.000000    0.000000    0.117400
H      3.000000    0.757000   -0.469600
H      3.000000   -0.757000   -0.469600
''', basis='def2-svp', verbose=0, output='/dev/null')

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _pcm(method='C-PCM', cls=None, **opts):
    cm = (cls or pcm.PCM)(mol)
    cm.method = method
    for k, v in opts.items():
        setattr(cm, k, v)
    return cm

def _on_dm(cm, dm):
    '''The solvent energy, potential and gradient of cm on a fixed density.'''
    e, v = cm.kernel(dm)
    g = cm.grad(dm)
    return float(e), v, g

def _scf(cm):
    mf = scf.RHF(mol).PCM(cm)
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.kernel()
    return mf

class _TwoSolvePCM(pcm.PCM):
    '''The arithmetic before the change: both solves for every model.'''
    def _symmetrize_q(self, q, v_grids):
        vK_1 = self.left_solve_K(v_grids.T, K_transpose = True)
        qt = self.left_multiply_R(vK_1, R_transpose = True).T
        return (q + qt)/2.0

def _close(test, got, ref, rel, what):
    got, ref = cupy.asarray(got), cupy.asarray(ref)
    test.assertEqual(got.shape, ref.shape, what)
    err = float(abs(got - ref).max())
    scale = float(abs(ref).max())
    test.assertLessEqual(err, rel * scale, f'{what}: {err:.3e} vs {rel:g} x {scale:.3e}')

@unittest.skipIf(pcm.libsolvent is None, "solvent extension not compiled")
class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ref = _scf(_pcm())
        assert cls.ref.converged, 'reference SCF did not converge'
        cls.dm = cls.ref.make_rdm1()
        cls.ref_e, cls.ref_v, cls.ref_g = _on_dm(_pcm(), cls.dm)

    def test_defaults_unchanged(self):
        cm = pcm.PCM(mol)
        self.assertEqual(cm.cpcm_solver, 'lu')
        self.assertFalse(cm.grad_ds_fp32)
        self.assertIsNone(cm.esp_cache)
        self.assertIn('K_LU', self.ref.with_solvent._intermediates)
        self.assertNotIn('esp_cache', self.ref.with_solvent._intermediates)
        self.assertLess(self.ref_e, -1e-4)                  # the solvent really acted

    def test_skipped_transposed_solve_is_the_same_answer(self):
        e, v, g = _on_dm(_pcm(cls=_TwoSolvePCM), self.dm)
        self.assertAlmostEqual(e, self.ref_e, delta=1e-12)
        _close(self, v, self.ref_v, 1e-10, 'Fock')
        _close(self, g, self.ref_g, 1e-9, 'gradient')

    def test_symmetrize_skips_only_for_cpcm_category(self):
        for method, skip in (('C-PCM', True), ('COSMO', True), ('IEF-PCM', False),
                             ('SS(V)PE', False)):
            cm = _pcm(method)
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
        cm = _pcm()
        cm.kernel(self.dm)
        it = cm._intermediates
        fast = pcm_grad.grad_solver(cm, self.dm)
        # explicit copies switch the identity off: the general two-call path
        slow = pcm_grad.grad_solver(cm, self.dm, v_grids=it['v_grids'].copy(),
                                    v_grids_l=it['v_grids'].copy(), q=it['q'].copy())
        self.assertGreater(float(abs(slow).max()), 1e-5)
        _close(self, fast, slow, 1e-10, 'grad_solver')

    def test_cholesky_solvers_match_lu(self):
        for solver in ('cholesky', 'cholesky_ir'):
            cm = _pcm(cpcm_solver=solver)
            e, v, g = _on_dm(cm, self.dm)
            self.assertAlmostEqual(e, self.ref_e, delta=1e-12 * abs(self.ref_e) + 1e-14)
            _close(self, v, self.ref_v, 1e-10, f'Fock {solver}')
            _close(self, g, self.ref_g, 1e-9, f'gradient {solver}')
            it = cm._intermediates
            self.assertNotIn('K_LU', it)
            if solver == 'cholesky_ir':
                self.assertIn('K_chol32', it)
                self.assertNotIn('K_chol', it)
                st = it['ir_stats']
                self.assertGreater(st['solves'], 0)
                self.assertGreater(st['iters'], 0)          # FP32 alone is not FP64
                self.assertIsNone(st['fp64_fallback'])

    def test_cpcm_solve_against_lu_solve(self):
        cm = _pcm()
        cm.build()
        K_LU, piv = cm._intermediates['K_LU'], cm._intermediates['K_LU_pivot']
        n = K_LU.shape[0]
        _, K = pcm.get_D_S(cm.surface, with_S=True, with_D=False)
        rng = cupy.random.default_rng(7)
        for b in (rng.standard_normal(n), rng.standard_normal((n, 3))):
            ref = lu_solve((K_LU, piv), b)
            for solver in ('cholesky', 'cholesky_ir'):
                x = pcm.cpcm_solve(pcm.cpcm_factorize(K.copy(), solver), b)
                _close(self, x, ref, 1e-11, f'solve {solver} {b.shape}')

    def test_refinement_on_a_cavity_like_diagonal_spread(self):
        '''PREREG-cpcm-speed-1 SB2: on drug-sized cavities a few points have a
        nearly-zero switching function, so their diagonal of K is huge. An
        unscaled stopping test was then vacuous (0 refinement iterations,
        FP32-accurate charges). Reproduce the spread and require refinement
        and per-element FP64 agreement with LU.'''
        from cupyx.scipy.linalg import lu_factor
        cm = _pcm()
        cm.build()
        _, K = pcm.get_D_S(cm.surface, with_S=True, with_D=False)
        n = K.shape[0]
        rng = cupy.random.default_rng(13)
        idx = rng.choice(n, size=max(1, n // 20), replace=False)
        K[idx, idx] *= 1e9
        lu = lu_factor(K.copy())
        for b in (rng.standard_normal(n), rng.standard_normal((n, 2))):
            ref = lu_solve(lu, b)
            it = pcm.cpcm_factorize(K.copy(), 'cholesky_ir')
            x = pcm.cpcm_solve(it, b)
            st = it['ir_stats']
            self.assertIsNone(st['fp64_fallback'])
            self.assertGreater(st['max_iters'], 0)
            err = abs(x - ref)
            bound = 1e-10 * abs(ref) + 1e-13 * float(abs(ref).max())
            self.assertTrue(bool((err <= bound).all()),
                            f'worst per-element error {float((err / (abs(ref) + 1e-300)).max()):.3e}')
            self.assertAlmostEqual(float((b * x).sum()), float((b * ref).sum()),
                                   delta=1e-12 * abs(float((b * ref).sum())))

    def test_refinement_fallback_when_not_converged(self):
        cm = _pcm()
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
        _close(self, x, ref, 1e-13, 'fallback solve')

    def test_fp32_cholesky_failure_falls_back(self):
        cm = _pcm()
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
        cm = _pcm(cpcm_solver='qr')
        self.assertRaises(ValueError, cm.build)
        for method in ('IEF-PCM', 'SS(V)PE'):
            cm = _pcm(method, cpcm_solver='cholesky')
            self.assertRaises(NotImplementedError, cm.build)
        cm = _pcm(cpcm_solver='cholesky_ir', lowmem_intermediate_storage=True)
        self.assertRaises(NotImplementedError, cm.build)
        from gpu4pyscf.solvent import smd
        sm = smd.SMD(mol)
        sm.solvent = 'water'
        sm.build()
        sm.cpcm_solver = 'cholesky'
        self.assertRaises(NotImplementedError, sm.left_solve_K,
                          cupy.ones(sm.surface['grid_coords'].shape[0]))
        cm = _pcm('IEF-PCM', grad_ds_fp32=True)
        cm.kernel(self.dm)
        self.assertRaises(NotImplementedError, cm.grad, self.dm)

    def test_fp32_ds_kernel(self):
        cm = _pcm()
        cm.build()
        n = cm.surface['grid_coords'].shape[0]
        v = cupy.random.default_rng(3).standard_normal(n)
        for transpose in (False, True):
            ref = pcm_grad.left_multiply_dS_offdiagonal(cm.surface, v, transpose=transpose)
            got = pcm_grad.left_multiply_dS_offdiagonal_fp32(cm.surface, v, transpose=transpose)
            _close(self, got, ref, 1e-5, f'dS fp32 transpose={transpose}')

    def test_fp32_ds_gradient(self):
        e, v, g = _on_dm(_pcm(grad_ds_fp32=True), self.dm)
        self.assertAlmostEqual(e, self.ref_e, delta=1e-12 * abs(self.ref_e) + 1e-14)
        self.assertLess(float(abs(g - self.ref_g).max()), 1e-8)

    def test_esp_cache_matches_int1e_grids(self):
        from gpu4pyscf.gto.int3c1e import int1e_grids
        rng = cupy.random.default_rng(11)
        nao = mol.nao
        a = rng.standard_normal((2, nao, nao))
        dms = a + a.transpose(0, 2, 1)
        for tier, tol in (('fp64', 1e-12), ('fp32', 1e-6)):
            cm = _pcm(esp_cache=tier)
            cm.build()
            info = cm._intermediates['esp_cache_info']
            self.assertTrue(info['built'], info)
            self.assertEqual(cm._intermediates['esp_cache']['T'].dtype,
                             numpy.float64 if tier == 'fp64' else numpy.float32)
            coords = cm.surface['grid_coords']
            exps = cm.surface['charge_exp']**2
            ref_v = int1e_grids(mol, coords, dm=dms, charge_exponents=exps, intopt=cm.intopt)
            _close(self, cm._get_v(dms), ref_v, tol, f'esp {tier}')
            q = rng.standard_normal((2, coords.shape[0]))
            ref_m = cupy.stack([-int1e_grids(mol, coords, charges=q[i], charge_exponents=exps,
                                             intopt=cm.intopt) for i in range(2)])
            _close(self, cm._get_vmat(q), ref_m, tol, f'vmat {tier}')

    def test_esp_cache_on_the_reference_density(self):
        for tier, rel_e, rel_v, dg in (('fp64', 1e-12, 1e-10, 1e-9), ('fp32', 1e-6, 1e-6, 1e-7)):
            e, v, g = _on_dm(_pcm(esp_cache=tier), self.dm)
            self.assertAlmostEqual(e, self.ref_e, delta=rel_e * abs(self.ref_e) + 1e-14, msg=tier)
            _close(self, v, self.ref_v, rel_v, f'Fock {tier}')
            self.assertLess(float(abs(g - self.ref_g).max()), dg, tier)

    def test_esp_cache_over_budget_falls_back_and_records(self):
        cm = _pcm(esp_cache='fp64', esp_cache_memory_fraction=1e-9)
        e, v, g = _on_dm(cm, self.dm)
        it = cm._intermediates
        self.assertIsNone(it['esp_cache'])
        self.assertFalse(it['esp_cache_info']['built'])
        self.assertIn('does not fit', it['esp_cache_info']['reason'])
        self.assertAlmostEqual(e, self.ref_e, delta=1e-12 * abs(self.ref_e) + 1e-14)

    def test_esp_cache_bad_tier(self):
        cm = _pcm(esp_cache='fp16')
        self.assertRaises(ValueError, cm.build)

    def test_end_to_end_scf(self):
        '''Each option converges to the reference SCF energy (to the convergence
        tolerance, not to roundoff: the iterates differ).'''
        for opts in ({'cpcm_solver': 'cholesky'}, {'cpcm_solver': 'cholesky_ir'},
                     {'esp_cache': 'fp64'}, {'esp_cache': 'fp32'},
                     {'cpcm_solver': 'cholesky_ir', 'esp_cache': 'fp32', 'grad_ds_fp32': True}):
            mf = _scf(_pcm(**opts))
            self.assertTrue(mf.converged, opts)
            self.assertAlmostEqual(mf.e_tot, self.ref.e_tot, delta=1e-8, msg=str(opts))

if __name__ == "__main__":
    print("Full Tests for C-PCM solver options")
    unittest.main()
