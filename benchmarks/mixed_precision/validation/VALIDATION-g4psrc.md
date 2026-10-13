# GPU validation of the master-based code branch, built from source (g4psrc)

The branch proposed upstream, `chris-lee-mc/gpu4pyscf@mixed-precision-upstream`, was validated on
upstream master with gpu4pyscf **built from source**: for sm_120 on one RTX PRO 6000 Blackwell
Workstation Edition (r1–r6), and in r4–r6 also for sm_70 on one Tesla V100-SXM2-16GB. The protocol,
gates, profiles and pre-dispatch sections are in the engine repository:
`docs/upstream/PORT-PLAN-upstream-source-build.md`, in `chris-lee-mc/gpu-conformer-engine`.

The method is one build and two trees:
- gpu4pyscf's CUDA libraries are built once, from the base commit, with
  `-DCUDA_ARCHITECTURES=<arch>-real`, then copied byte for byte into the branch tree. The branch
  changes only `.py` files, and the pod checks this.
- Stock master and the branch then run the same tests with the same libraries.
- The released wheel is uninstalled first, and every test run records which tree it imported.

## Runs

| run | fork commit | upstream master | profile (card; packages after the lock) | result |
|---|---|---|---|---|
| r1 (`37723300240`) | `5b9c87c0` | `eef5f5b3` | PRO 6000; the lock | PASS: regression 43/43 on both trees, 49/49 new tests, trio \|ΔE\| ≤ 5.9e-12 Ha |
| r2 (`37881608304`) | `122e2d5e` | `82bc7028` | PRO 6000; the lock | PASS: regression 44 passed + 1 skipped on both trees, 49/49, \|ΔE\| ≤ 6.8e-12 Ha |
| r3 (`38017282503`) | `ac4c668e` | `c1a6e371` | PRO 6000; the lock | PASS: regression 44 passed + 1 skipped on both trees, 49/49, \|ΔE\| ≤ 6.4e-12 Ha |
| r4 `pro6000-lock` (`38022477235`) | `4d2f3de4` | `c1a6e371` | PRO 6000; the lock (pyscf 2.14.0, cupy 14.1.1, libxc 0.8.1, no cutensor) | PASS G1–G8: regression 44 passed + 1 skipped on both trees, 50/50, \|ΔE\| ≤ 5.9e-12 Ha |
| r4 `pro6000-pyscf28` (`38023769024`) | `4d2f3de4` | `c1a6e371` | PRO 6000; upstream's `requirements.txt`, then pyscf 2.8.0 and libxc 0.8.1 (cupy 13.4.1, scipy 1.15.3, cutensor 2.2.0) | PASS G1–G8: regression 44 passed + 1 skipped on both trees, 50/50, \|ΔE\| ≤ 2.7e-12 Ha |
| r4 `v100-ci` (`38024957471`) | `4d2f3de4` | `c1a6e371` | Tesla V100-SXM2-16GB (sm_70, `70-real`); upstream's `requirements.txt`, then libxc 0.9.0 (pyscf 2.14.0, cupy 13.4.1, cutensor 2.2.0) | PASS G1–G7; G8 skipped by profile, no verdict. Regression 44 passed + 1 failed on both trees (upstream's own, below); 50/50 new tests in 133 s |
| r5 superseded (`38194143624`) | `3d9de5d7` | `e1020c32` | PRO 6000; the lock | PASS G1–G8: regression 44 passed + 1 skipped on both trees, 50/50, \|ΔE\| ≤ 7.7e-12 Ha. **Superseded**: the head before the review fixes; no record banked |
| r5 `pro6000-lock` (`38198811227`) | `d642f3da` | `e1020c32` | PRO 6000; the lock (pyscf 2.14.0, cupy 14.1.1, libxc 0.8.1, no cutensor) | PASS G1–G8: regression 44 passed + 1 skipped on both trees, 56/56 in 136 s, \|ΔE\| ≤ 5.9e-12 Ha, cycles identical, VV10 certificate ok in all 3 wB97M-V cells |
| r5 `pro6000-pyscf28` (`38201109296`) | `d642f3da` | `e1020c32` | PRO 6000; upstream's `requirements.txt`, then pyscf 2.8.0 and libxc 0.8.1 (cupy 13.4.1, scipy 1.15.3, cutensor 2.2.0) | PASS G1–G8: regression 44 passed + 1 skipped on both trees, 56/56 in 123 s, \|ΔE\| ≤ 5.0e-12 Ha, cycles identical |
| r5 `v100-ci` (`38200149447`) | `d642f3da` | `e1020c32` | Tesla V100-SXM2-16GB (sm_70, `70-real`); upstream's `requirements.txt`, then libxc 0.9.0 (pyscf 2.14.0, cupy 13.4.1, cutensor 2.2.0) | PASS G1–G7; G8 skipped by profile, no verdict. Regression 44 passed + 1 skipped on both trees; 56/56 new tests in 136 s |
| **r6 `pro6000-lock`** (`38333841493`) | **`cb1a2d78`** | **`38df3b74`** | PRO 6000; the lock (pyscf 2.14.0, cupy 14.1.1, libxc 0.8.1, no cutensor) | **PASS G1–G8**: regression 44 passed + 1 skipped on both trees, 56/56 in 123 s, \|ΔE\| ≤ 5.9e-12 Ha, cycles identical, VV10 certificate ok in all 3 wB97M-V cells |
| **r6 `pro6000-pyscf28`** (`38336271676`) | **`cb1a2d78`** | **`38df3b74`** | PRO 6000; upstream's `requirements.txt`, then pyscf 2.8.0 and libxc 0.8.1 (cupy 13.4.1, scipy 1.15.3, cutensor 2.2.0) | **PASS G1–G8**: regression 44 passed + 1 skipped on both trees, 56/56 in 123 s, \|ΔE\| ≤ 4.1e-12 Ha, cycles identical, VV10 certificate ok ×3 |
| **r6 `v100-ci`** (`38335226746`) | **`cb1a2d78`** | **`38df3b74`** | Tesla V100-SXM2-16GB (sm_70, `70-real`); upstream's `requirements.txt`, then libxc 0.9.0 (pyscf 2.14.0, cupy 13.4.1, cutensor 2.2.0) | **PASS G1–G7**; G8 skipped by profile, no verdict. Regression 44 passed + 1 skipped on both trees; 56/56 new tests in 132 s |

**r6 validates the current head of the branch, `cb1a2d7817422a9136c502dab9419f4c1ead10b2`**, on
upstream master `38df3b74dc67658e79e3e442cf95f177ab00f2bf`. It is r5's four commits rebased, with no
conflicts, onto `38df3b7`, one upstream commit past `e1020c3`: #950 ("Actual SAD guess"), which
edits `scf/hf.py`, a file the branch also edits (the branch's hunks, in `scf()` and `_kernel`, are
textually independent of it). The branch's own added and removed lines are identical to
`d642f3da`'s, so the diff is still +3054/−39 over ten files with 56 new tests. Upstream's
`requirements.txt` and `unittest.yml` pins are unchanged, so the profiles are r5's. The split
commits are `85a7395` (XC + K), `1348d50` (VV10), `124ee0b` (AO cache) and `cb1a2d7` (docstring and
README).

r5 validated `d642f3da` on upstream master `e1020c32` (runs `38198811227`, `38201109296` and
`38200149447`); its rows above are kept as history. In r5 the base moved from `c1a6e371` by five
upstream commits, none touching the branch's files; one of them, #960, makes upstream's
`test_nr_coach` skip on a pyscf older than 2.15 or a libxc without `COACH`. The code changed since
r4 by the independent review: the four new files carry `# Copyright 2026`; `mf.newton()` with a
policy set refuses (`NotImplementedError`, in `scf/soscf.py`, so the diff now touches ten files);
`k=True` on a functional without exact exchange refuses (it was a silent no-op);
`mf.mixed_precision_record` defaults to `None`; with `xc` and `k` both on, K is held at FP64
whenever XC falls back to FP64; and six new tests cover these, so the new tests number 56 (38 in
`test_mixed_precision.py`, 18 in `test_mixed_precision_vv10.py`). Its split commits were `9b129af`
(XC + K), `301cfe9` (VV10), `5e81866` (AO cache) and `d642f3d` (docstring and README).

The superseded run `38194143624` validated `3d9de5d7` on the same base before the review fixes
(G1–G8 PASS, 50/50, \|ΔE\| ≤ 7.7e-12 Ha); it is recorded here only as superseded and its JSON is
not banked. r4 validated `4d2f3de4` on `c1a6e371`; its change since r3 was tests only
(`@unittest.skipIf(num_devices > 1, ...)` on the test classes that run an SCF with a policy, and a
`MultiGPU` test that patches the device count and asserts the refusal). The earlier heads are kept
on the fork as `mixed-precision-upstream-on-e1020c3` (r5, `d642f3da`),
`mixed-precision-upstream-on-e1020c3-prereview` (`3d9de5d`),
`mixed-precision-upstream-on-c1a6e37-r4` (r4, `4d2f3de`), `mixed-precision-upstream-on-c1a6e37`
(r3, `ac4c668e`), `-on-82bc702` (r2) and `-on-eef5f5b` (r1). The r4 `v100-ci` run was the one
infrastructure re-run of `38023715438`, which found no V100 capacity and created no pod. The
records are `g4psrc_r2.json`, `g4psrc_r3.json`, the three `g4psrc_r4_*.json`, the three
`g4psrc_r5_*.json` and the three `g4psrc_r6_*.json` (`pro6000-lock`, `pro6000-pyscf28`, `v100-ci`): the pod's JSON, with the build
log tail dropped and the run id and status added.

**Why three profiles.** Upstream's `unittest.yml` runs its single-GPU job on a V100 with libxc
0.9.0 and its multi-GPU job on two T4s with pyscf 2.8; r1–r3 ran neither card nor package set. A
profile names the card and the packages installed after the lock, and the gate set: `v100-ci`
omits G8 because its bands and walls are banked on the PRO 6000 and the trio's tier-L estimate does
not fit a 16 GB card beside another tenant.

## Gates

The gate definitions are unchanged since r3; r6 changed only the harness's `BASE_SHA`. The r4 and r5
values per profile are in the runs table above and in the `g4psrc_r4_*.json` and `g4psrc_r5_*.json`
records.

| gate | definition | r6 `pro6000-lock` | r6 `pro6000-pyscf28` | r6 `v100-ci` |
|---|---|---|---|---|
| G1 commits | HEAD is the named SHA; the base is an ancestor | `cb1a2d78` on `38df3b74` | same | same |
| G2 scope | the branch changes only `.py` files under `gpu4pyscf/` plus `README.md`; no compiled input differs | 9 files under `gpu4pyscf/` (`scf/soscf.py` new in r5) plus `README.md` | same | same |
| G3 build | rc 0, 12 libraries, branch copies byte-identical | 12 libraries, 88 s | 12 libraries, 138 s | 12 libraries for `70-real`, 308 s |
| G4 provenance | the wheel was removed; each tree imported itself; libxc came from its wheel; the installed versions equal the profile's | same | same | same |
| G5 regression | upstream `test_df_rks`, `test_df_jk`, `test_rks`, `test_scf`: no test that passed on master failed or skipped on the branch | 44 passed + 1 skipped / same | 44 passed + 1 skipped / same | 44 passed + 1 skipped / same |
| G6 new tests | `test_mixed_precision.py` and `test_mixed_precision_vv10.py`, every test passes, no skips | 38 + 18: 56/56, 123 s | 56/56, 123 s | 56/56, 132 s |
| G7 VV10 identity | the FP32 and df64 launchers are bitwise identical to the NumPy emulation | bitwise | bitwise | bitwise, on sm_70 |
| G8 trio | paracetamol, propranolol and celecoxib × r2SCAN, B3LYP, wB97M-V: \|ΔE\| ≤ 1e-8 Ha, cycles ±1, an FP64 tail, real FP32 use; VV10 certificate in band | \|ΔE\| ≤ 5.9e-12 Ha, cycles identical, certificate ok ×3 | \|ΔE\| ≤ 4.1e-12 Ha, cycles identical, certificate ok ×3 | not run (skipped by profile) |

**Disclosed, not gated (r6).**
- **#950 changes nothing the gates read.** The results match r5's: the same counts on every
  profile, the same trio cycle counts, and \|ΔE\| in the same 1e-12 range. Where #950 changed the
  initial guess, both trees and both sides of the trio start from it.
- **COACH stays skipped.** The V100 profile again reads 44 passed + 1 skipped on both trees.
- **V100 durations, new tests only** (`pytest --durations=20`, in `g4psrc_r6_v100-ci.json`): 132 s
  in all. The slowest single test is 18.5 s (`test_vv10_paracetamol`), then 16.8 s
  (`test_kernels_against_stock_vv10nlc`), 10.8 s and 9.2 s; everything else is under 8 s. One
  RunPod V100, serially, not upstream's runner with `-n 4`.
- **The contraction engine** read `cupy` under the lock and `None` under both `requirements.txt`
  pin sets, on both trees, as in r4 and r5. Recorded only.

**Disclosed, not gated (r5; kept as the record of that run).**
- **The COACH failure is gone.** On the V100 profile (libxc 0.9.0) r5 reads 44 passed + 1 skipped
  on both trees, where r4 read 44 passed + 1 failed: upstream #960, on the new base `e1020c3`,
  makes `test_nr_coach` skip there. The pod's summary counts do not name the skipped test.
- **V100 durations, new tests only** (`pytest --durations=20`, in `g4psrc_r5_v100-ci.json`): 136 s
  in all. The slowest single test is 18.4 s (`test_vv10_paracetamol`), then 16.9 s
  (`test_kernels_against_stock_vv10nlc`), 10.9 s and 9.4 s; everything else is under 9 s. One
  RunPod V100, serially, not upstream's runner with `-n 4`.
- **The contraction engine** read `cupy` under the lock and `None` under both `requirements.txt`
  pin sets, on both trees, as in r4. Recorded only.

**Disclosed, not gated (r4; kept as the record of that run).**
- **The V100 regression failure is upstream's.** `dft/tests/test_rks.py::KnownValues::test_nr_coach`
  fails on master and on the branch alike with `KeyError: "LibXCFunctional: name 'COACH' not
  found."`: the libxc 0.9.0 wheel that upstream's V100 job installs does not name the functional
  that upstream #938 added. G5 records base failures and does not gate them; on this pin set no
  test is skipped, which is why the base's skip count is 0 rather than 1. Upstream has since
  addressed it: #960 (on master at `e1020c3`) makes `test_nr_coach` skip on a pyscf older than
  2.15 or a libxc without `COACH`, so this base failure will not recur on a rebase past it.
- **V100 durations, new tests only** (`pytest --durations=20`, in the JSON): 133 s in all. The
  slowest single test is 18.3 s (`test_vv10_paracetamol`), then 17.5 s
  (`test_kernels_against_stock_vv10nlc`), 11.8 s and 10.0 s; everything else is under 10 s. One
  RunPod V100, serially, not upstream's runner with `-n 4`.
- **The contraction engine** read `cupy` under the lock and `None` under both `requirements.txt`
  pin sets, on both trees. Recorded only.

**Not covered (r4–r6):** multi-GPU (no T4 or multi-GPU pod was rented; the `skipIf(num_devices > 1)`
decorators never fire on one card, and the skip path is exercised only by the patched refusal
test), scipy 1.17 (it declares `requires_python >= 3.11` and the image ships 3.10; r4 Amendment 1
kept the lock's 1.15.3, and r5 and r6 kept it), upstream's full test suite, other architectures than sm_120 and sm_70,
and speed. The walls in the JSON are one run each and are not speed readings.
