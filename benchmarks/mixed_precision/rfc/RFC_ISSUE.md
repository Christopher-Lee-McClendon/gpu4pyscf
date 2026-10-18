<!--
STAGING FILE: the text of a GitHub issue for pyscf/gpu4pyscf. Not yet posted. Open this issue
first and wait for maintainer feedback before any PR (see CHECKLIST.md). Every number below is
checked against the evidence package by `python3 benchmarks/mixed_precision/rfc/verify_rfc.py`.
-->

# RFC: opt-in mixed-precision SCF for density-fitted RKS on FP64-limited GPUs

## Problem

On GPUs whose FP64 throughput is a small fraction of FP32 (RTX and workstation Blackwell cards,
L40S), a DF-RKS SCF spends most of its time in two FP64 contractions: the XC quadrature and, for
global hybrids, the two DF exchange contractions. Early SCF iterations do not need FP64 there, but
the result must still be an FP64 SCF, with stock convergence and stock gradients.

We have an implementation on a fork and would like to know whether the maintainers want it, and
in what shape, before opening a PR. The code is
[`chris-lee-mc/gpu4pyscf@mixed-precision-upstream-on-9af748a`](https://github.com/chris-lee-mc/gpu4pyscf/tree/mixed-precision-upstream-on-9af748a)
(four commits on master `9af748a`; +3054/−39 over ten files, of which the first PR we would
propose is +1033/−17 over six). The evidence package is
[`chris-lee-mc/gpu4pyscf@mixed-precision-evidence`, `benchmarks/mixed_precision/`](https://github.com/chris-lee-mc/gpu4pyscf/tree/mixed-precision-evidence/benchmarks/mixed_precision)
(upstream master plus that directory): its `native/verify_native.py` recomputes every benchmark
figure quoted here from the per-run data, its `validation/` holds the GPU validation records, and
its `README.md` documents the API, the refusals and the VV10 and AO-cache details left out here.

## Proposed API

A policy object on the SCF object, off by default (`mf.mixed_precision = None`):

```python
from gpu4pyscf.dft import rks
from gpu4pyscf.dft.mixed_precision import MixedPrecision

mf = rks.RKS(mol, xc='b3lyp').density_fit()
mf.mixed_precision = MixedPrecision(xc=True, k=True)   # or MixedPrecision(vv10=True) for wB97M-V
mf.kernel()
mf.mixed_precision_record      # precision of every iteration and why it switched; None before kernel()
# other fields: xc_switch_tol=1e-3, k_switch_tol=1e-3, vv10_switch_tol=1e-5, stall=2, call_cap=30,
#               ao_cache_mem_fraction=0.7, ao_cache_fp64=False
```

Every component defaults to off; nothing is enabled by GPU model, in line with the policy stated in
#855 ("We don't plan to enable fp32 by default, as its numerical behavior is not guaranteed"), which
landed FP32 multigrid as an experimental, default-OFF build option, `ENABLE_FP32_MULTIGRID`. Other
reduced-precision switches in tree: `mp/dfmp2.py`'s `fp_type` (a `__config__`-backed attribute in
`_keys`, default `'FP64'`), `tdscf/ris.py`'s `single` (default True, so not an opt-in; it warns
below `conv_tol` 1e-5) and the cuEST wrapper's `math_mode` (a whole-handle setting, initialised to
`native_fp64`, that holds for every call); the policy could take a `__config__` default the same
way. The closest prior art is #214, a mixed-precision direct-SCF trial closed unmerged, which split
direct-SCF J/K by integral magnitude for the whole run; this proposal is DF only, uses FP32 only in
early iterations, always finishes convergence in FP64, and refuses unsupported cases instead of
falling back silently.

Unsupported configurations are refused with `NotImplementedError` at `kernel()`, never silently:
UKS/ROKS/GKS and spin ≠ 0; multi-GPU; the second-order solver (`mf.newton()`); with `xc`/`k`,
range-separated and NLC functionals; `k` without density fitting or on a functional without exact
exchange; with `vv10`, a functional without exactly one VV10 term. The 56 tests are unmarked, skip
on multi-GPU hosts (`num_devices > 1`, upstream's pattern; the refusal is tested with a patched
device count), and pass on a V100 with upstream's single-GPU CI package set (libxc 0.9.0, 133 s)
and under pyscf 2.8.

## How precision is controlled

| component | FP32 phase | always FP64 |
|---|---|---|
| `xc` | rho and vxc contractions, against an FP32 copy of the AO values built once per SCF | `eval_xc_eff`, grid weights, electron count, the E_xc reduction, the Vxc accumulator; every block result is promoted before accumulation |
| `k` (global hybrids, DF only) | the two K contractions of `df_jk.get_jk`, on FP32 chunks of `cderi` | J, and the K accumulator |
| `vv10` (exactly one VV10 term, runs alone) | the O(N²) U/W/E pair sum, FP32 pair terms with FP64 accumulation | everything outside the pair sum; after the switch the pair sum runs a df64 (double-float, ~48-bit) kernel, and the last call is certified against stock FP64 |

- **FP32 only in early iterations.** Each component switches one way when the change in its energy
  (E_xc for `xc` and `k`, E_nlc for `vv10`) falls below its threshold, backed by a stall detector
  and an iteration cap; the thresholds are controller settings, not error bounds. With `xc` and `k`
  both on, K returns to FP64 no later than XC and is held at FP64 whenever XC falls back to it.
- **Convergence is always finished in FP64.** `scf._kernel` accepts convergence only after two
  consecutive iterations built without FP32-phase arithmetic; if the tests pass earlier, the switch
  is forced and the SCF continues. DF `get_veff` rebuilds J/K from the full density every iteration.
- **Energies are accumulated in FP64**; gradients and Hessians run stock, in FP64. **With
  `mf.mixed_precision` unset, no code path changes**: every hook is behind a `None` check, and
  four of upstream's test files (`test_df_rks`, `test_df_jk`, `test_rks`, `test_scf`) pass
  identically on both trees.

## Evidence

Speed was measured with an earlier head of the fork (tree `63af0568`: the same algorithms; the
modules changed since only in docstrings, one `block_loop` keyword, the newton and non-hybrid `k` refusals
and the XC-fallback fix, a path the measured runs did not take) overlaid on the v1.8.1 `cuda12x` wheel with
`cupy-cuda12x` 14.1.1 and cuTENSOR 2.3.1 (required and checked as loaded in every run); the
master-based branch was re-validated for correctness, not speed (below). Two caveats on that stack:
the cuda12x wheel ships no sm_120 code, so on the Blackwell card both arms ran gpu4pyscf's own
kernels from JIT-compiled compute_80 PTX (a cuda13x wheel, with native sm_120 code, was not
measured); and upstream's README pairs CuPy with cuTENSOR 2.0.2 or 2.2.0 and #518 recommends 2.0.2,
so a re-measurement on a recommended pair is a fair ask. Every ratio is **stock wall / mixed wall**
(medians of three warm pairs per cell; above 1 means mixed is faster), to two decimals here and
three in the package. The **library defaults** (the default grid, level 3 with `nwchem_prune`; no AO
cache; default tolerances; the policies above) were measured on three molecules and one card
(CLAIMS C7); everything else is in the **campaign configuration** (`ao_cache_fp64=True`,
`grids.prune = None`, B3LYP `xc_switch_tol=3e-4`, cuTENSOR 2.3.1, def2-mTZVPP / def2-tzvpp-jkfit,
`conv_tol=1e-9`; CLAIMS C1–C6).

**Speed, RTX PRO 6000 Blackwell Workstation** (one card family; one pod for the r2SCAN/B3LYP ladder):

| functional | library defaults, 3 molecules (C7) | campaign configuration, 24 molecules of 20–44 atoms (C1) | campaign per-cell range | campaign pods |
|---|---|---|---|---|
| r2SCAN (`xc`) | **1.58** | 1.66 | 1.52–1.87 | 1 |
| B3LYP (`xc`, `k`) | **1.64** | 1.78 | 1.58–1.99 | 1 |
| wB97M-V (`vv10`) | **3.09** | 2.91 | 2.59–3.25 | 5, on 3 cards |

Two further wB97M-V pods read about 30 % lower (2.06 on three molecules; 1.92–2.13 on four) and are
excluded under a contention rule (idle draw above 200 W, PREREG-0 Amendment 3) written after they
had been seen; their re-runs on quiet cards read 2.96 and 2.87–3.05. On shared GPUs expect the lower
readings. Without the AO cache, on the unpruned grid, the FP32 switch reads 1.63 / 1.95 / 2.88, so
the cache is a small part of the gain. On six molecules of 63–98 atoms: r2SCAN 1.93 (1.80 without
one anomalous stock wall), B3LYP 1.34 (MODEL-MISS, below the L tier's 1.897). B3LYP's gain falls
with size; at 475–559 Da the J/K stage, about half the stock SCF, gains only 5–7 % while XC gains
2–3×, which accounts for part of the drop, and the rest is unexplained (CLAIMS C2). The 24-molecule
pod's two replicates were both DEGRADED; on clean cells they read 0.05–0.09 higher (CLAIMS C1).

**Negative result, FP64-strong cards** (CLAIMS C3). On an L40S the mode pays (r2SCAN 1.73, B3LYP
1.52, wB97M-V 2.99). On an H100 it does not: wB97M-V 0.58–0.61 (i.e. slower), B3LYP 0.91–1.02, and
r2SCAN's 1.32–1.40 comes almost entirely from the FP64 AO cache (the FP32 switch itself reads
1.08). The pre-registered predictions for H100 B3LYP and wB97M-V were 1.20 and 1.35 (MODEL-MISS,
far below). On an A100 (both pods noisy, so clean cells only), the FP32 switch's own effect was neutral or slower in 11 of 12 readings; against stock, wB97M-V reads 0.49–0.60 and B3LYP 0.71–1.09, while r2SCAN stays at 1.14–1.39 because of the AO cache. **The code does not detect
the device**; see question 2.

**Accuracy** (CLAIMS C4, PREREG-8 A1/A2; mixed M against stock S run from the same guess with the
same settings, at `conv_tol_grad = 3e-6`, the setting we recommend for gradients and properties):

| max \|M−S\| at `conv_tol_grad` 3e-6 | gradient (Ha/Bohr) | dipole (D) |
|---|---|---|
| r2SCAN (6 molecules) | 1.5e-9 – 2.3e-8 | ≤ 1.4e-6 |
| B3LYP (same 6) | 6.8e-9 – 2.2e-7 | ≤ 8.7e-6 |
| wB97M-V (2 molecules) | 1.1e-8 – 2.1e-8 | ≤ 4.5e-12 |

Cycle counts are identical in every one of these 14 cells; only omeprazole B3LYP exceeds 1e-7 Ha/Bohr.
The caveat: this is the tighter tolerance, not gpu4pyscf's default `conv_tol_grad` (√`conv_tol`).

*At the default tolerance* (CLAIMS C4, PREREG-10 DR1, `STATUS=PASS`; quotable): the twelve r2SCAN
and B3LYP cells (the same six molecules) were measured in triplicate on one pod, three same-guess
pairs S_k/M_k per cell plus a tight stock reference R (`conv_tol` 1e-11). All 36 pairs agree to
\|ΔE\| ≤ 1.7e-10 Ha, with cycle counts equal in 33 of 36. Mixed vs stock, range over the cells and
the three repeats:

| max \|M−S\| at the default `conv_tol_grad` | gradient (Ha/Bohr) | dipole (D) |
|---|---|---|
| r2SCAN (6 molecules, 3 repeats) | 3.2e-9 – 6.5e-8 | 2.2e-7 – 2.9e-6 |
| B3LYP (5 molecules, 3 repeats) | 2.0e-8 – 2.7e-7 | 2.0e-7 – 1.0e-5 |
| omeprazole B3LYP (3 repeats) | 4.4e-6 | 1.9e-4 |
| wB97M-V (6 molecules, single runs, PREREG-4 D2a–c, PASS) | 1.2e-8 – 3.2e-8 | ≤ 5.9e-12 |

Omeprazole B3LYP is the one cell where mixed and stock stop on different iterates: mixed takes one
extra cycle in every repeat (14 against 13) and lands closer than stock to R (gradient 1.0e-6 –
1.05e-6 against 3.4e-6 Ha/Bohr; dipole 3.5e-5 – 3.6e-5 against 1.57e-4 D). In every cell mixed is
no further from R than twice stock's own distance plus a small floor (the largest ratio is 0.56).
Stock's own repeat scatter is ≤ 3.2e-12 Ha/Bohr for r2SCAN and up to 5.6e-8 for B3LYP (predicted
≤ 3e-8: MODEL-MISS). Label: an earlier single-run measurement of the same twelve cells (PREREG-4
D1) is `STATUS=FAIL` on one pre-registered gate, a dipole residual stock itself carries
(fluconazole r2SCAN, 1.004e-4 D against a 1e-4 D ceiling; S−R 9.93e-5 D, M−S 1.05e-6 D), and stays
UNQUOTABLE; the protocol was redesigned after seeing it and re-run in triplicate. At the default
`conv_tol_grad` both arms keep a 1e-6-scale residual against R, so the tight-tolerance table above
stays the recommendation for gradients and properties.

**Geometry optimisation** (CLAIMS C6; geomeTRIC, stock gradients, one card, one run per arm):
paracetamol r2SCAN, paracetamol B3LYP and celecoxib r2SCAN take the same 23, 26 and 16 steps as
stock, end within 9.9e-6 Å of stock's endpoint and pass a fresh stock SCF and gradient there;
warm-step SCFs run 1.41–1.44 and the whole optimisations 1.19–1.30 faster than stock.[^mw]

[^mw]: The whole-optimisation ratios are from the arm that ran a warm-start rule (a fork option,
not proposed; it never fired, so that arm's steps and cycles equal the plain arm's) because the
plain arm's first SCF paid the driver JIT in one cell. Where the plain arm is valid it reads 1.16
(paracetamol B3LYP) and 1.27 (celecoxib r2SCAN).

**Correctness of the master-based branch** (fork `2b27755` on `9af748a`, built from source, three
profiles; `validation/` in the package): upstream `test_df_rks`, `test_df_jk`, `test_rks` and
`test_scf` give 44 passed and 1 skipped on master and the same on the branch, at the lock's pins,
under pyscf 2.8 and on the V100; the 56 new tests pass on all three profiles; stock vs mixed on
three molecules agrees to \|ΔE\| ≤ 7.3e-12 Ha (3.2e-12 Ha under pyscf 2.8) with identical cycle
counts.

## What is not claimed

- **Library-default settings**: measured only on three molecules, one card and def2-mTZVPP (C7).
- **Running without cuTENSOR**: speed not measured; correctness was validated on both engines.
- **UKS, multi-GPU, gradients, Hessians**: out of scope (refused or untouched). Range-separated
  functionals with `xc`/`k`: refused; NLC functionals are treated only by the separate `vv10`.
- **Architectures**: speed on RTX PRO 6000 (Workstation and Server), L40S, H100 and A100;
  correctness also on a V100; not the RTX 5090, B200 or any consumer card. Only two FP64:FP32
  throughput levels were measured, so no gating threshold is established.

## Questions for the maintainers

0. **Do you want this in core at all?** If not, would you accept two small seams so the rest can
   live outside the tree: `check_convergence` parity in `scf._kernel` (`soscf` already honours
   `mf.check_convergence`; `_kernel` does not), and the keywords `nr_nlc_vxc(vv10_kernel=)` /
   `_vv10nlc(uwe_kernel=)` in `numint`, default `None`? Everything else can then subclass `RKS`.
1. **API shape.** A policy object or a flag on `density_fit()` / `RKS`? The object keeps the
   switch settings and the record in one place; a `__config__` default could sit beside it.
2. **Device guard.** On the H100 and A100 the FP32 switch is slower than stock. Should the mode
   warn or refuse by compute capability (sm_80/sm_90 class), or only document it? We have no
   measured threshold to gate on.
3. **Scope of a first PR.** XC + K alone (+1033/−17, six files, 27 tests; applies on master alone),
   with the VV10 component (two NVRTC `cupy.RawModule` kernels and the two `numint` keywords) and
   the FP64 AO cache as follow-ups? The cache commit applies cleanly only after VV10 and would be
   re-authored if you want it second. XC + K alone gives the no-cache readings above (1.63 r2SCAN,
   1.95 B3LYP on the unpruned grid; 1.58 and 1.64 at the defaults).
4. **Convergence contract.** Is "two consecutive FP64 iterations before convergence is accepted",
   enforced in `scf._kernel`, acceptable there, or should it live in the RKS classes? The
   alternative is question 0's parity call, with the policy installing `mf.check_convergence`.
5. **Defaults, tolerances and the record.** Are `xc_switch_tol = k_switch_tol = 1e-3` and the test
   tolerance \|ΔE\| ≤ 1e-8 Ha against stock acceptable (the campaign used `xc_switch_tol=3e-4` for
   B3LYP; at 1e-3 the B3LYP switch came after 8 FP32 XC calls instead of 9–11)? And is
   `mf.mixed_precision_record` (a dict, `None` before `kernel()`) with two new `KohnShamDFT._keys`
   entries fine, or would you prefer the record on the logger only?
