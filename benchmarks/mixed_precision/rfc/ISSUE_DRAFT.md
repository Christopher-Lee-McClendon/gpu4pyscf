Title: RFC: opt-in mixed-precision SCF for density-fitted RKS on FP64-limited GPUs

On GPUs whose FP64 throughput is a small fraction of FP32 (RTX and workstation Blackwell cards, L40S), a
DF-RKS SCF spends most of its time in two FP64 contractions: the XC quadrature and, for global hybrids, the
two DF exchange contractions. The early iterations do not need FP64 there, but the result must still be an
FP64 SCF with stock convergence and stock gradients. We have an implementation on a fork and would like to
know whether you want it, and in what shape, before opening a PR.

## Proposal

A policy object on the SCF object, off by default (`mf.mixed_precision = None`):

```python
from gpu4pyscf.dft import rks
from gpu4pyscf.dft.mixed_precision import MixedPrecision

mf = rks.RKS(mol, xc='b3lyp').density_fit()
mf.mixed_precision = MixedPrecision(xc=True, k=True)   # or MixedPrecision(vv10=True) for wB97M-V
mf.kernel()
mf.mixed_precision_record      # precision of every iteration and why it switched; None before kernel()
```

Every component defaults to off and nothing is enabled by GPU model, as with `fp_type` in `mp/dfmp2.py` and
`single` in `tdscf/ris.py`. Unsupported cases are refused with `NotImplementedError` at `kernel()`, never
silently: UKS/ROKS/GKS and spin ≠ 0, multi-GPU, `mf.newton()`, range-separated and NLC functionals with
`xc`/`k`, `k` without density fitting or without exact exchange, and `vv10` without exactly one VV10 term.

How precision is kept safe:

| component | FP32 phase | always FP64 |
|---|---|---|
| `xc` | rho and vxc contractions, against an FP32 copy of the AO values built once per SCF | `eval_xc_eff`, grid weights, electron count, the E_xc reduction, the Vxc accumulator; every block result is promoted before accumulation |
| `k` (global hybrids, DF only) | the two K contractions of `df_jk.get_jk`, on FP32 chunks of `cderi` | J, and the K accumulator |
| `vv10` (exactly one VV10 term) | the O(N²) U/W/E pair sum, FP32 pair terms with FP64 accumulation | everything outside the pair sum; after the switch the pair sum runs a double-float (~48-bit) kernel, and the last call is certified against the stock FP64 kernel |

- FP32 only in early iterations: each component switches one way when the change in its energy (E_xc for `xc`
  and `k`, E_nlc for `vv10`) falls below its threshold, with a stall detector and an iteration cap behind it;
  the thresholds are controller settings, not error bounds. With both on, K returns to FP64 no later than XC.
- Convergence is always finished in FP64: `scf._kernel` accepts convergence only after two consecutive
  iterations built without FP32 arithmetic, forcing the switch if the tests pass earlier. DF `get_veff`
  rebuilds J/K from the full density every iteration, so nothing computed in FP32 survives.
- Energies accumulate in FP64; gradients and Hessians run stock. With `mf.mixed_precision` unset nothing
  changes (every hook is behind a `None` check), and four upstream test files pass identically on both trees.

## Evidence

Speed was measured with an earlier head of the fork (the same modules up to a docstring and one keyword) on
the v1.8.1 wheel; the master-based branch was re-validated for correctness only. Every ratio is stock wall /
mixed wall (medians of three warm pairs per cell; above 1 means mixed is faster). "Library defaults" is the
default pruned grid, no AO cache and the default tolerances; the "measured configuration" is
`ao_cache_fp64=True`, grid pruning off, B3LYP `xc_switch_tol=3e-4`, cuTENSOR 2.3.1, def2-mTZVPP /
def2-tzvpp-jkfit and `conv_tol=1e-9`. On an RTX PRO 6000 Blackwell Workstation:

| functional | library defaults, 3 molecules | measured configuration, 24 molecules of 20–44 atoms | per-cell range |
|---|---|---|---|
| r2SCAN (`xc`) | 1.58 | 1.66 | 1.52–1.87 |
| B3LYP (`xc`, `k`) | 1.64 | 1.78 | 1.58–1.99 |
| wB97M-V (`vv10`) | 3.09 | 2.91 | 2.59–3.25 |

Two further wB97M-V runs read about 30 % lower (2.06; 1.92–2.13) and were excluded under a contention rule
written after they had been seen; their re-runs on quiet cards read 2.96 and 2.87–3.05, but on shared GPUs
expect the lower readings. On six molecules of 63–98 atoms, r2SCAN reads 1.93 (1.80 without one anomalous
stock wall) and B3LYP 1.34, a pre-registered prediction missed (the largest size tier among the 24 reads
1.897). B3LYP's gain falls with size (at 475–559 Da the J/K stage, about half the stock SCF, gains only 5–7 %
while XC gains 2–3×); that explains part of the drop, the rest is unexplained.

Negative result on FP64-strong cards. An L40S behaves like the Blackwell card (r2SCAN 1.73, B3LYP 1.52,
wB97M-V 2.99). An H100 does not: wB97M-V 0.58–0.61 (slower than stock), B3LYP 0.91–1.02, and r2SCAN's
1.32–1.40 comes almost entirely from the FP64 AO cache (the FP32 switch alone reads 1.08); our pre-registered
predictions for H100 B3LYP and wB97M-V, 1.20 and 1.35, were far off. On an A100, 11 of 12 clean cells were
neutral or slower. The code does not detect the device; see question 2.

Accuracy (18 cells, three functionals × six molecules, mixed vs stock from the same guess and settings):
gradients agree to 3.8e-9 – 2.7e-7 Ha/Bohr and dipoles to within 9.3e-6 D in 17 of 18 cells, with identical
cycle counts. The exception, omeprazole B3LYP, took one extra cycle and sits 4.4e-6 Ha/Bohr and 1.9e-4 D from
stock, but 3.4× closer than stock to a tight reference (`conv_tol` 1e-11), against which every |ΔE| is ≤
1.6e-10 Ha and both arms carry the same 6.6e-7 – 3.4e-6 Ha/Bohr residual set by the default `conv_tol_grad`.
One disclosure: the run holding the 12 r2SCAN and B3LYP cells failed one pre-registered gate, fluconazole
r2SCAN's dipole against that reference (1.004e-4 D on a 1e-4 D ceiling, a residual stock itself carries), so
under our own rules the package labels those 12 cells as not quotable.

Geometry optimisation (geomeTRIC, stock gradients, one card, one run per arm): paracetamol r2SCAN, paracetamol
B3LYP and celecoxib r2SCAN take the same 23, 26 and 16 steps as stock, end within 9.9e-6 Å of stock's endpoint
and pass a fresh stock SCF and gradient there; warm-step SCFs run 1.41–1.44 and whole optimisations 1.19–1.30
faster (the latter from an arm whose fork-only warm-start rule, not proposed, never fired; the plain arm paid
the driver JIT in one cell and reads 1.16 and 1.27 where valid).

Correctness of the branch itself (`d642f3d` on `e1020c3`, built from source on a Blackwell card at pinned
versions, under pyscf 2.8, and on a V100 with the single-GPU CI package set, libxc 0.9.0): upstream
`test_df_rks`, `test_df_jk`, `test_rks` and `test_scf` give 44 passed and 1 skipped on master and on the
branch in all three profiles, the 56 new tests pass (136 s on the V100; skipped on multi-GPU hosts as
upstream's are), and stock vs mixed on three molecules agree to |ΔE| ≤ 5.9e-12 Ha with identical cycles.

## Limitations

- Library defaults were measured only on three molecules, one card and def2-mTZVPP; everything else is in the
  measured configuration above. Without cuTENSOR, speed was not measured (correctness was).
- UKS, multi-GPU, gradients and Hessians are out of scope (refused or untouched); range-separated functionals
  with `xc`/`k` are refused, and NLC functionals are handled only by the separate `vv10`.
- Speed on the RTX PRO 6000 (Workstation and Server), L40S, H100 and A100, correctness also on a V100; not the
  RTX 5090, B200 or consumer cards. Two FP64:FP32 throughput levels only, so there is no threshold to gate on.

## Questions

0. Do you want this in core at all? If not, would you accept two seams so the rest can live outside the tree
   as an `RKS` subclass: `check_convergence` parity in `scf._kernel` (`soscf` honours `mf.check_convergence`;
   `_kernel` does not) and the keywords `nr_nlc_vxc(vv10_kernel=)` / `_vv10nlc(uwe_kernel=)` in `numint`,
   default `None`?
1. API shape: a policy object, or a flag on `density_fit()` / `RKS`? The object keeps the switch settings and
   the record in one place; a `__config__` default could sit beside it.
2. Device guard: on the H100 and A100 the FP32 switch is slower than stock. Should the mode warn or refuse by
   compute capability (sm_80/sm_90 class), or only document it?
3. Scope of a first PR: XC + K alone (+1033/−17 over six files, 27 tests; applies on master alone), with VV10
   and the FP64 AO cache as follow-ups? Without the cache: 1.63 / 1.95 / 2.88 on the unpruned grid, and 1.58 /
   1.64 at the defaults for XC + K (three molecules); the cache is a small part of the gain.
4. Convergence contract: is "two consecutive FP64 iterations before convergence is accepted", enforced in
   `scf._kernel`, acceptable there, or should it live in the RKS classes?
5. Defaults and the record: are `xc_switch_tol = k_switch_tol = 1e-3` and a test tolerance of |ΔE| ≤ 1e-8 Ha
   against stock acceptable (the benchmarks used `3e-4` for B3LYP; at `1e-3` the switch came after 8 FP32 XC
   calls instead of 9–11)? Is `mf.mixed_precision_record` (a dict, `None` before `kernel()`) with two new
   `KohnShamDFT._keys` entries fine, or would you prefer the logger only?

## Links

- Code: [`chris-lee-mc/gpu4pyscf@mixed-precision-upstream`](https://github.com/chris-lee-mc/gpu4pyscf/tree/mixed-precision-upstream), head `d642f3da`, four commits on master `e1020c3`, +3054/−39 over ten files.
- Evidence: [`chris-lee-mc/gpu4pyscf@mixed-precision-evidence`, `benchmarks/mixed_precision/`](https://github.com/chris-lee-mc/gpu4pyscf/tree/mixed-precision-evidence/benchmarks/mixed_precision):
  `native/verify_native.py` recomputes every benchmark figure above from the per-run data; `native/CLAIMS.md`
  states what the data do and do not support; `validation/` holds the GPU validation records; `README.md`
  documents the API, the refusals and the VV10 and AO-cache details left out here.
- A PR is drafted and ready to open once the scope is agreed.
