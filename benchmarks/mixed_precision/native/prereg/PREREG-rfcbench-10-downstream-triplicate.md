# PREREG rfcbench-10: downstream accuracy at the default `conv_tol_grad`, in triplicate

**Status 2026-10-12: PRE-REGISTERED, not yet run.** Governed by `PREREG-rfcbench-0-measurand.md`.
It freezes at the first PREREG-10 sentinel. Changes go only in dated amendments above RESULT,
written before the dispatch they govern.

**Owner decision of 2026-10-12.** Run one new pod over D1's twelve cells, in triplicate, under a new
protocol whose gates measure the claim.

## What this protocol is, and what it is not

- **It is a redesign made after seeing D1.** Every gate below was written with D1's sentinel and
  PREREG-8 A1's sentinel in hand. The values they gave are listed in the next section, and the gate
  parameters were chosen knowing them.
- **D1 stays FAIL on record.** PREREG-4 RESULT part 1 (run `37038641497`) failed its pre-registered
  dipole gate, max |μ_M − μ_R| ≤ 1e-4 D, on fluconazole r2SCAN (1.004e-4 D). Nothing here
  re-evaluates that gate, and nothing here changes PREREG-4.
- **D1's twelve cells stay UNQUOTABLE** under PREREG-0 Amendment 1, whatever this pod reads.
- **This is a new measurement, not a rescue of D1.** It runs on a new pod, with new gates and a new
  sentinel. Its numbers are quotable only under its own rules (Quotability, below), and they never
  replace D1's on record.
- **The wB97M-V cells are not re-measured.** They came from D2, which passed (PREREG-4 RESULT
  part 2).

## Why

PREREG-4 gated mixed (M) against a tight stock reference (R) with fixed ceilings. PREREG-4 Erratum 2
and PREREG-8 then showed that what those ceilings measured was **stock's own convergence residual**
at `conv_tol` 1e-9 and the library-default `conv_tol_grad` (√1e-9 = 3.16e-5), not the mode:

- In D1's failing cell, fluconazole r2SCAN, S−R is 9.93e-5 D and M−S is 1.05e-6 D. Stock itself
  sits within 1 % of the ceiling.
- Stock itself would breach the same ceiling on omeprazole B3LYP (S−R 1.57e-4 D).
- PREREG-8 A1/A2 measured mixed against stock at the default and at a tight `conv_tol_grad`, but
  in single runs.

The claim to be measured is narrower and testable: **at the default `conv_tol_grad`, mixed reaches
the same answer as stock from the same start (C1), and it is no further from the converged answer
than stock's own convergence residual allows (C2).** Three repeats show whether that holds run to
run, and they measure stock's own repeat scatter beside it.

## The values known when the gates were chosen

All values are max over atoms and components, from the raw vectors in D1's sentinel
(`ci-results/runs/rfcbench-3890e545d9cc-37038641497-1-3890e545.txt`, `RUN_ID=37038641497-1-3890e545`)
and A1's (`docs/upstream/rfcbench-data/prereg8/a1.txt`, run `37734904781`). R is the same
reference in both (`conv_tol` 1e-11, `conv_tol_grad` 3e-6). S and M are the same-guess stock and
mixed runs at the default `conv_tol_grad`. Gradients are in Ha/Bohr, dipoles in D.

| cell | S−R grad (D1) | M−R grad (D1) | S−R dip (D1) | M−R dip (D1) | M−S grad, D1 / A1 | M−S dip, D1 / A1 | cycles S/M, D1 · A1 | C2 grad, LHS/bound, D1 / A1 | C2 dip, LHS/bound, D1 / A1 | stock S2−S grad (A1) | stock S2−S dip (A1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paracetamol r2SCAN | 1.36e-06 | 1.37e-06 | 4.99e-05 | 5.00e-05 | 9.9e-09 / 9.3e-09 | 6.2e-07 / 6.0e-07 | 13/13 · 13/13 | 0.49 / 0.49 | 0.46 / 0.46 | 5.1e-13 | 1.7e-12 |
| propranolol r2SCAN | 7.03e-07 | 7.04e-07 | 2.13e-05 | 2.10e-05 | 3.8e-09 / 4.0e-09 | 3.0e-07 / 2.6e-07 | 13/13 · 13/13 | 0.47 / 0.47 | 0.40 / 0.40 | 1.1e-12 | 2.9e-12 |
| celecoxib r2SCAN | 1.15e-06 | 1.16e-06 | 6.98e-05 | 7.00e-05 | 1.4e-08 / 1.5e-08 | 5.0e-07 / 5.0e-07 | 14/14 · 14/14 | 0.49 / 0.49 | 0.47 / 0.47 | 1.3e-12 | 2.2e-12 |
| fluconazole r2SCAN | 1.95e-06 | 1.96e-06 | 9.93e-05 | 1.00e-04 | 1.3e-08 / 1.4e-08 | 1.1e-06 / 1.2e-06 | 13/13 · 13/13 | 0.49 / 0.49 | 0.48 / 0.48 | 7.2e-13 | 6.4e-12 |
| warfarin r2SCAN | 6.65e-07 | 6.55e-07 | 6.19e-05 | 6.43e-05 | 5.3e-08 / 5.2e-08 | 2.4e-06 / 2.4e-06 | 15/15 · 15/15 | 0.46 / 0.46 | 0.48 / 0.48 | 1.8e-12 | 7.2e-12 |
| omeprazole r2SCAN | 1.47e-06 | 1.49e-06 | 6.34e-05 | 6.46e-05 | 2.4e-08 / 2.6e-08 | 1.2e-06 / 1.2e-06 | 15/15 · 15/15 | 0.49 / 0.49 | 0.47 / 0.47 | 8.9e-13 | 9.9e-12 |
| paracetamol B3LYP | 1.33e-06 | 1.34e-06 | 4.81e-05 | 4.79e-05 | 2.1e-08 / 2.3e-08 | 2.0e-07 / 2.0e-07 | 12/12 · 12/12 | 0.48 / 0.49 | 0.45 / 0.45 | 7.5e-09 | 5.9e-13 |
| propranolol B3LYP | 7.20e-07 | 7.36e-07 | 1.89e-05 | 1.91e-05 | 4.5e-08 / 4.3e-08 | 1.0e-06 / 1.1e-06 | 12/12 · 12/12 | 0.48 / 0.47 | 0.40 / 0.40 | 2.3e-08 | 8.1e-13 |
| celecoxib B3LYP | 1.79e-06 | 1.84e-06 | 8.42e-05 | 8.49e-05 | 4.6e-08 / 3.4e-08 | 6.4e-07 / 5.1e-07 | 13/13 · 13/13 | 0.50 / 0.49 | 0.48 / 0.47 | 8.4e-09 | 9.1e-12 |
| fluconazole B3LYP | 6.88e-07 | 8.29e-07 | 4.23e-05 | 5.16e-05 | 1.4e-07 / 1.3e-07 | 9.3e-06 / 9.2e-06 | 13/13 · 13/13 | 0.56 / 0.55 | 0.55 / 0.54 | 5.4e-09 | 2.3e-12 |
| warfarin B3LYP | 1.70e-06 | 1.67e-06 | 2.94e-05 | 2.01e-05 | 2.7e-07 / 2.8e-07 | 9.3e-06 / 9.6e-06 | 14/14 · 14/14 | 0.48 / 0.48 | 0.29 / 0.29 | 2.9e-08 | 6.1e-12 |
| omeprazole B3LYP | 3.39e-06 | 1.00e-06 | 1.57e-04 | 3.48e-05 | 4.4e-06 / 4.5e-06 | 1.9e-04 / 1.9e-04 | 13/14 · 13/14 | 0.15 / 0.16 | 0.11 / 0.11 | 2.4e-08 | 2.4e-12 |

Also known:
- **D1's one gate FAIL:** fluconazole r2SCAN, max |μ_M − μ_R| = 1.004e-4 D against the 1e-4 D
  ceiling; S−R 9.93e-5 D; M−S 1.05e-6 D.
- **|E_M − E_S|** is ≤ 1.7e-10 Ha in every D1 and A1 cell (the largest is omeprazole B3LYP).
  Cycles are equal in 11 of 12 cells. Omeprazole B3LYP takes one more mixed cycle in both pods.
- **Omeprazole B3LYP** is the one cell where mixed and stock stop on different iterates. Mixed
  lands 3.4× closer to R than stock in D1 (gradient 1.00e-6 against 3.39e-6), and the raw M−S
  dipole is 1.9e-4 D in both pods.
- **What C2 would have read.** Applied to D1's and A1's single S and M, C2 passes every cell. The
  largest LHS/bound ratio is 0.56 (fluconazole B3LYP). Because M ≈ S in 11 of 12 cells, the ratio
  sits near 0.5 there by construction. **The gate was chosen knowing this.** It is a gate on a
  new measurement, not a test that D1's data could have failed.

## The fork commit measured

`chris-lee-mc/gpu4pyscf` @ `63af0568d4fd19935bef51b7fd71f161a9cee56f`, **D1's own tree** (D1's
sentinel line `FORK_SHA=63af0568d4fd19935bef51b7fd71f161a9cee56f`), the first entry of
`VALIDATED_FORK_SHAS`.

- **Why this tree.** The values above that the gates were chosen against come from D1 on this tree
  (A1 ran on `09271907`, which is this tree plus two opt-in fields, both off in S and M). Measuring
  D1's cells again on D1's tree changes only what the protocol changes.
- **Enforced.** `MODE_REQUIRED_TREE["downrep"]` is `63af0568`, as PREREG-9 does for `defaults`. The
  free runner refuses any other fork SHA, and the pod refuses any tree but `63af0568`'s.
- The timed build is the 1.8.1-overlay build, as for D1.

## Cells

- **D1's twelve cells:** paracetamol, propranolol, celecoxib, fluconazole, warfarin and omeprazole
  (PREREG-4's six) × {r2SCAN, B3LYP}. Cell kind `downrep`.
- **Settings as D1 (the campaign configuration, PREREG-0 §1):**
  - def2-mTZVPP / def2-tzvpp-jkfit;
  - grid level 3 with `grids.prune = None`;
  - cuTENSOR required and observed;
  - S_k and M_k: `conv_tol` 1e-9, `max_cycle` 60, the library-default `conv_tol_grad`, the default
    `init_guess`;
  - R: `conv_tol` 1e-11, `conv_tol_grad` 3e-6, `max_cycle` 100, the default `init_guess`;
  - M_k's policy is the campaign policy, exactly D1's M: r2SCAN `MixedPrecision(xc=True,
    ao_cache_fp64=True)`; B3LYP `MixedPrecision(xc=True, k=True, xc_switch_tol=3e-4,
    k_switch_tol=1e-3, ao_cache_fp64=True)`.
- The harness's downstream runner builds R, S and M this way. `downrep` reuses its pieces
  unchanged: `build_and_run` with run kind `R` or `speed`, and the campaign `policy_kwargs`.
- **Card:** RTX PRO 6000 Blackwell Workstation (`pro6000`), with no fallback.

## Runs per cell (seven, in this order)

| order | run | SCF | settings | k |
|---|---|---|---|---|
| 1 | `R` | stock | R (1e-11 / 3e-6 / 100) | — |
| 2 | `S1` | stock | campaign, default `conv_tol_grad` | 1 |
| 3 | `M1` | mixed | campaign, default `conv_tol_grad` | 1 |
| 4 | `M2` | mixed | campaign, default `conv_tol_grad` | 2 |
| 5 | `S2` | stock | campaign, default `conv_tol_grad` | 2 |
| 6 | `S3` | stock | campaign, default `conv_tol_grad` | 3 |
| 7 | `M3` | mixed | campaign, default `conv_tol_grad` | 3 |

- Each run is a fresh `pyscf.M` and a fresh `rks.RKS(...).density_fit(...)` from the default guess,
  in one process per functional, followed by `mf.dip_moment()` (Debye) and the stock FP64 analytic
  gradient, as in the downstream mode.
- **Pair order alternates:** S first in k = 1 and 3, M first in k = 2. No arm is always the one that
  runs later in the process, on a warmer memory pool.
- Every run records the energy, the cycles, the converged flag, the full gradient and the dipole.
  Each sentinel line `RFCBENCH_CELL` carries one run.
- **S′ is dropped.** It measured stock's dependence on the initial guess (PREREG-4 erratum 1). Stock's
  own scatter is now read from S_1..S_3 at the same guess.

## Gates (fail the pod; fixed now)

Per cell. max |Δ| is over all atoms and components.

- **C1, the claim, for every k = 1..3:**
  - S_k and M_k both converged;
  - |E_Mk − E_Sk| ≤ 1e-8 Ha;
  - |cycles_Mk − cycles_Sk| ≤ 1.
- **C2, convergence quality, the per-cell worst over k on both sides:**
  - max_k max |g_Mk − g_R| ≤ 2 · max_k max |g_Sk − g_R| + 1e-7 Ha/Bohr;
  - max_k max |μ_Mk − μ_R| ≤ 2 · max_k max |μ_Sk − μ_R| + 1e-5 D.
  - In words: mixed is no further from the converged answer than twice stock's own distance from
    it, plus a small floor.
  - **Why the floors.** If stock happens to land very close to R in some cell, twice its distance
    would be a bound far below anything the default `conv_tol_grad` controls. The floors keep the
    gate meaningful there. They are about 1/7 of the smallest gradient S−R (6.65e-7, warfarin r2SCAN) and
    about 1/2 of the smallest dipole S−R (1.89e-5, propranolol B3LYP) seen in D1.
  - **The factor 2 and both floors are fixed now.**
- **C3, the reference:** R converged. If R does not converge, the cell FAILs. **There is no
  fallback** (PREREG-4's fallback to S is not carried over).
- **Treatment observed** on every M_k, with the downstream mode's evidence checks:
  - the fit outcome is `TREATED`, read from the fork's own tier and note;
  - an FP64 tail;
  - FP32 used: at least one FP32 XC call and the last two XC calls FP64; for B3LYP also at least one
    FP32 K call, the last two K calls FP64, and the full K rebuild recorded;
  - the AO cache ran: tier `fp64+fp32`, every FP64 XC call from the cache, the FP32 mirror released
    on the XC switch call.
- **Observation, as in the downstream mode** (PREREG-0 Amendment 1):
  - the seven runs present, in the order above, with no error;
  - settings read back off `mf` on every run, R's against R's settings;
  - every M_k's recorded `policy` equals the repr of the campaign `MixedPrecision`, built from the
    real class on the pod; every stock run carries no policy and no record;
  - `ngrids`, `naux`, `nao` identical across the seven runs, and one `init_guess`, not `'atom'`;
  - no mixed state alive before any gradient;
  - every gradient and dipole present, rectangular and finite.

## Reported, not gated

Per cell, from the raw vectors:
- the range over k of max |g_Mk − g_Sk| and of max |μ_Mk − μ_Sk|, and |E_Mk − E_Sk| per k;
- **stock's own repeat scatter:** the max over k ≠ k′ of max |g_Sk − g_Sk′| and of
  max |μ_Sk − μ_Sk′|;
- S_k − R and M_k − R per k (gradient and dipole);
- C2's left side, bound and ratio;
- the cycle counts of all seven runs.

`rfcbench_extract.py` recomputes every reading and the C1/C2/C3 verdicts from the streamed records,
independently of the pod's judge. Any cell where the two disagree is listed, and the summary is then
UNQUOTABLE.

## Predictions (fixed now, with their reasons)

- **C1 passes in all 36 pairs, including omeprazole B3LYP.** In D1 and in A1, |E_M − E_S| ≤ 1.7e-10
  Ha in every cell, and omeprazole B3LYP's extra mixed cycle is one cycle, within C1's ±1.
- **C2 passes in all 12 cells.** It would have passed every D1 and A1 cell, with an LHS/bound ratio
  of at most 0.56. In omeprazole B3LYP mixed was 3.4× closer to R than stock in D1.
- **Omeprazole B3LYP's raw M−S dipole is above 1e-4 D.** D1 and A1 both read 1.9e-4 D, because mixed
  and stock stop on different iterates there. Predicted: in every k, max |μ_Mk − μ_Sk| is in
  (1e-4, 3.8e-4] D. This is **disclosed, not gated**, and the RFC states it beside the quotable
  figures.
- **Stock's repeat scatter**, from PREREG-8's S2 against S:
  - B3LYP: max |g_Sk − g_Sk′| ≤ 3e-8 Ha/Bohr in every cell (A1: 5.4e-9 to 2.9e-8, so this is at
    the edge);
  - r2SCAN: bit-identical or about 1e-12, read as ≤ 1e-11 Ha/Bohr in every cell (A1: 5.1e-13 to
    1.8e-12);
  - dipoles, both functionals: max |μ_Sk − μ_Sk′| ≤ 1e-10 D (A1: at most 9.9e-12).
- A prediction that misses is a MODEL-MISS. It is stated, and it does not fail the pod.

## Pre-registered falsifier

**If C2 fails in any cell**, the RFC keeps the tight-tolerance results (PREREG-8 A1/A2, Mt against
St at `conv_tol_grad` 3e-6) as its **only** quotable accuracy evidence, and it states the C2 failure
with the cell and the numbers. A C2 failure also fails the pod.

The falsifier is UNDECIDED on a pod without all twelve cells' records; the re-run rule below
applies. UNDECIDED is never read as NOT FIRED.

## Quotability

- **PASS, and neither DEGRADED nor CONTENDED:** the pod's M−S figures (range over k, per cell and
  per functional) become **the quotable default-tolerance accuracy for r2SCAN and B3LYP**.
  - DEGRADED is PREREG-0's speed-reading label. This pod has no speed reading, so it cannot be
    DEGRADED; the extractor still checks it.
  - CONTENDED is PREREG-0 Amendment 3's idle-power rule (> 200 W before the first group). A
    CONTENDED or CONTENTION-UNKNOWN pod is flagged, not quoted, and re-run once only after the owner
    says so.
- **FAIL:** the M−S figures are UNQUOTABLE, and the failure is reported with its cell and gate.
- **Re-run rule.** One reserve pod, for an **infrastructure failure only**: no sentinel, the card gate,
  a `ModuleNotFoundError`. It re-runs the pod unchanged. **A gate FAIL is a finding and is not re-run.**

## Pod and budget

| pod | mode | cells | estimate |
|---|---|---|---|
| DR1 | `downrep` | the twelve: six molecules × {r2SCAN, B3LYP} (2 groups) | 2762 s |

- **Total: 1 pod**, plus the reserve above. Dispatched alone, with nothing else queued or in
  progress. Inputs: `mode=downrep`, `fork_sha=63af0568d4fd19935bef51b7fd71f161a9cee56f`, `gpu`
  blank or `pro6000`, `mols`, `xcs` and `repeats` blank. The mode is fixed: the free runner refuses
  any `mols` or `xcs` filter, and any `repeats` (three repeats are pre-registered).
- **Estimate:** `rfcbench_sets.work_estimate_s` with the committed C0 calibration gives **2762 s**,
  against the 4140 s budget (`WORK_CAP` 4440 s − 300 s). It fits in one pod, so no split is needed.
  Per cell it budgets:
  - R at 1.6 stock walls (as the downstream mode);
  - each S_k at one stock wall, each M_k at one calibrated mixed wall;
  - each of the seven gradients at one stock wall.
- **For scale:** D1 (four runs per cell) estimated 1999 s and measured 643 s of group work (355 s
  r2SCAN, 288 s B3LYP). Seven runs per cell is 1.75× D1's, about 1130 s.
- The sentinel window is `tail_lines(12, 2)` = 516 lines. A host test drives the pod's real emitters
  at their all-FAIL worst case against it.

## Not measured

- wB97M-V (D2 passed; not re-measured).
- A tighter `conv_tol_grad` (PREREG-8 A1/A2 measured 3e-6).
- Other cards, bases or trees.

## Amendment rule

Dated, above RESULT, before the dispatch it governs.

## RESULT

### RESULT (2026-10-12): **PASS. C1, C2 and C3 hold in every cell; the falsifier did not fire**

**The run.**
- DR1: run `38280778155`, sentinel `ci-results/runs/rfcbench-16aa4e3a98d9-38280778155-1-16aa4e3a.txt`,
  `RUN_ID=38280778155-1-16aa4e3a` (line 2 checked). Repo `16aa4e3a`, fork `63af0568`, one RTX PRO
  6000 Blackwell Workstation Edition.
- `STATUS=PASS`, all 12 cells `OK` and `TREATED`. The extractor recomputed every gate from the
  streamed records and agrees with the pod in every cell.
- Not CONTENDED: idle power 75.6 W. The reserve pod was not used.

**Gates.**
- **C1: 36 of 36 pairs pass.** Every |E_Mk − E_Sk| ≤ 1.7e-10 Ha. Cycle counts are equal in 33 pairs.
  The 3 exceptions are omeprazole B3LYP, where M takes 14 cycles against S's 13 in every k, as in D1.
- **C2: 12 of 12 cells pass.** The largest ratio of the left side to its bound is 0.56 (fluconazole
  B3LYP), the same cell and value as the pre-dispatch table.
- **C3: R converged in all 12 cells.**

**Mixed vs stock at the default `conv_tol_grad`, range over the three repeats.** These are now
quotable; this pod is the measurement, and D1 remains FAIL.

| | max \|g_M − g_S\| (Ha/Bohr) | max \|μ_M − μ_S\| (D) |
|---|---|---|
| r2SCAN, 6 molecules | 3.2e-9 – 6.5e-8 | 2.2e-7 – 2.9e-6 |
| B3LYP, 5 molecules (not omeprazole) | 2.0e-8 – 2.7e-7 | 2.0e-7 – 1.0e-5 |
| omeprazole B3LYP | 4.4e-6 (4.39e-6 – 4.44e-6) | 1.9e-4 (1.92e-4 – 1.93e-4) |

- **Omeprazole B3LYP reproduces D1 three times over.** One extra mixed cycle. Mixed is 4.4e-6
  Ha/Bohr and 1.9e-4 D from stock, but closer to the converged reference than stock:
  - gradient: M−R 1.0e-6–1.05e-6, against S−R 3.38e-6;
  - dipole: M−R 3.5e-5–3.6e-5 D, against S−R 1.57e-4 D.

  The raw M−S dipole is above 1e-4 D in every k, as predicted, and is disclosed, not gated.
- **Stock's own repeat scatter.**
  - r2SCAN: ≤ 3.2e-12 Ha/Bohr and ≤ 1.9e-11 D.
  - B3LYP: ≤ 5.6e-8 Ha/Bohr (worst omeprazole) and ≤ 5.6e-12 D.

**Predictions.**
- **Met:**
  - C1 in all 36 pairs;
  - C2 in all 12 cells;
  - the omeprazole B3LYP dipole in (1e-4, 3.8e-4] D in every k;
  - r2SCAN scatter (gradient ≤ 1e-11, dipole ≤ 1e-10);
  - B3LYP dipole scatter ≤ 1e-10.
- **MODEL-MISS:** B3LYP stock gradient scatter ≤ 3e-8 Ha/Bohr. It read 5.6e-8, in omeprazole B3LYP.
  The protocol flagged this prediction as at its edge (A1's maximum was 2.9e-8). It is stock's own
  non-determinism, not a mixed-precision effect.

**What this establishes.**
- At gpu4pyscf's default convergence tolerance, mixed reproduces stock to the ranges above, with equal
  cycles in 33 of 36 pairs, and is never further from the converged answer than C2 allows.
- The ranges replace D1's UNQUOTABLE ones in the RFC. They agree with D1's single-run values (D1:
  r2SCAN 3.8e-9 – 5.3e-8, B3LYP but omeprazole 2.1e-8 – 2.7e-7, omeprazole 4.4e-6 / 1.9e-4).
- Scope: one card, the same six molecules and settings as D1, and r2SCAN and B3LYP only. wB97M-V's
  default-tolerance cells are D2's, which passed.
