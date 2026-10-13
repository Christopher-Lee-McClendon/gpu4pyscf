#!/usr/bin/env python3
"""Check every number in rfc/RFC_ISSUE.md, rfc/ISSUE_DRAFT.md and rfc/PR_BODY.md against the evidence
package. ISSUE_DRAFT.md is the posting-ready text; RFC_ISSUE.md is its reviewed source, so the draft is
bound to the subset of the source's headline figures that it still prints.

Stdlib only. Two tiers, both fail closed:

  BOUND     headline figures, each tied by name to one check of native/verify_native.py. The
            figure must appear verbatim in the file, and the verifier's recomputed value (from
            data/, not from CLAIMS.md; the unformatted value where the verifier exposes it, so a
            two-decimal figure in the issue is rounded from the data, not from a printed
            three-decimal string) must round to it at the figure's printed precision.
  COVERED   every other numeric token in the two files must equal, at its printed precision,
            a number printed in native/CLAIMS.md, a value recomputed by verify_native.py, or an
            entry of ALLOWLIST (code constants, test counts and the GPU validation run, each with
            its source). A token covered by nothing is printed as UNVERIFIED and fails the run.

Also fails on: a BOUND figure missing from its file (a stale binding), a forbidden string (model
names, secrets), a mention of the rejected geometry-safety options without "not proposed" on the
same line, and a verify_native.py that does not itself pass.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
NATIVE = os.path.join(PKG, "native")
FILES = ["RFC_ISSUE.md", "ISSUE_DRAFT.md", "PR_BODY.md"]

# --- tier A: headline figures bound to verify_native.py checks (check name -> figure as printed)
BOUND = {
    "RFC_ISSUE.md": [
        # the issue quotes speed ratios to two decimals (the package keeps three); same() rounds the
        # verifier's recomputed value to the printed precision, so a 2-decimal figure binds directly
        ("LD trio geomean r2scan mixed-default/stock-default", "1.58"),
        ("LD trio geomean b3lyp mixed-default/stock-default", "1.64"),
        ("LD trio geomean wb97m-v mixed-default/stock-default", "3.09"),
        ("L12 r2scan ladder geomean (24)", "1.66"),
        ("L12 b3lyp ladder geomean (24)", "1.78"),
        ("wB97M-V ladder geomean (24, healthy pods)", "2.91"),
        ("L12 r2scan lowest cell", "1.52"),
        ("L12 r2scan highest cell", "1.87"),
        ("L12 b3lyp lowest cell", "1.58"),
        ("L12 b3lyp highest cell", "1.99"),
        ("wB97M-V lowest cell", "2.59"),
        ("wB97M-V highest cell", "3.25"),
        # the two excluded wB97M-V pods (CLAIMS C1) and their re-runs
        ("C0 wb97m-v mixed/stock (CONTENTION-UNKNOWN)", "2.06"),
        ("W4 (CONTENDED) diphenhydramine", "1.92"),
        ("W4 (CONTENDED) propranolol", "2.13"),
        ("C0w wB97M-V mixed/stock", "2.96"),
        ("W4r diphenhydramine", "2.87"),
        ("W4r ketoprofen", "3.05"),
        ("LD trio geomean r2scan mixed-nocache-campaign/stock-campaign", "1.63"),
        ("LD trio geomean b3lyp mixed-nocache-campaign/stock-campaign", "1.95"),
        ("LD trio geomean wb97m-v mixed-nocache-campaign/stock-campaign", "2.88"),
        ("XL r2SCAN geomean (6)", "1.93"),
        ("XL r2SCAN geomean without ritonavir (5)", "1.80"),
        ("XL B3LYP geomean (6)", "1.34"),
        ("L40S r2scan mixed/stock", "1.73"),
        ("L40S b3lyp mixed/stock", "1.52"),
        ("L40S wb97m-v mixed/stock", "2.99"),
        ("H100 r2scan mixed/stock", "1.32"),
        ("X3r H100 trio r2scan mixed/stock", "1.40"),
        ("H100 b3lyp mixed/stock", "0.91"),
        ("X3r H100 trio b3lyp mixed/stock", "1.02"),
        ("H100 wb97m-v mixed/stock", "0.58"),
        ("X3r H100 trio wb97m-v mixed/stock", "0.61"),
        ("H100 r2scan mixed/cache (precision effect)", "1.08"),
        # wB97M-V at the default tolerance: PREREG-4 D2a-c (PASS, single runs), not re-measured
        ("wb97m-v: lowest max |g_M - g_S| (Ha/Bohr)", "1.2e-8"),
        ("wb97m-v: highest max |g_M - g_S| (Ha/Bohr)", "3.2e-8"),
        ("wb97m-v: highest max |mu_M - mu_S| (D)", "5.9e-12"),
        # r2SCAN and B3LYP at the default tolerance: PREREG-10 DR1 (PASS, triplicate), the quotable
        # replacement for D1's UNQUOTABLE single-run cells (N12, from the raw vectors)
        ("DR1 r2SCAN (6 cells): lowest max |g_Mk - g_Sk| over k (Ha/Bohr)", "3.2e-9"),
        ("DR1 r2SCAN (6 cells): highest max |g_Mk - g_Sk| over k (Ha/Bohr)", "6.5e-8"),
        ("DR1 r2SCAN (6 cells): lowest max |mu_Mk - mu_Sk| over k (D)", "2.2e-7"),
        ("DR1 r2SCAN (6 cells): highest max |mu_Mk - mu_Sk| over k (D)", "2.9e-6"),
        ("DR1 B3LYP but omeprazole (5 cells): lowest max |g_Mk - g_Sk| over k (Ha/Bohr)", "2.0e-8"),
        ("DR1 B3LYP but omeprazole (5 cells): highest max |g_Mk - g_Sk| over k (Ha/Bohr)", "2.7e-7"),
        ("DR1 B3LYP but omeprazole (5 cells): lowest max |mu_Mk - mu_Sk| over k (D)", "2.0e-7"),
        ("DR1 B3LYP but omeprazole (5 cells): highest max |mu_Mk - mu_Sk| over k (D)", "1.0e-5"),
        ("DR1 omeprazole B3LYP: highest over k of max |g_M - g_S| (2 s.f.)", "4.4e-6"),
        ("DR1 omeprazole B3LYP: highest over k of max |mu_M - mu_S| (2 s.f.)", "1.9e-4"),
        ("DR1 omeprazole B3LYP: lowest over k of max |g_M - g_R| (2 s.f.)", "1.0e-6"),
        ("DR1 omeprazole B3LYP: highest over k of max |g_M - g_R| (3 s.f.)", "1.05e-6"),
        ("DR1 omeprazole B3LYP: highest over k of max |g_S - g_R| (2 s.f.)", "3.4e-6"),
        ("DR1 omeprazole B3LYP: lowest over k of max |mu_M - mu_R| (2 s.f.)", "3.5e-5"),
        ("DR1 omeprazole B3LYP: highest over k of max |mu_M - mu_R| (2 s.f.)", "3.6e-5"),
        ("DR1 omeprazole B3LYP: highest over k of max |mu_S - mu_R| (3 s.f.)", "1.57e-4"),
        ("DR1: largest |E_Mk - E_Sk| over 36 pairs (Ha)", "1.7e-10"),
        ("DR1: pairs with equal S and M cycle counts (of 36)", "33"),
        ("DR1: C1 pairs passing (of 36)", "36"),
        ("DR1: largest C2 LHS/bound ratio over 12 cells, gradient and dipole", "0.56"),
        ("DR1 stock repeat scatter, r2SCAN: max over k != k' of max |g_Sk - g_Sk'| (Ha/Bohr)", "3.2e-12"),
        ("DR1 stock repeat scatter, B3LYP: max |g_Sk - g_Sk'| (Ha/Bohr)", "5.6e-8"),
        # quotable accuracy at conv_tol_grad 3e-6 (PREREG-8 A1/A2, pods PASS): leads the section
        ("r2scan: lowest max |g_Mt - g_St|", "1.5e-9"),
        ("r2scan: highest max |g_Mt - g_St|", "2.3e-8"),
        ("b3lyp: lowest max |g_Mt - g_St|", "6.8e-9"),
        ("b3lyp: highest max |g_Mt - g_St|", "2.2e-7"),
        ("wb97m-v: lowest max |g_Mt - g_St|", "1.1e-8"),
        ("wb97m-v: highest max |g_Mt - g_St|", "2.1e-8"),
        ("r2scan: highest max |mu_Mt - mu_St| (D)", "1.4e-6"),
        ("b3lyp: highest max |mu_Mt - mu_St| (D)", "8.7e-6"),
        ("wb97m-v: highest max |mu_Mt - mu_St| (D)", "4.5e-12"),
        ("G1 paracetamol/r2scan: warm-step SCF wall, S / M0", "1.41"),
        ("G1 celecoxib/r2scan: warm-step SCF wall, S / M0", "1.44"),
        ("G1 paracetamol/b3lyp: whole optimisation (SCF + gradient), S / MW", "1.19"),
        ("G1 celecoxib/r2scan: whole optimisation (SCF + gradient), S / MW", "1.30"),
        # the plain arm, where valid (CLAIMS C6 Scope); 1.165 and 1.270 at three decimals
        ("G1 paracetamol/b3lyp: whole optimisation (SCF + gradient), S / M0 (the plain arm)", "1.16"),
        ("G1 celecoxib/r2scan: whole optimisation (SCF + gradient), S / M0 (the plain arm)", "1.27"),
        ("G1 paracetamol/r2scan: endpoint RMSD M0 vs S, aligned (A)", "9.9e-6"),
    ],
    "PR_BODY.md": [
        ("LD trio geomean r2scan mixed-default/stock-default", "1.577"),
        ("LD trio geomean b3lyp mixed-default/stock-default", "1.636"),
        ("LD trio geomean wb97m-v mixed-default/stock-default", "3.094"),
        ("LD trio geomean r2scan mixed-nocache-campaign/stock-campaign", "1.627"),
        ("LD trio geomean b3lyp mixed-nocache-campaign/stock-campaign", "1.954"),
        ("LD trio geomean wb97m-v mixed-nocache-campaign/stock-campaign", "2.884"),
        ("LD trio geomean r2scan mixed-campaign/stock-campaign", "1.678"),
        ("LD trio geomean b3lyp mixed-campaign/stock-campaign", "1.968"),
        ("LD trio geomean wb97m-v mixed-campaign/stock-campaign", "2.944"),
        ("anchor delta b3lyp: LD mixed-campaign/stock-campaign trio minus the banked 1.860", "0.108"),
        ("anchor delta wb97m-v: LD mixed-campaign/stock-campaign trio minus the banked 2.803", "0.141"),
        ("LD trio r2scan: stock-default/stock-campaign (the grid, on stock)", "1.356"),
        ("LD trio b3lyp: stock-default/stock-campaign (the grid, on stock)", "1.278"),
        ("LD trio wb97m-v: stock-default/stock-campaign (the grid, on stock)", "1.035"),
        ("LD trio b3lyp: mixed-campaign/mixed-nocache-campaign (the cache)", "1.007"),
        ("L12 r2scan ladder geomean (24)", "1.657"),
        ("L12 b3lyp ladder geomean (24)", "1.783"),
        ("wB97M-V ladder geomean (24, healthy pods)", "2.914"),
        ("XL r2SCAN geomean (6)", "1.931"),
        ("XL B3LYP geomean (6)", "1.338"),
        ("L40S r2scan mixed/stock", "1.729"),
        ("L40S b3lyp mixed/stock", "1.524"),
        ("L40S wb97m-v mixed/stock", "2.994"),
        ("H100 r2scan mixed/stock", "1.320"),
        ("H100 b3lyp mixed/stock", "0.905"),
        ("H100 wb97m-v mixed/stock", "0.577"),
        ("H100 r2scan mixed/cache (precision effect)", "1.084"),
        ("wb97m-v: lowest max |g_M - g_S| (Ha/Bohr)", "1.2e-8"),
        ("wb97m-v: highest max |g_M - g_S| (Ha/Bohr)", "3.2e-8"),
        # r2SCAN and B3LYP at the default tolerance: PREREG-10 DR1 (PASS, triplicate; N12)
        ("DR1 r2SCAN (6 cells): lowest max |g_Mk - g_Sk| over k (Ha/Bohr)", "3.2e-9"),
        ("DR1 r2SCAN (6 cells): highest max |g_Mk - g_Sk| over k (Ha/Bohr)", "6.5e-8"),
        ("DR1 r2SCAN (6 cells): highest max |mu_Mk - mu_Sk| over k (D)", "2.9e-6"),
        ("DR1 B3LYP but omeprazole (5 cells): lowest max |g_Mk - g_Sk| over k (Ha/Bohr)", "2.0e-8"),
        ("DR1 B3LYP but omeprazole (5 cells): highest max |g_Mk - g_Sk| over k (Ha/Bohr)", "2.7e-7"),
        ("DR1 B3LYP but omeprazole (5 cells): highest max |mu_Mk - mu_Sk| over k (D)", "1.0e-5"),
        ("DR1 omeprazole B3LYP: highest over k of max |g_M - g_S| (2 s.f.)", "4.4e-6"),
        ("DR1 omeprazole B3LYP: highest over k of max |mu_M - mu_S| (2 s.f.)", "1.9e-4"),
        ("DR1 omeprazole B3LYP: lowest over k of max |g_M - g_R| (2 s.f.)", "1.0e-6"),
        ("DR1 omeprazole B3LYP: highest over k of max |g_M - g_R| (3 s.f.)", "1.05e-6"),
        ("DR1 omeprazole B3LYP: highest over k of max |g_S - g_R| (2 s.f.)", "3.4e-6"),
        ("DR1 omeprazole B3LYP: lowest over k of max |mu_M - mu_R| (2 s.f.)", "3.5e-5"),
        ("DR1 omeprazole B3LYP: highest over k of max |mu_M - mu_R| (2 s.f.)", "3.6e-5"),
        ("DR1 omeprazole B3LYP: highest over k of max |mu_S - mu_R| (3 s.f.)", "1.57e-4"),
        ("DR1: largest |E_Mk - E_Sk| over 36 pairs (Ha)", "1.7e-10"),
        ("DR1: pairs with equal S and M cycle counts (of 36)", "33"),
        ("r2scan: lowest max |g_Mt - g_St|", "1.5e-9"),
        ("r2scan: highest max |g_Mt - g_St|", "2.3e-8"),
        ("b3lyp: lowest max |g_Mt - g_St|", "6.8e-9"),
        ("b3lyp: highest max |g_Mt - g_St|", "2.2e-7"),
        ("wb97m-v: lowest max |g_Mt - g_St|", "1.1e-8"),
        ("wb97m-v: highest max |g_Mt - g_St|", "2.1e-8"),
        ("G1 paracetamol/r2scan: endpoint RMSD M0 vs S, aligned (A)", "9.9e-6"),
        ("G1 paracetamol/r2scan: warm-step SCF wall, S / M0", "1.41"),
        ("G1 celecoxib/r2scan: warm-step SCF wall, S / M0", "1.44"),
        ("G1 paracetamol/b3lyp: whole optimisation (SCF + gradient), S / MW", "1.19"),
        ("G1 celecoxib/r2scan: whole optimisation (SCF + gradient), S / MW", "1.30"),
        ("G1: largest |E_stock(mixed endpoint) - E_S(final)| (Ha)", "3.2e-9"),
        ("G1 paracetamol/b3lyp: whole optimisation (SCF + gradient), S / M0 (the plain arm)", "1.165"),
        ("G1 celecoxib/r2scan: whole optimisation (SCF + gradient), S / M0 (the plain arm)", "1.270"),
        ("wb97m-v: highest max |mu_M - mu_S| (D)", "5.9e-12"),
        # quotable accuracy at conv_tol_grad 3e-6 (PREREG-8 A1/A2, pods PASS): leads the section
        ("r2scan: lowest max |g_Mt - g_St|", "1.5e-9"),
        ("r2scan: highest max |g_Mt - g_St|", "2.3e-8"),
        ("b3lyp: lowest max |g_Mt - g_St|", "6.8e-9"),
        ("b3lyp: highest max |g_Mt - g_St|", "2.2e-7"),
        ("wb97m-v: lowest max |g_Mt - g_St|", "1.1e-8"),
        ("wb97m-v: highest max |g_Mt - g_St|", "2.1e-8"),
        ("r2scan: highest max |mu_Mt - mu_St| (D)", "1.4e-6"),
        ("b3lyp: highest max |mu_Mt - mu_St| (D)", "8.7e-6"),
        ("wb97m-v: highest max |mu_Mt - mu_St| (D)", "4.5e-12"),
        ("C0 wb97m-v mixed/stock (CONTENTION-UNKNOWN)", "2.059"),
        ("W4 (CONTENDED) diphenhydramine", "1.92"),
        ("W4 (CONTENDED) propranolol", "2.13"),
        ("C0w wB97M-V mixed/stock", "2.963"),
        ("W4r diphenhydramine", "2.87"),
        ("W4r ketoprofen", "3.05"),
        ("CC1 stock: G4-MPS", "1.52"),
        ("CC1 mixed: G4-MPS", "1.75"),
        ("CS1 P1_stock_first: r2scan cold extra (s)", "134.0"),
    ],
}

# ISSUE_DRAFT.md prints every headline figure of RFC_ISSUE.md except these: it gives the default-tolerance
# accuracy in one paragraph (the r2SCAN/B3LYP triplicate's gradient ranges and dipole ceilings, omeprazole
# B3LYP, |dE| and cycles), so the dipole floors, the wB97M-V row and the C2 / scatter details are not
# printed there. Deriving the list
# from RFC_ISSUE.md's keeps the draft bound to the source: a figure revised in the source becomes a stale
# binding in the draft until the draft is revised too.
_NOT_IN_DRAFT = {
    # wB97M-V at the default tolerance (D2): the draft gives only the r2SCAN/B3LYP triplicate there
    "wb97m-v: lowest max |g_M - g_S| (Ha/Bohr)",
    "wb97m-v: highest max |g_M - g_S| (Ha/Bohr)",
    "wb97m-v: highest max |mu_M - mu_S| (D)",
    # the triplicate (DR1) in the draft's one paragraph: gradient ranges, dipole ceilings, omeprazole's
    # M-S and gradient M-R / S-R, |dE|, cycles; not the dipole floors, omeprazole's dipole M-R / S-R,
    # the C2 ratio or stock's scatter
    "DR1 r2SCAN (6 cells): lowest max |mu_Mk - mu_Sk| over k (D)",
    "DR1 B3LYP but omeprazole (5 cells): lowest max |mu_Mk - mu_Sk| over k (D)",
    "DR1 omeprazole B3LYP: lowest over k of max |mu_M - mu_R| (2 s.f.)",
    "DR1 omeprazole B3LYP: highest over k of max |mu_M - mu_R| (2 s.f.)",
    "DR1 omeprazole B3LYP: highest over k of max |mu_S - mu_R| (3 s.f.)",
    "DR1: largest C2 LHS/bound ratio over 12 cells, gradient and dipole",
    "DR1 stock repeat scatter, r2SCAN: max over k != k' of max |g_Sk - g_Sk'| (Ha/Bohr)",
    "DR1 stock repeat scatter, B3LYP: max |g_Sk - g_Sk'| (Ha/Bohr)",
}
BOUND["ISSUE_DRAFT.md"] = [b for b in BOUND["RFC_ISSUE.md"] if b[0] not in _NOT_IN_DRAFT]
assert len(BOUND["ISSUE_DRAFT.md"]) == len(BOUND["RFC_ISSUE.md"]) - len(_NOT_IN_DRAFT), \
    "a _NOT_IN_DRAFT name is not a RFC_ISSUE.md binding"

# --- tier B allowlist: numbers that are facts of the code, the tests or the GPU validation run,
# not of the benchmark data. Each with its source. Nothing else is allowed outside CLAIMS.md and
# the verifier's recomputed values.
ALLOWLIST = {
    # gpu4pyscf/dft/mixed_precision.py @ cb1a2d7 (code lines identical to d642f3da's): MixedPrecision
    # defaults and module constants
    "1e-3": "mixed_precision.py: XC_SWITCH_TOL = K_SWITCH_TOL = 1e-3",
    "2": "mixed_precision.py: SWITCH_STALL = 2; also 'two consecutive iterations' (fp64_tail); '2×T4'",
    "30": "mixed_precision.py: SWITCH_CALL_CAP = 30",
    "0.7": "mixed_precision.py: AO_CACHE_MEM_FRACTION = 0.7",
    # gpu4pyscf/dft/vv10_mixed.py @ cb1a2d7
    "1e-5": "vv10_mixed.py: VV10_SWITCH_TOL = 1e-5; upstream tdscf/ris.py warns below conv_tol 1e-5",
    "1e-10": "vv10_mixed.py: VVDF_REL = 1e-10 (certificate band, relative)",
    "1e-11": "vv10_mixed.py: VVDF_DENLC = 1e-11 (certificate band, Ha); tests: SAME = 1e-11",
    "48": "vv10_mixed.py / mixed_precision.py docstring: df64 is ~48-bit",
    # tests @ cb1a2d7
    "1e-8": "test_mixed_precision.py: ETOL = 1e-8",
    "38": "test_mixed_precision.py @ cb1a2d7: 38 test methods (26 KnownValues + 11 AOCache + 1 MultiGPU)",
    "18": "test_mixed_precision_vv10.py @ cb1a2d7: 18 test methods; also CLAIMS C4 (18 cells)",
    "56": "38 + 18 new tests; g4psrc r6 new_tests 56 passed, 0 skipped on all three profiles (g4psrc_r6_*.json)",
    "27": "git show 85a7395:gpu4pyscf/dft/tests/test_mixed_precision.py: 26 KnownValues + 1 MultiGPU tests in the first commit",
    # git show --numstat 85a7395 (the XC + K commit): +1033 / -17 over 6 files
    "1033": "git show --numstat 85a7395: total insertions of the XC + K commit",
    "17": "git show --numstat 85a7395: -17; CLAIMS C4: 17 of 18 cells",
    # git diff --numstat 38df3b7 cb1a2d7 (10 files; per file identical to e1020c3..d642f3da)
    "3054": "git diff --numstat 38df3b7 cb1a2d7: total insertions",
    "39": "git diff --numstat 38df3b7 cb1a2d7: total deletions (39 on code, 0 on tests/README)",
    "2115": "git diff --numstat 38df3b7 cb1a2d7, gpu4pyscf/ minus tests: insertions",
    "993": "numstat 38df3b7..cb1a2d7: dft/mixed_precision.py +993",
    "915": "numstat 38df3b7..cb1a2d7: dft/vv10_mixed.py +915",
    "80": "numstat 38df3b7..cb1a2d7: df/df_jk.py +80",
    "3": "numstat: df/df_jk.py -3; CLAIMS: '3 cards', '3 %'",
    "51": "numstat 38df3b7..cb1a2d7: dft/rks.py +51",
    "4": "numstat: dft/rks.py -4; RFC: 'four commits'",
    "29": "numstat 38df3b7..cb1a2d7: dft/numint.py +29",
    "20": "numstat: dft/numint.py -20; CLAIMS: '20–44 atoms'",
    "41": "numstat 38df3b7..cb1a2d7: scf/hf.py +41",
    "12": "numstat: scf/hf.py -12; CLAIMS: '12 readings'",
    "6": "numstat 38df3b7..cb1a2d7: scf/soscf.py +6 (the newton refusal, in 85a7395); six questions; CLAIMS: six molecules",
    "555": "numstat 38df3b7..cb1a2d7: dft/tests/test_mixed_precision.py +555",
    "383": "numstat 38df3b7..cb1a2d7: dft/tests/test_mixed_precision_vv10.py +383",
    "1": "numstat: README.md +1; g4psrc r6: 1 skipped; CLAIMS: 'above 1'",
    # g4psrc r6 (three profiles on fork cb1a2d7, master 38df3b7, built from source);
    # validation/VALIDATION-g4psrc.md and g4psrc_r6_*.json carry the records
    "38333841493": "g4psrc r6 pro6000-lock run id (engine repo; RUN_ID=38333841493-1-d35ca7a0)",
    "38336271676": "g4psrc r6 pro6000-pyscf28 run id (engine repo; RUN_ID=38336271676-1-d35ca7a0)",
    "38335226746": "g4psrc r6 v100-ci run id (engine repo; RUN_ID=38335226746-1-d35ca7a0)",
    "44": "g4psrc r6: upstream regression 44 passed (+ 1 skipped) on both trees, all profiles; CLAIMS: '20–44 atoms'",
    "5.9e-12": "g4psrc r6 pro6000-lock: trio max |dE| 5.91e-12 Ha (g4psrc_r6_pro6000-lock.json)",
    "4.1e-12": "g4psrc r6 pro6000-pyscf28: trio max |dE| 4.09e-12 Ha (g4psrc_r6_pro6000-pyscf28.json)",
    "132": "g4psrc r6 v100-ci: new_tests wall_s 132.4 (g4psrc_r6_v100-ci.json)",
    "123": "g4psrc r6 pro6000-lock and pro6000-pyscf28: new_tests wall_s 122.6 / 123.5, reported as 123 s in the engine repo's r6 RESULT (g4psrc_r6_*.json)",
    "18.5": "g4psrc r6 v100-ci: slowest test 18.51 s, test_vv10_paracetamol (durations in g4psrc_r6_v100-ci.json)",
    "16.8": "g4psrc r6 v100-ci: second slowest 16.83 s, test_kernels_against_stock_vv10nlc (g4psrc_r6_v100-ci.json)",
    "11": "g4psrc r6 v100-ci: third slowest 10.81 s, so 'the rest under 11 s'; CLAIMS C3: 11 of 12 A100 clean cells",
    "2.8": "upstream unittest.yml multi-gpu job: pip install pyscf==2.8; g4psrc r6 pro6000-pyscf28 ran 2.8.0",
    "0.9": "upstream unittest.yml single-gpu job: gpu4pyscf-libxc-cuda12x 0.9.0; g4psrc r6 v100-ci ran it",
    "1.17": "upstream requirements / multi-gpu job: scipy 1.17, not tested (r4 Amendment 1; r5 and r6 kept 1.15.3)",
    # version strings: the measurement stack (requirements.lock) and upstream's recommended cuTENSOR pairings
    "14.1": "requirements.lock / ENVIRONMENT.md: cupy-cuda12x==14.1.1",
    "2.0": "upstream README: CuPy 13.3.0 + cuTENSOR 2.0.2, the pairing pyscf/gpu4pyscf#518 recommends",
    "2.2": "upstream README: CuPy 13.4.1 + cuTENSOR 2.2.0",
    # RFC prose
    "5": "CLAIMS C1: 5 healthy pods",
    "24": "CLAIMS C1: 24 molecules",
}

FORBIDDEN = ["claude", "opus", "sonnet", "haiku", "fable", "anthropic", "gpt-", "openai", "runpod_api", "api_key",
             "api key", "secret"]
GEO_OPTIONS = ["diis_reset_at_switch", "warm_start_gorb", "DIIS reset", "warm-start rule"]

NUM = re.compile(r"(?<![A-Za-z0-9_.#/+])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# a git SHA: 7-40 hex characters with at least one letter (so a long run id is still a number)
SHA = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")


def present(printed, text):
    """The figure appears in the text as a whole number, not as part of a longer one."""
    return re.search(r"(?<![\d.])" + re.escape(printed) + r"(?!\d|\.\d|[eE][-+]?\d)", text) is not None


def precision(s):
    """('sci', significant figures) or ('fixed', decimals) of a printed number."""
    s = s.lstrip("+-")
    if "e" in s.lower():
        mant = s.lower().split("e")[0]
        return "sci", len(mant.replace(".", "").lstrip("0") or "0")
    return "fixed", (len(s.split(".")[1]) if "." in s else 0)


def same(printed, value):
    kind, p = precision(printed)
    try:
        v = float(value)
        x = float(printed)
    except ValueError:
        return False
    if kind == "sci":
        return f"{v:.{max(p - 1, 0)}e}" == f"{x:.{max(p - 1, 0)}e}"
    return f"{v:.{p}f}" == f"{x:.{p}f}"


def tokens(text):
    out = []
    for ln, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("<!--") or line.strip().startswith("-->"):
            continue
        scrub = DATE.sub(" ", line)
        scrub = SHA.sub(" ", scrub)
        scrub = re.sub(r"^\s*\d+\.\s", " ", scrub)          # ordered-list markers
        scrub = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " ", scrub)
        for m in NUM.finditer(scrub):
            tok = m.group(0).lstrip("+")
            out.append((ln, tok, line.strip()))
    return out


def main():
    sys.path.insert(0, NATIVE)
    cwd = os.getcwd()
    os.chdir(NATIVE)
    try:
        import verify_native as vn          # runs every check at import
        integrity = vn.data_integrity()
    finally:
        os.chdir(cwd)
    if integrity:
        for p in integrity:
            print(f"FAIL data  {p}")
        return 1
    failures = 0
    by_name = {}
    for claim, what, got, banked, ok in vn.results:
        by_name[what] = (got, ok, claim)
        if not ok and (claim, what) not in vn.KNOWN_DISCREPANCIES:
            print(f"FAIL native  {claim} {what}: {got} (banked {banked})")
            failures += 1
    if len(vn.results) != vn.EXPECTED_CHECKS:
        print(f"FAIL native  ran {len(vn.results)} checks, expected {vn.EXPECTED_CHECKS}")
        failures += 1

    # numbers CLAIMS.md prints, and numbers the verifier recomputed (as printed strings)
    with open(os.path.join(NATIVE, "CLAIMS.md")) as fh:
        claims_tokens = {t for _, t, _ in tokens(fh.read())}
    verifier_values = [got for got, ok, claim in by_name.values() if got not in ("True", "False")]

    for name in FILES:
        path = os.path.join(HERE, name)
        with open(path) as fh:
            text = fh.read()
        low = text.lower()
        print(f"\n== {name}")
        for word in FORBIDDEN:
            if word in low:
                print(f"FAIL forbidden string {word!r} in {name}")
                failures += 1
        # the rejected geometry-safety options may be mentioned only as "not proposed", in the
        # same paragraph (a blank-line-delimited block) as the mention
        for para in re.split(r"\n\s*\n", text):
            p = " ".join(para.split()).lower()
            if any(g.lower() in p for g in GEO_OPTIONS) and "not proposed" not in p:
                print(f"FAIL {name}: geometry-safety option mentioned without 'not proposed': "
                      f"{p[:90]!r}")
                failures += 1

        # tier A
        n_bound = 0
        for check, printed in BOUND.get(name, []):
            if check not in by_name:
                print(f"FAIL bound  no verify_native check named {check!r}")
                failures += 1
                continue
            got, ok, claim = by_name[check]
            if not present(printed, text):
                print(f"FAIL bound  {printed!r} ({check}) is not in {name}: stale binding")
                failures += 1
                continue
            if not same(printed, getattr(vn, "raw", {}).get(check, got)):
                print(f"FAIL bound  {name}: {printed} vs verifier {got} ({claim} {check})")
                failures += 1
                continue
            n_bound += 1
        print(f"bound: {n_bound}/{len(BOUND.get(name, []))} headline figures match "
              f"verify_native.py recomputed values")

        # tier B
        bound_printed = {p for _, p in BOUND.get(name, [])}
        covered = unverified = 0
        seen = set()
        for ln, tok, line in tokens(text):
            if tok in bound_printed or tok in ALLOWLIST:
                covered += 1
                continue
            # the same number at the same printed precision: 1.66 does not pass on 1.657
            if any(same(tok, c) and precision(tok) == precision(c) for c in claims_tokens):
                covered += 1
                continue
            if any(same(tok, v) and precision(tok) == precision(v) for v in verifier_values):
                covered += 1
                continue
            unverified += 1
            if (ln, tok) not in seen:
                seen.add((ln, tok))
                print(f"UNVERIFIED {name}:{ln}: {tok}    | {line[:100]}")
        print(f"covered: {covered} numeric tokens (CLAIMS.md, verifier values or ALLOWLIST); "
              f"unverified: {unverified}")
        failures += unverified

    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
