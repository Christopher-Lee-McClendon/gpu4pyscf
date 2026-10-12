# Staging checklist for submitting the mixed-precision RFC to pyscf/gpu4pyscf

For the person who will post. Nothing in this directory has been posted. The order matters:
**the issue first, the PR only after the maintainers have answered.**

State of play (updated 2026-10-12):

| item | value |
|---|---|
| code branch | `chris-lee-mc/gpu4pyscf@mixed-precision-upstream`, head `d642f3da971650d24dc147e212ce932d03757811`, four commits (`9b129af` XC + K, `301cfe9` VV10, `5e81866` AO cache, `d642f3d` docstring and README; +3054/−39 over ten files) on upstream master `e1020c32ae789cd8218c333e1ee2fc74348a9841` |
| review fixes on the code branch (validated by r5) | copyright `2026` on the four new files; `mf.newton()` with a policy set refuses (`NotImplementedError`, `scf/soscf.py`); `k=True` on a functional without exact exchange refuses (was a silent no-op); `mf.mixed_precision_record` defaults to `None`; with `xc` and `k` both on, K is held at FP64 whenever XC falls back to FP64; six new tests (56 in all: 38 + 18) |
| evidence package (staging) | `chris-lee-mc/gpu4pyscf@mixed-precision-rfc-package-v2`, `benchmarks/mixed_precision/`, with `rfc/` (this directory). Not linked from the texts |
| evidence branch (the one the issue links) | `chris-lee-mc/gpu4pyscf@mixed-precision-evidence`: upstream `e1020c3` plus `benchmarks/mixed_precision/` without `rfc/` and `PR_DESCRIPTION.md`, built by `rfc/make_evidence_branch.sh`. **The pushed `798714d` (built from `3e8c5f7`) carries none of the 56 `*.csv` files** (upstream's `.gitignore` dropped them; the script now forces the add and checks the committed tree) and must be rebuilt and force-pushed before posting (§5) |
| upstream master | `e1020c3`, re-checked with `git ls-remote` on 2026-10-12 after r5: unchanged |
| last GPU validation of the code branch | g4psrc r5, engine-repo runs `38198811227` (`pro6000-lock`), `38201109296` (`pro6000-pyscf28`) and `38200149447` (`v100-ci`), on master `e1020c3`, from source for sm_120 and sm_70: PASS (`validation/VALIDATION-g4psrc.md`, `g4psrc_r5_*.json`). The pre-review head `3d9de5d` passed run `38194143624` and is superseded |
| earlier heads, kept as branches | `mixed-precision-upstream-on-e1020c3-prereview` (`3d9de5d`), `mixed-precision-upstream-on-c1a6e37-r4` (r4, `4d2f3de`), `mixed-precision-upstream-on-c1a6e37` (r3, `ac4c668e`), `mixed-precision-upstream-on-82bc702` (r2), `mixed-precision-upstream-on-eef5f5b` (r1) |

## 1. Base: re-verify, rebase if needed, re-validate

- [x] Rebased onto `c1a6e37` (no conflicts; the branch adds no `eval_ao` call, so `a2f3c36`'s API
      change does not touch it) and re-validated as g4psrc r3: regression 44 passed + 1 skipped on
      both trees, 49/49 new tests, VV10 bitwise identity, trio \|ΔE\| ≤ 6.4e-12 Ha with
      identical cycles.
- [x] Tests-only change for upstream's CI (multi-GPU skips, `MultiGPU` refusal test; head
      `4d2f3de4`, same base) re-validated as g4psrc r4 on three profiles: `pro6000-lock` (as r3:
      44 passed + 1 skipped on both trees, 50/50, trio \|ΔE\| ≤ 5.9e-12 Ha), `pro6000-pyscf28`
      (pyscf 2.8: the same gates, \|ΔE\| ≤ 2.7e-12 Ha) and `v100-ci` (one V100, sm_70, libxc
      0.9.0: regression, 50/50 in 133 s, VV10 bitwise identity; no trio). `PR_BODY.md`,
      `RFC_ISSUE.md` and `verify_rfc.py` cite r4.
- [x] `validation/VALIDATION-g4psrc.md` with `g4psrc_r2.json`, `g4psrc_r3.json` and the three
      `g4psrc_r4_*.json` records added; `VALIDATION-vv10.md`'s "not validated" line points to it.
- [x] Base re-verified on 2026-10-12: upstream master was `e1020c3` (five commits past `c1a6e37`,
      none touching the branch's files; #960 makes `test_nr_coach` skip on older pyscf / libxc).
      Rebased onto it with the review fixes (head `d642f3d`).
- [x] Review fixes (newton refusal, non-hybrid `k` refusal, record default `None`, K held at FP64
      on an XC fallback, six new tests, copyright `2026`) re-validated as g4psrc r5 on three
      profiles: `pro6000-lock` (run `38198811227`: 44 passed + 1 skipped on both trees, 56/56 in
      136 s, trio \|ΔE\| ≤ 5.9e-12 Ha with identical cycles, VV10 certificate ok ×3),
      `pro6000-pyscf28` (run `38201109296`: the same gates, 56/56 in 123 s, \|ΔE\| ≤ 5.0e-12 Ha)
      and `v100-ci` (run `38200149447`: libxc 0.9.0, 44 passed + 1 skipped on both trees, 56/56 in
      136 s, slowest 18.4 s and 16.9 s, VV10 bitwise identity; no trio). The superseded pre-review
      run `38194143624` (`3d9de5d`) is recorded as such. `validation/g4psrc_r5_*.json` added;
      `PR_BODY.md`, `RFC_ISSUE.md` and `verify_rfc.py` cite r5.
- [x] Re-checked on 2026-10-12 after r5 and the second review: `git ls-remote
      https://github.com/pyscf/gpu4pyscf.git refs/heads/master` still returns `e1020c3`.
- [ ] Immediately before posting: `git fetch upstream && git log --oneline e1020c3..upstream/master`
      (this was `c1a6e37..upstream/master` before r5). If the list is non-empty, rebase again,
      re-run g4psrc (the engine repo pins `BASE_SHA` in `.github/scripts/runpod_g4psrc.py`, with a
      dated section in the port plan before dispatch),
      update the run id, base and numbers in `PR_BODY.md` and `RFC_ISSUE.md`, add them to
      `verify_rfc.py`'s `ALLOWLIST`, and re-run it.

## 2. Lint, exactly as upstream's `.github/workflows/lint.yml` runs it

From the root of the code branch:

```
pip install ruff==0.15 "flake8>=3.7.0"
ruff check --config .ruff.toml --unsafe-fixes gpu4pyscf
ruff check --select NPY --ignore NPY002 gpu4pyscf
flake8 --config .flake8 gpu4pyscf
```

- [x] All three clean on the whole `gpu4pyscf` tree at `ac4c668e` (ruff 0.15, flake8 7.3.0), and
      again at `4d2f3de4` after the tests-only change (ruff 0.15.0, flake8 7.3.0, from a
      `git archive` of that commit), and at `d642f3da` after the review fixes (ruff 0.15.0,
      flake8 7.4.1, from a `git archive` of that commit).
- [ ] Re-run if the branch changes again.

## 3. Licence headers and CLA

- [ ] Upstream has no CLA file and `CONTRIBUTING.md` mentions none. Watch for a CLA bot on the PR.
- [x] Every file upstream carries the Apache-2.0 header `# Copyright <years> The PySCF Developers.
      All Rights Reserved.` Files added to upstream in 2026 (for example
      `gpu4pyscf/df/tests/test_df_ao2mo.py`, `gpu4pyscf/pbc/dft/gks.py`) use `# Copyright 2026 The
      PySCF Developers`. The four new files on the branch carried `2021-2026` up to `4d2f3de4`;
      they carry `2026` at `d642f3da` (the existing edited files keep their own years). Lint and
      g4psrc r5 were re-run after the change.
- [ ] `CONTRIBUTING.md` asks for sm_70 compatibility and for tests against CPU PySCF values. The
      VV10 kernels are NVRTC-compiled C++14 (`cupy.RawModule`, options `--std=c++14`, no fast
      math); the only device intrinsics in `vv10_mixed.py` are `__fadd_rn`, `__fsub_rn`,
      `__fmul_rn`, `__syncthreads` and `__longlong_as_double`, and there is no `__CUDA_ARCH__`
      branching, so nothing requires sm_80 or newer. They were run only on sm_120 (and the L40S,
      H100 and A100 pods of the campaign). `test_against_cpu_pyscf` covers the second point for
      `xc`/`k`. State both in the PR if asked.

## 4. Consistency of the staged texts (found while preparing them; fix before posting)

- [x] `PR_DESCRIPTION.md` is marked as superseded by `rfc/PR_BODY.md` (it describes the v1.8.1
      branch: "21 tests", "+914 / −18"); its dead `DISCUSSION-DRAFT` reference now points to
      `rfc/RFC_ISSUE.md`. `rfc/make_evidence_branch.sh` excludes it from the evidence branch, so
      it never reaches the branch the issue links; it may stay here.
- [x] `README.md` now says 47 pods (44 plus PREREG-9's LD1 and LD2 and PREREG-10's DR1), matching
      `native/README.md` and `native/SOURCES.md`.
- [x] PREREG-10 DR1 (the default-tolerance r2SCAN/B3LYP cells in triplicate, PASS) banked as
      `native/data/dr1.csv` and checked by `verify_native.py` N12; the three RFC texts now quote it in
      place of D1's UNQUOTABLE cells, and `verify_rfc.py` binds the new figures to N12.
- [ ] **Engine-repo erratum to file (not done here).** `native/prereg/PREREG-rfcbench-10-downstream-triplicate.md`
      line 3 still reads "PRE-REGISTERED, not yet run" although its RESULT is appended, and the
      protocol is not yet merged into the engine repository's `main` (mirrored from `36004fe0`).
      As for PREREG-9: correct it there by dated erratum, merge, and re-mirror.
- [x] The `mixed_precision.py` docstring (amended into the fourth commit, then `4d2f3de4`, now
      `d642f3d`) now gives CLAIMS C3's figures (wB97M-V 1.6–2× slower; H100 B3LYP neutral to about 1.1× slower; A100
      11 of 12 clean cells neutral or slower), and says K is held to XC's switch only when `xc` and
      `k` are both on.
- [ ] The usage snippet in that docstring uses `gpu4pyscf.dft.RKS` (a factory function in
      `gpu4pyscf/dft/__init__.py` on `c1a6e37` and `e1020c3`); the tests and `rfc/` use
      `gpu4pyscf.dft.rks.RKS`. Both resolve; no action unless the rebase changes `dft/__init__.py`.
- [ ] **Engine-repo erratum to file (not done here).** `native/prereg/PREREG-rfcbench-9-library-defaults.md`
      line 3 still reads "PRE-REGISTERED, not yet run" although its RESULT is appended (from line
      275). `native/prereg/` is a byte-identical mirror of the engine repository's protocols, so it
      is **not** edited in this package: correct the status line in the engine repo (as a dated
      erratum, like PREREG-8 Erratum 1) and re-mirror.
- [x] `test_nr_coach` under libxc 0.9.0: the "unrelated to this change" note was dropped from
      `RFC_ISSUE.md` and `PR_BODY.md`, since upstream #960 (master `e1020c3`) now skips that test
      on older pyscf / libxc; `validation/VALIDATION-g4psrc.md` keeps r4's observation as a record
      with one line saying so.

## 5. Final verification before posting anything

- [ ] `python3 benchmarks/mixed_precision/native/verify_native.py` ends with
      `1024/1033 checks reproduced; 9 known discrepancies (listed above, not adjusted); 0 failures`
      (or a higher reproduced count if checks were added) and exits 0.
- [ ] `python3 benchmarks/mixed_precision/verify_aggregates.py` ends with
      `419 checks: 418 ok, 1 known DISCREPANCY (reported, not adjusted), 0 FAIL`.
- [ ] `python3 benchmarks/mixed_precision/rfc/verify_rfc.py` exits 0 with no `UNVERIFIED` line.
- [ ] On the package branch, this returns nothing:

      ```
      git grep -n -i -E 'claude|opus|sonnet|haiku|fable|gpt|anthropic|runpod_api|api_key' -- benchmarks/mixed_precision \
        ':!benchmarks/mixed_precision/native/provenance/harness' \
        ':!benchmarks/mixed_precision/rfc/CHECKLIST.md' \
        ':!benchmarks/mixed_precision/rfc/verify_rfc.py'
      ```

      The last two exemptions are the files that hold the pattern itself: this checklist, and
      `verify_rfc.py`'s `FORBIDDEN` list.
- [ ] On the code branch, grep only what the branch adds. Upstream's own sources already contain
      matches (for example, ten "Generated by ChatGPT" comments in `lib/ecp/cart2sph.cu` and
      `lib/gdft/nr_eval_gto.cu` on master). This returns nothing:

      ```
      git diff upstream/master mixed-precision-upstream | grep -i -E '^\+.*(claude|opus|sonnet|haiku|fable|gpt|anthropic|runpod_api|api_key)'
      ```
- [ ] **The harness copy is exempt on purpose.** `native/provenance/harness/` is a byte-for-byte
      snapshot of the benchmark harness at the commit that produced the data, so it is not edited.
      It holds 7 matches in 4 files, and none is a credential:
      - `rfcbench.yml` 1: the Actions secret's name, `${{ secrets.RUNPOD_API_KEY }}`;
      - `runpod_rfcbench.py` 4: the docstring naming the required variable, the line that reads it
        from the environment (`os.environ["RUNPOD_API_KEY"]`), and two comments citing the harness
        repository's `CLAUDE.md`;
      - `rfcbench_sets.py` 1 and `scfbench_requirements.lock` 1: comments citing that
        repository's `.claude/skills/` notes and `CLAUDE.md`.
      Check that nothing else has appeared. This should total 7, in those four files:
      `git grep -c -i -E 'claude|opus|sonnet|haiku|fable|gpt|anthropic|runpod_api|api_key' -- benchmarks/mixed_precision/native/provenance/harness`
- [x] **Public-branch hygiene: a clean evidence-only branch.** The branch the issue links must not
      carry the planning material (`rfc/SUBMISSION_PLAN.md`, this checklist, the superseded
      `PR_DESCRIPTION.md`). `rfc/make_evidence_branch.sh <base-sha>` builds `mixed-precision-evidence`
      as upstream master plus `benchmarks/mixed_precision/` without `rfc/` and `PR_DESCRIPTION.md`,
      refuses if any remaining file mentions the staging files, checks the committed tree file for
      file against the package archive, and runs both evidence checkers on an archive of the commit.
      `RFC_ISSUE.md` and `PR_BODY.md` link that branch (code: the `mixed-precision-upstream` tree;
      evidence: `benchmarks/mixed_precision/` on `mixed-precision-evidence`); the package branch
      is not linked. `verify_rfc.py` needs both `rfc/` and `native/`, so it runs here, on the
      package branch, never on the evidence branch.
- [ ] **After any change to the package, rebuild and push the evidence branch**, from the root
      of the package branch with the change committed:

      ```
      benchmarks/mixed_precision/rfc/make_evidence_branch.sh e1020c32ae789cd8218c333e1ee2fc74348a9841
      git push -f origin mixed-precision-evidence
      ```

      Then, on the pushed branch: `git ls-tree -r --name-only origin/mixed-precision-evidence
      benchmarks/mixed_precision | grep -c '\.csv$'` must read **57** (56 before PREREG-10's
      `dr1.csv` was banked; the first build, `798714d`,
      read 0: upstream's `.gitignore` has `*.csv` without the package branch's exceptions, and the
      script's `git add -A` dropped every data file; the script now adds with `-f` and verifies the
      commit), and `native/verify_native.py` and `verify_aggregates.py` must pass from a checkout
      of that branch. The two links in `RFC_ISSUE.md` and the one in `PR_BODY.md` must resolve.

## 6. Post the issue

- [ ] **`ISSUE_DRAFT.md` is the text to post**; `RFC_ISSUE.md` is its reviewed source (the same
      numbers, with the staging comment, the internal labels and the long form kept for the
      record). Both are checked by `verify_rfc.py`, the draft against the subset of the source's
      headline figures it prints. Edit a number in the source first, then in the draft.
- [ ] Open a GitHub issue on `pyscf/gpu4pyscf` with the body of `ISSUE_DRAFT.md` (everything after
      its `Title:` line and the blank line that follows). Title: "RFC: opt-in mixed-precision SCF for
      density-fitted RKS on FP64-limited GPUs". Upstream has no issue template.
- [ ] Attach nothing to the issue beyond the two links (the code branch and the evidence branch);
      the evidence branch is the attachment.
- [ ] **Wait for maintainer feedback.** The six questions (0–5) at the end of the issue decide the
      shape of the PR: whether it belongs in core at all or only as two seams (Q0, SUBMISSION_PLAN
      §5 Fallback A), policy object vs flag, device guard, whether VV10 and the AO cache go in the
      first PR, where the convergence check lives, and the default thresholds and the record.

## 7. Only then: the PR

- [ ] Apply the maintainers' answers to the code branch first (API shape, scope split, guard),
      re-run lint and the GPU validation, and update `PR_BODY.md` and `verify_rfc.py` to match.
- [ ] Open the PR from `mixed-precision-upstream` (rebased) against `pyscf/gpu4pyscf:master` with
      the body of `PR_BODY.md`, `#<issue>` filled in, and the HTML comment removed. Upstream has no
      PR template. The code branch contains only the ten files in the reviewer's guide; the
      evidence package stays on the fork's package branch and is linked, not included.
- [ ] If the maintainers asked for XC + K only, drop the VV10 commit (`301cfe9`) and the AO-cache
      commit (`5e81866`) from the PR branch, remove the VV10 section and the AO-cache rows from
      `PR_BODY.md`, and re-run `verify_rfc.py` (its `BOUND` table will then need the VV10 and
      cache entries removed, which is deliberate: every edit to a number goes through the checker).
