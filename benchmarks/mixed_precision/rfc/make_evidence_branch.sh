#!/usr/bin/env bash
# Build the public, evidence-only branch that the RFC issue links to.
#
# It is upstream master (BASE) plus benchmarks/mixed_precision/ from this package branch,
# WITHOUT rfc/ (staging texts, checklists, plans) and the superseded PR_DESCRIPTION.md.
# One commit; re-running it replaces the branch with a fresh build. It never touches the
# package branch or the code branch.
#
# usage: make_evidence_branch.sh <base-sha> [branch]   (run from the repository root)
set -euo pipefail
BASE=${1:?upstream master SHA the code branch is on}
BRANCH=${2:-mixed-precision-evidence}
SRC=$(git rev-parse HEAD)
WT=$(mktemp -d)
git worktree add -q --detach "$WT" "$BASE"
trap 'git worktree remove --force "$WT"' EXIT
git archive "$SRC" benchmarks/mixed_precision | tar -x -C "$WT"
rm -rf "$WT/benchmarks/mixed_precision/rfc" "$WT/benchmarks/mixed_precision/PR_DESCRIPTION.md"
if grep -rIl -E 'rfc/|PR_DESCRIPTION|RFC_ISSUE|PR_BODY|SUBMISSION_PLAN' "$WT/benchmarks/mixed_precision"; then
  echo "evidence still references the staging files above; fix them on the package branch" >&2
  exit 1
fi
# -f: upstream's .gitignore has `*.csv` without the package branch's exceptions for
# benchmarks/mixed_precision, so a plain `git add -A` here silently drops every data file
# (the first build, 798714d, shipped no CSV at all). git archive only ships tracked files,
# so nothing ignorable but the data is extracted.
git -C "$WT" add -A -f benchmarks/mixed_precision
git -C "$WT" -c core.hooksPath=/dev/null commit -q -m "benchmarks/mixed_precision: evidence for the mixed-precision RFC

Built from package branch commit $SRC on upstream $BASE by
benchmarks/mixed_precision/rfc/make_evidence_branch.sh (staging texts excluded)."
NEW=$(git -C "$WT" rev-parse HEAD)
# The committed tree must be the package archive minus the two exclusions, file for file.
if ! diff <(git ls-tree -r --name-only "$SRC" benchmarks/mixed_precision \
             | grep -v -E '^benchmarks/mixed_precision/(rfc/|PR_DESCRIPTION\.md$)' | sort) \
          <(git ls-tree -r --name-only "$NEW" benchmarks/mixed_precision | sort); then
  echo "committed tree differs from the package archive (above); not updating $BRANCH" >&2
  exit 1
fi
# Verify the COMMIT, not the working tree: a fresh archive of it must pass both checkers.
CHK=$(mktemp -d)
git archive "$NEW" benchmarks/mixed_precision | tar -x -C "$CHK"
( cd "$CHK" && python3 benchmarks/mixed_precision/native/verify_native.py | tail -1 \
  && python3 benchmarks/mixed_precision/verify_aggregates.py | tail -1 )
rm -rf "$CHK"
git branch -f "$BRANCH" "$NEW"
echo "$BRANCH -> $(git rev-parse --short "$BRANCH") (base $BASE, source $SRC); $(git ls-tree -r --name-only "$NEW" benchmarks/mixed_precision | grep -c '\.csv$') csv files"
