# Paste-ready PR body for split issue delivery

## Motivation
- Deliver each issue track independently while keeping modules standalone and importable.
- Ensure each file can be reviewed/merged against its own GitHub issue.

## Linked issues
Fixes #<issue-kelly>
Fixes #<issue-sharpe-tests>
Fixes #<issue-transaction-costs>
Fixes #<issue-ruin-probability>
Fixes #<issue-cpcv>

## File-to-issue mapping
- #<issue-kelly>: `src/bet_sizing/kelly.py`, `tests/test_kelly.py`
- #<issue-sharpe-tests>: `src/statistics/sharpe_tests.py`, `tests/test_sharpe_tests.py`
- #<issue-transaction-costs>: `src/costs/transaction_costs.py`, `tests/test_transaction_costs.py`
- #<issue-ruin-probability>: `src/risk/ruin_probability.py`, `tests/test_ruin.py`
- #<issue-cpcv>: `src/cross_validation/cpcv.py`, `tests/test_cpcv.py`

## Transfer to GitHub
1. Push branch:
   - `git push -u origin <branch-name>`
2. Open PR:
   - `gh pr create --title "<title>" --body-file docs/issue_split_pr_body.md`
3. Replace placeholder issue numbers before submitting.
4. On merge, GitHub auto-closes linked issues via `Fixes #...`.
