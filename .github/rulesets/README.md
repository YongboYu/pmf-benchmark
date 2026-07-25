# Branch rulesets

`main-protection.json` is a **reference export** of the repository ruleset that
protects `main`. GitHub does **not** apply rulesets from files in the repo — this
copy is version-controlled documentation so the protection rules are reviewable
and re-importable.

**Active rules on `main`:**

| Rule | Effect |
|------|--------|
| Require a pull request | Changes reach `main` only through a PR (0 approvals required) |
| Require status checks | `pre-commit`, `test`, `Analyze (python)` must pass (non-strict) |
| Require linear history | Merges must be **squash** (only squash is allowed) |
| Block force-pushes | `main` cannot be force-updated |
| Block deletion | `main` cannot be deleted |

Repository admins are a bypass actor.

**To re-import after editing** (requires admin):

```bash
gh api -X PUT repos/YongboYu/pmf-benchmark/rulesets/<id> --input .github/rulesets/main-protection.json
```
