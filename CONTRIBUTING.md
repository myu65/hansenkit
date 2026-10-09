# Contributing

Read [AGENTS.md](AGENTS.md), [data policy](docs/data-policy.md) and [evaluation contract](docs/evaluation.md).
Open one issue with a scientific/software question and explicit acceptance criteria. Link a focused
branch/PR to that issue, run the checks below, and merge after all CI jobs pass under the repository
owner's authorization. Keep experimental protocol and result changes reviewable separately.

```sh
uv sync --locked --extra lightgbm
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
```

Do not commit private data, model artifacts or unknown external tables. New datasets/models need
asset-specific clearance evidence, hashes, allowed uses and a ledger entry. Changing the feature
vocabulary requires a feature-version increment. Do not fit transforms/calibration on test data.
Do not publish weights or claim validated HSP accuracy without independent evidence and rights review.

CI covers Linux/Windows and Python 3.11/3.12, all default paths plus optional LightGBM, a synthetic
end-to-end CLI experiment and package build. GPU and external encoder extraction are not CI tasks.
The owner can add repository-specific review/branch rules later; no organization or global account
settings are required or changed by the initial setup.

Pytest temporary files stay in this checkout's ignored `.pytest_tmp`, avoiding inherited
permissions on a user's shared temporary folder. Use a different `--basetemp` for concurrent local runs.
