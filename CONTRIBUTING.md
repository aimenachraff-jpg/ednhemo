# Contributing

Thanks for your interest in `ednhemo`. This is a research project, but issues
and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/aimenachraff-jpg/ednhemo.git
cd ednhemo
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
```

The checkpoints are tracked in `src/ednhemo/weights/` (~60 MB), so a clone is
enough to run the full test suite.

## Before opening a pull request

```bash
ruff check .
pytest
```

- Keep the public API backward compatible; this project follows SemVer.
- Add a test for every behavior change.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Do not commit patient data. Use synthetic volumes or the public BHSD dataset.

## Reporting a bug

Open an issue with:

- the command or code you ran,
- the full traceback,
- the input type (NIfTI / PNG, shape, spacing) — **never attach real patient
  data**, use a synthetic or public scan.

## Medical disclaimer

`ednhemo` is not a medical device. Contributions must not claim clinical
validity or diagnostic performance beyond the published research results.
