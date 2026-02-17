# UV Package Manager Quick Reference

## Why UV

- Much faster dependency resolution and installation than plain `pip`
- Deterministic installs with lock support
- Works with `pyproject.toml`
- Compatible workflow for most pip-style tasks

## Environment management

```bash
# create virtual environment
uv venv

# activate
source .venv/bin/activate

# deactivate
deactivate
```

## Dependency management

```bash
# install project package (editable)
uv pip install -e .

# install dev extras
uv pip install -e ".[dev]"

# list installed packages
uv pip list

# upgrade project deps
uv pip install --upgrade -e .

# uninstall package
uv pip uninstall package-name
```

## Project tasks

```bash
# one-click setup script
./setup_uv.sh

# run commands inside uv environment without manual activation
uv run python run_pageindex.py --pdf_path tests/pdfs/q1-fy25-earnings.pdf
```

## Troubleshooting

### `uv: command not found`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
brew install uv
source ~/.zshrc
```

### Virtual env is not active

```bash
which python
# expected: .../PageIndex/.venv/bin/python
source .venv/bin/activate
```

### Rebuild clean environment

```bash
rm -rf .venv
uv venv
uv pip install -e .
```

## Checklist

- `which uv` returns a path
- `.venv` exists
- `which python` points to `.venv/bin/python`
- `uv pip list` runs successfully
- `python run_pageindex.py --help` works
