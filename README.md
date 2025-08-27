# Garden Planner

Camera-perspective garden bed visualizer with CSV import/export and bloom-time filtering.

![CI](https://github.com/EternalAmaiti/garden-planner/actions/workflows/ci.yml/badge.svg)

## Quickstart (Poetry)
```bash
# 1) Install Poetry (if needed): https://python-poetry.org/docs/#installation
# 2) From the project root:
poetry install
poetry run pre-commit install
poetry run pytest
poetry run garden-planner --help
```

## Usage examples

Generate a fresh plan, export CSV, and render PNGs (orthographic + camera views):
```bash
poetry run garden-planner --density 0.6 --halos
```

Load an existing plan from CSV and render only blooming species in March:
```bash
poetry run garden-planner --from-csv camera_perspective_v6_5_YYYYMMDD_HHMMSS_plan.csv --month March --halos
```

Output files are timestamped (PNG + CSV).

Write all outputs into a dedicated folder
```bash
poetry run garden-planner --density 0.6 --halos --output-dir out
```

## Project layout
```
src/garden_planner/visualizer.py   # main logic + CLI
tests/                             # pytest tests
```

## Tooling
- **Poetry** for dependency management
- **Ruff** + **Black** for lint/format
- **Pytest** for tests
- **pre-commit** to run checks before each commit

## Troubleshooting

- **Headless warnings (`FigureCanvasAgg is non-interactive`)**  
  Expected in CI or when `MPLBACKEND=Agg`. Files still save. Use `maybe_show()` or unset `MPLBACKEND`.

- **Git push rejects `.github/workflows` changes**  
  Use a Personal Access Token with `workflow` + `public_repo` (or `repo`) scopes.

- **Poetry says metadata is deprecated**  
  This repo uses PEP 621 `[project]`. Run `poetry lock && poetry install` after metadata changes.

- **Windows line endings (CRLF) warnings**  
  Harmless. You can `git config --global core.autocrlf true`. Optional `.gitattributes` below.

## License
MIT — feel free to use and adapt.
