# Garden Planner

Camera-perspective garden bed visualizer with CSV import/export and bloom-time filtering.

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

## License
MIT — feel free to use and adapt.
