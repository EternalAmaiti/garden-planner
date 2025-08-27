# Contributing

Thanks for your interest in improving Garden Planner!

## Development environment
```bash
poetry install
poetry run pre-commit install
```

## Running tests
```bash
poetry run pytest
```

## Commit messages
Use **Conventional Commits**, e.g.:
- `feat: add CSV export for clusters`
- `fix: prevent overlap in layout step`
- `docs: update README with usage`

## Code style
- Black for formatting
- Ruff for lint and import sorting
- Docstrings: Google style preferred

## Releasing
- Update `CHANGELOG.md`
- Bump version in `pyproject.toml`
