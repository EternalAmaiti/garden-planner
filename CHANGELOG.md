# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `--output-dir` flag to route PNG/CSV writes to a chosen folder.
- Module-level output helpers: `set_output_dir()`, `ensure_output_dir()`, `out()`.
- Guarded display: `maybe_show()` avoids warnings on headless backends.

### Changed
- CLI stubs for strict interleaving present (no-op) to prevent runtime errors.
- Ruff config migrated to `[tool.ruff.lint]`; PEP 621 metadata under `[project]`.

## [0.1.0] - 2025-08-27
Released at 2025-08-27T11:35:00+02:00 (Europe/Berlin).
### Added
- Initial project scaffolding with Poetry, Ruff, Black, Pytest, and pre-commit.
- Imported visualizer script as `garden_planner.visualizer`.
- Basic tests and documentation.

## [0.1.1] - 2025-08-27
Released at 2025-08-27T14:35:00+02:00 (Europe/Berlin).
### Added
- `--output-dir` flag and output helpers.
### Fixed
- Headless `plt.show()` warnings via `maybe_show()`.
### Chore
- Ruff/Black config updates; PEP 621 migration.