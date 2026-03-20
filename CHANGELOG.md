# Change Log
All notable changes to this project will be documented in this file.

## Beta - 2026-01-27

### Added
- [x] Added Docker support to run PageIndex in a containerized environment.
- [x] Added `Dockerfile` for building a reproducible CLI image.
- [x] Added `docker-compose.yml` for simplified execution with volume mounting.
- [x] Added `.dockerignore` to optimize Docker image size and build performance.
- [x] Enabled execution of `run_pageindex.py` as the container entrypoint.

### Changed
- [x] Standardized execution environment using Python 3.11 slim image.
- [x] Improved dependency installation reliability for PDF processing libraries.

---

## Beta - 2025-04-23

### Fixed
- [x] Fixed a bug introduced on April 18 where `start_index` was incorrectly passed.

---

## Beta - 2025-04-03

### Added
- [x] Add node_id, node summary
- [x] Add document discription

### Changed
- [x] Change "child_nodes" -> "nodes" to simplify the structure
