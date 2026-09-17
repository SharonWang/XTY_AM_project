# Stage 2 Donor Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add validated primary and scale-sensitivity donor plots for Stage 2.

**Architecture:** Public validation wrappers consume donor-level Stage 2 summaries and call the supplied plotting implementations without altering their visual encodings. A corrected private significance helper supplies conventional asterisk labels.

**Tech Stack:** Python, pandas, NumPy, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-stage2-plots-design.md`

## Global Constraints

- Modify only the D-drive project repository.
- Keep one `scripts/*.py` source file.
- Preserve the supplied visual behavior and macaron colors.
- Use donor-level summaries; never treat cores or cells as replicates in these plots.
- Update README and proposal with the source API.

### Task 1: Plot contracts and statistical guards

**Files:**
- Modify: `tests/test_stage2_spatial.py`
- Modify: `scripts/xty_am_pipeline.py`

**Interfaces:**
- Produces: `plot_stage2_primary(...) -> (fig, axes, plot_data)`.
- Produces: `plot_stage2_scale_sensitivity(...) -> (fig, axes, summary)`.

- [ ] Add tests for expected returns, conventional significance labels, method/scale selection, missing columns, duplicate donor rows, duplicate test rows, and file saving.
- [ ] Run focused tests and confirm they fail because the APIs are absent.
- [ ] Add validated public plotting functions and preserve the supplied visual bodies.
- [ ] Run focused tests and confirm they pass.

### Task 2: Documentation and release validation

**Files:**
- Modify: `README.md`
- Modify: `XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md`
- Modify: `scripts/xty_am_pipeline.py`

**Interfaces:**
- Exports and documents both new plotting functions.

- [ ] Update `__all__`, README function catalogue, proposal status/methods/change log.
- [ ] Run syntax compilation, focused tests, full tests, and `git diff --check`.
- [ ] Commit, obtain independent review, merge to main, rerun tests, and push only to the explicitly approved remote.
