# Spatial AM–AT2 Ligand–Receptor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add validated Spatial CellChat export and donor-aware continuous spatial LR co-occurrence analysis.

**Architecture:** Reuse existing FDR and stable-seed helpers without overwriting them. Prepare bounded per-core arrays before parallel work, return schema-stable tables, and separate R export from exploratory Python inference.

**Tech Stack:** Python, AnnData, NumPy, pandas, SciPy sparse/IO/spatial/statistics, joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-spatial-lr-design.md`

## Global Constraints

- Modify only the D-drive repository and retain one `scripts/*.py` source file.
- Use donor-level inference; core P values remain diagnostic.
- Do not overwrite existing `_safe_spearman` or `_bh_adjust` helpers.
- Update README and proposal with every public function and limitation.

### Task 1: Grouping, panel audit, and CellChat export

**Files:** `tests/test_spatial_lr.py`, `scripts/xty_am_pipeline.py`

- [ ] Add failing tests for tie-safe balanced groups, CellChat labels, sparse export, manifest contents, invalid expression/coordinates/metadata, and simple-pair panel audit.
- [ ] Implement the grouping, audit, and export functions with NumPy-style docstrings.
- [ ] Run focused tests until green.

### Task 2: Core spatial LR calculations

**Files:** `tests/test_spatial_lr.py`, `scripts/xty_am_pipeline.py`

- [ ] Add failing tests for uniform/Gaussian neighborhoods, both directions, empirical P values, prevalence, score-signature overlap, empty schemas, validation, and order-stable seeds.
- [ ] Implement array-level and all-core calculations without capturing AnnData in workers.
- [ ] Correct FDR within core/direction/radius and run focused tests until green.

### Task 3: Donor inference and documentation

**Files:** `tests/test_spatial_lr.py`, `scripts/xty_am_pipeline.py`, `README.md`, `XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md`

- [ ] Add failing tests for equal-core donor Fisher-z summaries, donor uniqueness, two-sided Wilcoxon tests, family-specific FDR, aliases, docstrings, and exports.
- [ ] Implement donor summarization/testing and update documentation/version history.
- [ ] Run compilation, focused tests, full tests, `git diff --check`, and one-source audit.
- [ ] Commit, obtain independent review, merge locally, rerun tests, and push only to the explicitly approved remote.
