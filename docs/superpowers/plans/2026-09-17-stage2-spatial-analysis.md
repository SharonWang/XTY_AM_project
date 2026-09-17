# Stage 2 AM–AT2 Spatial Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add validated Stage 1 plotting, parallel per-core execution, and donor-aware Stage 2 AM MHCII–AT2 spatial analyses to the reusable Xenium pipeline.

**Architecture:** Keep the user-requested single Python source file and reuse the existing spatial adjacency, permutation, metadata, and FDR helpers. One generic lightweight per-core parallel runner supports Stage 1 and Stage 2; Stage 2 functions share strict data preparation and return schema-stable core tables before donor-level inference.

**Tech Stack:** Python, AnnData, NumPy, pandas, SciPy, matplotlib, joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-stage2-spatial-analysis-design.md`

## Global Constraints

- Modify only files inside `D:\Xiaonan\CODEX_projects\Xiaotong_AM\XTY_AM_project`.
- Keep exactly one source Python file under `scripts/*.py`.
- Preserve the supplied Stage 1 plotting bodies.
- Every public function requires a NumPy-style docstring covering parameters and returns.
- Core-level tests are diagnostic; final inference uses donors as biological replicates.
- Local validation uses small synthetic data and all temporary outputs remain on D:.
- Update `README.md` and `XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md` with every functional change.

---

### Task 1: Stage 1 plotting contracts

**Files:**
- Modify: `tests/test_stage2_spatial.py`
- Modify: `scripts/xty_am_pipeline.py`

**Interfaces:**
- Consumes: outputs of `summarize_stage1_by_donor_and_tissue`.
- Produces: `plot_stage1A_niche_dotmap(...) -> tuple[dict, dict, dict]` and `plot_stage1B_primary(...) -> tuple[Figure, ndarray, DataFrame]`.

- [ ] Add tests constructing minimal donor and tissue result tables, asserting figure/table returns, file saving, required-column errors, and public exports.
- [ ] Run `python -m pytest tests/test_stage2_spatial.py -k stage1_plot -q` and confirm failure because the functions are absent.
- [ ] Add the supplied plotting bodies, consolidate imports, and add complete parameter/return docstrings without changing visual behavior.
- [ ] Rerun the focused tests and confirm they pass.

### Task 2: Generic parallel per-core runner

**Files:**
- Modify: `tests/test_stage2_spatial.py`
- Modify: `scripts/xty_am_pipeline.py`

**Interfaces:**
- Consumes: a supported callable with `analysis_function(adata, random_state=..., **kwargs) -> pandas.DataFrame`.
- Produces: `run_spatial_function_multicore(analysis_function, adata, n_jobs=16, random_state=123, verbose=10, **function_kwargs) -> pandas.DataFrame`.

- [ ] Add tests showing serial/parallel reproducibility, per-core seed stability, no expression matrix requirement in workers, invalid `n_jobs` rejection, required metadata validation, and empty-schema preservation.
- [ ] Run the focused runner tests and confirm failure because the API is absent.
- [ ] Implement one private `SimpleNamespace` worker, lazy joblib import, strict input validation, deterministic seed spawning, and concatenation that preserves empty schemas.
- [ ] Rerun the focused tests and confirm they pass.

### Task 3: Stage 2 data preparation and continuous estimands

**Files:**
- Modify: `tests/test_stage2_spatial.py`
- Modify: `scripts/xty_am_pipeline.py`

**Interfaces:**
- Produces validated private helpers plus `calculate_stage2_knn_continuum_by_core`, `calculate_stage2_radius_continuum_by_core`, and `calculate_stage2_nearest_at2_by_core` returning core-level DataFrames.

- [ ] Add synthetic spatial tests with increasing scores and AT2 exposure, asserting positive direction, deterministic permutations, nearest-distance sign orientation, and schema-stable empty results.
- [ ] Add a radius test in which one AM has no neighbors and assert it is excluded rather than assigned AT2 fraction zero.
- [ ] Add validation tests for coordinates, scaling, scales, permutations, metadata consistency, and invariant scores.
- [ ] Run the focused tests and confirm failure because Stage 2 functions are absent.
- [ ] Implement strict Stage 2 preparation, safe Spearman calculation, continuous kNN/radius calculations, zero-neighbor exclusion, nearest-AT2 calculation, and complete output schemas.
- [ ] Rerun the focused tests and confirm they pass.

### Task 4: Balanced extremes and compatibility wrapper

**Files:**
- Modify: `tests/test_stage2_spatial.py`
- Modify: `scripts/xty_am_pipeline.py`

**Interfaces:**
- Produces: `calculate_stage2_balanced_extremes_by_core(...) -> pandas.DataFrame` and `run_stage2_multicore(...) -> pandas.DataFrame`.

- [ ] Add tests for equal group sizes, positive high-minus-low direction, boundary ties, invalid `extreme_fraction`, zero-neighbor exclusion, and wrapper equivalence with the generic runner.
- [ ] Run the focused tests and confirm failure for the missing APIs.
- [ ] Implement the balanced-extremes estimator and the thin compatibility wrapper.
- [ ] Rerun the focused tests and confirm they pass.

### Task 5: Donor-level Stage 2 inference

**Files:**
- Modify: `tests/test_stage2_spatial.py`
- Modify: `scripts/xty_am_pipeline.py`

**Interfaces:**
- Produces: `summarize_stage2_by_donor_and_tissue(core_results, min_donors=3) -> tuple[pandas.DataFrame, pandas.DataFrame]`.

- [ ] Add tests proving multiple cores from one donor collapse to one donor effect, Wilcoxon uses donor effects, FDR stays within the declared method/scale/tissue family, and empty inputs preserve schemas.
- [ ] Run the focused summary tests and confirm failure because the function is absent.
- [ ] Implement donor aggregation, one-sided signed-rank testing, and BH correction.
- [ ] Rerun the focused tests and confirm they pass.

### Task 6: Documentation and full validation

**Files:**
- Modify: `README.md`
- Modify: `XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md`
- Modify: `scripts/xty_am_pipeline.py`
- Test: `tests/test_stage2_spatial.py`

**Interfaces:**
- Documents all newly exported functions, estimands, output interpretation, local/HPC use, and scientific limitations.

- [ ] Update `__all__`, README function catalogue, notebook-facing examples, proposal workflow, version history, and zero-neighbor/donor-replicate methodology.
- [ ] Run syntax compilation and the focused Stage 2 tests.
- [ ] Run the complete pytest suite with D-drive temporary and matplotlib directories.
- [ ] Inspect `git diff --check`, source-file count, and repository status.
- [ ] Commit the tested feature branch and request code review before integration.
