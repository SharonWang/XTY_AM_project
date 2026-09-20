# Stage 3 AM MHCII and AT2 VIM Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the supplied reusable plotting, signature-scoring, Stage 3 AM–AT2 VIM spatial analysis, spatial-core, and age-trend functions into the single validated Python pipeline and document the public API.

**Architecture:** Keep all production functions in `scripts/xty_am_pipeline.py`, but prefix every new private Stage 3 helper with `_stage3_` so existing private helpers are not overwritten. Continuous MHCII–VIM coupling is primary; categorical coupling and balanced tails are sensitivity analyses. Tests use small synthetic AnnData-like objects and verify donor/core isolation, fixed-geometry permutation behavior, effect signs, plotting outputs, and statistical aggregation.

**Tech Stack:** Python, NumPy, pandas, SciPy, Matplotlib, joblib, AnnData and Scanpy for the optional scoring test, pytest, Git.

**Spec:** `docs/superpowers/specs/2026-09-20-stage3-vim-pipeline-design.md`

## Global Constraints

- Modify files only under `D:\Xiaonan\CODEX_projects\Xiaotong_AM\XTY_AM_project`.
- Keep exactly one production source module: `scripts/xty_am_pipeline.py`.
- Do not add import-time analysis or load Xenium data during module import.
- Preserve supplied plot colours and panel geometry; change only validation and scientifically necessary labels.
- Use continuous MHCII–VIM coupling as primary and categorical coupling as sensitivity analysis.
- Keep cell/core P values diagnostic and use donors as biological replicates.
- Validate locally with synthetic or small subset data before HPC execution.
- Update `README.md` and `XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md` whenever the public API or method changes.
- Commit each independently validated task, then fast-forward merge to `main` and push `origin/main`.
- For every Python command, set `$python` to the bundled runtime and
  `PYTHONPATH` to the D-drive dependency target created in Task 0.

## Review Focus

- Reused `core_id` values across donors must never combine cells; Task 2 adds a regression test using identical core labels in two donors.
- All-cell kNN must exclude the focal AM itself before selecting `k` neighbors; Task 2 tests an AM whose self-coordinate would otherwise consume a slot.
- Tied MHCII tail boundaries must not be split into high/low groups; Task 2 tests an all-tied boundary and expects no analyzable result.
- Missing/ambiguous MHCII or VIM groups must be excluded transparently with retained-fraction QC; Task 3 tests explicit ambiguous cells.
- Zero-cell, constant-score, insufficient-donor, and absent-method inputs must return empty results or focused errors rather than misleading statistics; Tasks 1–5 exercise these cases.

---

### Task 0: D-drive test environment and clean baseline

**Files:**
- Create (ignored): `.tmp/testdeps/`

**Interfaces:**
- Consumes: bundled Python at `C:\Users\Xiaonan_Wang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.
- Produces: an isolated dependency directory under the D-drive worktree and a verified baseline test result.

- [ ] **Step 1: Install test-only dependencies under the D-drive worktree**

```powershell
$python = 'C:\Users\Xiaonan_Wang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
New-Item -ItemType Directory -Force -Path '.tmp\testdeps' | Out-Null
& $python -m pip install --target '.tmp\testdeps' pytest numpy pandas scipy matplotlib seaborn anndata scanpy joblib nbformat nbclient ipykernel
$env:PYTHONPATH = (Resolve-Path '.tmp\testdeps').Path
```

Expected: packages are installed only under the ignored D-drive `.tmp` directory.

- [ ] **Step 2: Run the unmodified full suite**

```powershell
& $python -m pytest tests -q -p no:cacheprovider --basetemp='.tmp\pytest-baseline'
```

Expected: the repository baseline passes before production changes. If it fails, record the failing test names and stop before implementation.

### Task 1: Reusable UMAP and dual-signature utilities

**Files:**
- Modify: `scripts/xty_am_pipeline.py`
- Create: `tests/test_reusable_plots_and_scoring.py`

**Interfaces:**
- Consumes: AnnData-compatible `obs`, `obsm`, `uns`, expression matrix, layers, and optional `raw`.
- Produces: `plot_anndata_group_umap` returning `(Figure, numpy.ndarray)` and `score_and_assign_two_signatures` returning the modified AnnData object.

- [ ] **Step 1: Write failing UMAP behavior tests**

```python
def test_plot_anndata_group_umap_supports_highlight_and_split(tmp_path):
    adata = make_plot_adata()
    output = tmp_path / "groups.pdf"
    fig, axes = pipeline.plot_anndata_group_umap(
        adata,
        group_col="group",
        split_by="group",
        split_categories=["A", "B"],
        palette={"A": "#E8928F", "B": "#7FAED2"},
        save=output,
    )
    assert len(axes) == 2
    assert output.exists()
    assert [axis.get_title() for axis in axes] == ["A", "B"]


def test_plot_anndata_group_umap_rejects_incomplete_palette():
    with pytest.raises(KeyError, match="No colour supplied"):
        pipeline.plot_anndata_group_umap(
            make_plot_adata(), group_col="group", palette={"A": "#E8928F"}
        )
```

- [ ] **Step 2: Run the UMAP tests and verify RED**

Run:

```powershell
& $python -m pytest tests/test_reusable_plots_and_scoring.py -k umap -q
```

Expected: collection or attribute failure because `plot_anndata_group_umap` is not exported.

- [ ] **Step 3: Add the UMAP function without mandatory AnnData imports**

Add `TYPE_CHECKING` to the existing typing import and use the exact signature and body from attachment `ad3183fb-d577-4a8d-9b04-49b6ebb8ee07/pasted-text.txt`. Replace its runtime AnnData import with this import pattern so environments that only use spatial functions do not require AnnData:

```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anndata import AnnData
```

Change only the annotation `adata: AnnData` to `adata: "AnnData"`; retain every supplied argument and default exactly.

Retain the supplied exact-dimension axes, stable draw order, split/highlight logic, optional rasterization, and vector legend. Add the public name to `__all__`.

- [ ] **Step 4: Run UMAP tests and verify GREEN**

Run the Step 2 command. Expected: all UMAP-selected tests pass.

- [ ] **Step 5: Write failing signature-scoring tests**

```python
def test_score_two_signatures_assigns_groups_and_margin_ambiguity():
    adata = make_expression_adata()
    result = pipeline.score_and_assign_two_signatures(
        adata,
        ["G1", "G2"],
        ["G3", "G4"],
        use_raw=False,
        scale=False,
        ambiguous=True,
        ambiguous_threshold=-100,
        min_score_difference=0.05,
        verbose=False,
    )
    assert result is adata
    assert {
        "Signature1_score", "Signature2_score", "Signature_group"
    }.issubset(adata.obs.columns)
    assert "Ambiguous" in adata.obs["Signature_group"].cat.categories


def test_score_two_signatures_rejects_raw_and_layer_together():
    with pytest.raises(ValueError, match="either use_raw"):
        pipeline.score_and_assign_two_signatures(
            make_expression_adata(), ["G1"], ["G3"], use_raw=True, layer="counts"
        )
```

- [ ] **Step 6: Run signature tests and verify RED**

Run:

```powershell
& $python -m pytest tests/test_reusable_plots_and_scoring.py -k signature -q
```

Expected: failure because the function and margin argument are absent.

- [ ] **Step 7: Add the supplied signature function with margin ambiguity**

Use the exact signature and body from attachment `4988dfd2-5cdd-4ce9-a02c-312c2cb73898/pasted-text.txt`, retain its lazy `anndata`/`scanpy` imports, insert `min_score_difference: float = 0.0` immediately after `ambiguous_threshold`, and apply this validation and classification rule after the two scores are available:

```python
if not np.isfinite(min_score_difference) or min_score_difference < 0:
    raise ValueError("min_score_difference must be finite and nonnegative")
score_difference = (score_1 - score_2).abs()
if ambiguous:
    ambiguous_mask = (
        ((score_1 < ambiguous_threshold) & (score_2 < ambiguous_threshold))
        | (score_difference < min_score_difference)
    )
    groups[ambiguous_mask] = ambiguous_label
```

Document that comparing independently controlled Scanpy scores is heuristic and add the function to `__all__`.

- [ ] **Step 8: Run Task 1 tests and commit**

```powershell
& $python -m pytest tests/test_reusable_plots_and_scoring.py -q
git add scripts/xty_am_pipeline.py tests/test_reusable_plots_and_scoring.py
git commit -m "Add reusable UMAP and signature utilities"
```

Expected: tests pass and the commit contains only Task 1 files.

### Task 2: Continuous Stage 3 core analyses and reproducible runner

**Files:**
- Modify: `scripts/xty_am_pipeline.py`
- Create: `tests/test_stage3_vim_pipeline.py`

**Interfaces:**
- Consumes: cell metadata containing donor, tissue, core, cell type, finite coordinates, continuous AM MHCII scores, and continuous AT2 VIM scores.
- Produces: four tidy core-result functions, `run_stage3_all_methods`, and `run_stage3_multicore`.

- [ ] **Step 1: Write failing core-isolation, kNN, tie, sign, and seed tests**

```python
def test_stage3_reused_core_ids_remain_separate_by_donor():
    adata = make_stage3_adata(reuse_core_id=True)
    result = pipeline.calculate_stage3_radius_continuum_by_core(
        adata, radii=(3,), min_at2_neighbors=1, min_am=3,
        n_permutations=19, random_state=11,
    )
    assert set(result["donor_id"]) == {"D1", "D2"}
    assert len(result) == 2


def test_stage3_all_cell_knn_excludes_focal_am_self():
    result = pipeline.calculate_stage3_knn_continuum_by_core(
        make_self_neighbor_adata(), k_values=(1,), neighbor_pool="all",
        min_at2_neighbors=1, min_am=3, n_permutations=19,
    )
    assert result.iloc[0]["median_at2_neighbors"] == 1


def test_stage3_balanced_extremes_do_not_split_tied_boundaries():
    result = pipeline.calculate_stage3_balanced_extremes_by_core(
        make_tied_stage3_adata(), radii=(3,), min_at2_neighbors=1,
        min_extreme_cells=2, n_permutations=19,
    )
    assert result.empty


def test_stage3_negative_coupling_has_negative_effect():
    result = pipeline.calculate_stage3_nearest_at2_by_core(
        make_stage3_adata(negative_coupling=True), min_am=3,
        n_permutations=19, random_state=17,
    )
    assert result.iloc[0]["effect"] < 0


def test_stage3_multicore_seed_is_stable_when_unrelated_core_is_removed():
    full = pipeline.run_stage3_multicore(
        pipeline.calculate_stage3_radius_continuum_by_core,
        make_stage3_adata(include_unrelated=True), n_jobs=1,
        radii=(3,), min_at2_neighbors=1, min_am=3, n_permutations=19,
    )
    subset = pipeline.run_stage3_multicore(
        pipeline.calculate_stage3_radius_continuum_by_core,
        make_stage3_adata(include_unrelated=False), n_jobs=1,
        radii=(3,), min_at2_neighbors=1, min_am=3, n_permutations=19,
    )
    columns = ["donor_id", "core_id", "effect", "p_value"]
    pd.testing.assert_frame_equal(
        full.loc[full["donor_id"].eq("D1"), columns].reset_index(drop=True),
        subset.loc[:, columns].reset_index(drop=True),
    )
```

- [ ] **Step 2: Run continuous Stage 3 tests and verify RED**

```powershell
& $python -m pytest tests/test_stage3_vim_pipeline.py -k "reused or self or tied or negative or seed" -q
```

Expected: missing public functions.

- [ ] **Step 3: Add namespaced Stage 3 validation and statistic helpers**

Add the supplied helpers under Stage 3-specific names:

```python
class _Stage3ObsOnly:
    def __init__(self, obs: pd.DataFrame):
        self.obs = obs


def _stage3_group_columns(core_col, donor_col, tissue_col):
    return [donor_col] + ([tissue_col] if tissue_col is not None else []) + [core_col]


def _stage3_iter_cores(obs, core_col, donor_col, tissue_col):
    columns = _stage3_group_columns(core_col, donor_col, tissue_col)
    yield from obs.groupby(columns, observed=True, dropna=False, sort=True)


def _stage3_seed(master_seed, *identity):
    payload = "|".join(map(str, (master_seed, *identity))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
```

Rename all supplied generic helpers (`_safe_spearman`, `_permutation_summary`, `_bh_fdr`, `_finish`, `_common_kwargs`, and related helpers) with `_stage3_` prefixes. Validate a positive finite `coordinate_scale_to_um` and multiply coordinate columns once during preparation.

- [ ] **Step 4: Add the four continuous methods**

Insert the supplied continuous method bodies with these corrections:

```python
# Tie-safe balanced tails: do not split equal boundary values.
order = np.argsort(am_score, kind="mergesort")
n_tail = min(int(np.floor(len(order) * extreme_fraction)), len(order) // 2)
if n_tail < min_extreme_cells:
    continue
values = am_score[order]
if values[n_tail - 1] == values[n_tail] or values[-n_tail] == values[-n_tail - 1]:
    continue
low, high = order[:n_tail], order[-n_tail:]

# All-cell kNN: query one additional point and remove the focal row itself.
query_k = min(int(k) + 1, len(pool_xy))
_, raw_indices = tree.query(am_xy, k=query_k)
for focal_pool_index, row in zip(am_pool_indices, np.atleast_2d(raw_indices)):
    other = row[row != focal_pool_index][: int(k)]
    retained = other[pool_is_at2[other]]
    neighbors.append(pool_at2_index[retained])
```

Group by donor/tissue/core, use stable identity-derived seeds for each core/method/scale, preserve fixed spatial neighborhoods during within-core AT2-score permutations, and report coverage/count diagnostics. Export all four functions.

- [ ] **Step 5: Add all-method and multicore runners**

Use the supplied nested keyword dictionaries for `run_stage3_all_methods`. In `run_stage3_multicore`, send only per-core metadata and derive each worker seed from donor, tissue, core, function name, and master seed rather than encounter order. Confirm serial and `loky` outputs sort identically.

- [ ] **Step 6: Run Task 2 tests and commit**

```powershell
& $python -m pytest tests/test_stage3_vim_pipeline.py -k "continuous or reused or self or tied or negative or seed" -q
git add scripts/xty_am_pipeline.py tests/test_stage3_vim_pipeline.py
git commit -m "Add continuous Stage 3 MHCII VIM analyses"
```

Expected: all selected tests pass.

### Task 3: Categorical Stage 3 sensitivity analyses

**Files:**
- Modify: `scripts/xty_am_pipeline.py`
- Modify: `tests/test_stage3_vim_pipeline.py`

**Interfaces:**
- Consumes: categorical `MHCII_group` and `AT2_VIM_group` labels plus the Stage 3 spatial metadata contract from Task 2.
- Produces: pair-enrichment, kNN, radius, and nearest-AT2 categorical core tables with retention QC.

- [ ] **Step 1: Write failing categorical effect and QC tests**

```python
def test_stage3_categorical_pair_effect_is_negative_for_discordant_pairing():
    result = pipeline.calculate_stage3_categorical_pair_enrichment_by_core(
        make_categorical_stage3_adata(discordant=True), radii=(3,),
        min_am_per_group=2, min_at2_per_group=2, n_permutations=19,
    )
    assert result.iloc[0]["effect"] < 0
    assert result.iloc[0]["effect_name"] == "log_concordance_odds_ratio"


def test_stage3_categorical_reports_ambiguous_cell_exclusion():
    result = pipeline.calculate_stage3_categorical_radius_by_core(
        make_categorical_stage3_adata(include_ambiguous=True), radii=(3,),
        min_at2_neighbors=1, min_am_per_group=2,
        n_balance_repeats=5, n_permutations=19,
    )
    row = result.iloc[0]
    assert row["n_am_excluded"] > 0
    assert row["n_at2_excluded"] > 0
    assert 0 < row["am_retained_fraction"] < 1


def test_stage3_categorical_nearest_requires_both_am_groups():
    result = pipeline.calculate_stage3_categorical_nearest_at2_by_core(
        make_categorical_stage3_adata(one_am_group=True),
        min_am_per_group=2, n_permutations=19,
    )
    assert result.empty
```

- [ ] **Step 2: Run categorical tests and verify RED**

```powershell
& $python -m pytest tests/test_stage3_vim_pipeline.py -k categorical -q
```

Expected: missing categorical functions.

- [ ] **Step 3: Add namespaced categorical helpers and four methods**

Port the categorical section from the authoritative attachment. Prefix every helper `_stage3_categorical_` and reuse `_stage3_iter_cores`, `_stage3_permutation_summary`, and `_stage3_finish`. Preserve these effect definitions:

```python
def _stage3_log_concordance_or(counts, correction=0.5):
    corrected = np.asarray(counts, dtype=float) + float(correction)
    return float(np.log(
        (corrected[1, 1] * corrected[0, 0])
        / (corrected[1, 0] * corrected[0, 1])
    ))


def _stage3_local_vimhi_difference(local_fraction, high, low):
    return float(np.mean(local_fraction[high]) - np.mean(local_fraction[low]))
```

Use tie-safe labels supplied in `obs`, retain explicit excluded-cell QC, exclude the focal AM from all-cell categorical kNN, and state that edge-count odds ratios are interpreted alongside per-AM local-fraction analyses.

- [ ] **Step 4: Run Task 3 tests and commit**

```powershell
& $python -m pytest tests/test_stage3_vim_pipeline.py -k categorical -q
git add scripts/xty_am_pipeline.py tests/test_stage3_vim_pipeline.py
git commit -m "Add categorical Stage 3 sensitivity analyses"
```

Expected: all categorical tests pass.

### Task 4: Donor inference and Stage 3 figures

**Files:**
- Modify: `scripts/xty_am_pipeline.py`
- Modify: `tests/test_stage3_vim_pipeline.py`

**Interfaces:**
- Consumes: tidy continuous or categorical Stage 3 core-result tables from Tasks 2–3.
- Produces: donor summaries, tissue tests, primary figures, scale-sensitivity figures, and categorical heatmaps.

- [ ] **Step 1: Write failing donor inference tests**

```python
def test_stage3_summary_equal_weights_cores_and_supports_less_alternative():
    core_results = make_stage3_core_results()
    donor, tissue = pipeline.summarize_stage3_by_donor_and_tissue(
        core_results, alternative="less", min_donors=3,
    )
    d1 = donor.loc[donor["donor_id"].eq("D1")].iloc[0]
    expected = np.tanh(np.mean(np.arctanh([0.2, 0.8])))
    assert d1["donor_effect"] == pytest.approx(expected)
    assert d1["n_cores"] == 2
    assert tissue.iloc[0]["alternative"] == "less"


def test_stage3_summary_fdr_is_scoped_by_method_effect_and_scale():
    _, tissue = pipeline.summarize_stage3_by_donor_and_tissue(
        make_multifamily_stage3_results(), min_donors=3,
    )
    assert "fdr_bh" in tissue
    assert set(tissue["fdr_family_size"]) == {3}
```

- [ ] **Step 2: Write failing plot tests**

```python
def test_stage3_plots_save_outputs(tmp_path):
    donor, tissue = make_stage3_summaries()
    primary = tmp_path / "primary.pdf"
    sensitivity = tmp_path / "sensitivity.pdf"
    heatmap = tmp_path / "pairs.pdf"
    pipeline.plot_stage3_primary(donor, tissue, save=primary)
    pipeline.plot_stage3_scale_sensitivity(donor, save=sensitivity)
    pipeline.plot_stage3_categorical_pair_heatmap(
        make_pair_results(), save=heatmap
    )
    assert primary.exists() and sensitivity.exists() and heatmap.exists()
```

- [ ] **Step 3: Run Task 4 tests and verify RED**

```powershell
& $python -m pytest tests/test_stage3_vim_pipeline.py -k "summary or plots" -q
```

Expected: missing summary and plotting functions.

- [ ] **Step 4: Add donor/tissue aggregation**

Port the supplied summary with an explicit alternative and family-scoped FDR:

```python
def summarize_stage3_by_donor_and_tissue(
    stage3_results,
    *,
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    core_col="core_id",
    min_donors=3,
    alternative="two-sided",
):
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    # Fisher-z equal-core mean for correlations; native equal-core mean otherwise.
    # One donor value enters each signed-rank test.
```

Apply BH independently within each `(method, effect_name, scale_type, scale)` family across tissues, and record `alternative` and `fdr_family_size` in `tissue_tests`.

- [ ] **Step 5: Add the supplied public plots**

Port the final definitions of `plot_stage3_primary` and `plot_stage3_scale_sensitivity`; do not expose the earlier `_v1` duplicate implementations. Add the three categorical plot wrappers/functions. Keep donor points, median/IQR overlays, macaron tissue palette, continuous sign labels, categorical sign labels, and optional FDR annotations. Export public plot functions.

- [ ] **Step 6: Run Task 4 tests and commit**

```powershell
& $python -m pytest tests/test_stage3_vim_pipeline.py -k "summary or plot" -q
git add scripts/xty_am_pipeline.py tests/test_stage3_vim_pipeline.py
git commit -m "Add Stage 3 donor inference and figures"
```

Expected: all selected tests pass.

### Task 5: Extended spatial-core plot and exploratory age model

**Files:**
- Modify: `scripts/xty_am_pipeline.py`
- Create: `tests/test_stage3_supporting_plots.py`

**Interfaces:**
- Consumes: current `plot_mhcii_at2_spatial_core` inputs plus optional AT2-state metadata; donor/core AM counts plus age.
- Produces: state-aware spatial PDFs and donor-level MHCII-high age-trend figures/tables.

- [ ] **Step 1: Write failing spatial extension tests**

```python
def test_spatial_core_plot_supports_at2_state_groups(tmp_path):
    output = pipeline.plot_mhcii_at2_spatial_core(
        make_spatial_plot_adata(), "C1", output_dir=tmp_path,
        at2_display="group", at2_group_col="AT2_VIM_group", show=False,
    )
    assert output.exists()
    assert output.suffix == ".pdf"


def test_spatial_core_plot_default_remains_all_at2(tmp_path):
    output = pipeline.plot_mhcii_at2_spatial_core(
        make_spatial_plot_adata(), "C1", output_dir=tmp_path, show=False,
    )
    assert output.exists()
```

- [ ] **Step 2: Run spatial tests and verify RED**

```powershell
& $python -m pytest tests/test_stage3_supporting_plots.py -k spatial -q
```

Expected: `at2_display` is not accepted.

- [ ] **Step 3: Extend the existing spatial function**

Replace only the current function body/signature with the supplied extended version. Preserve the current defaults and add:

```python
at2_display="all",
at2_group_col="AT2_VIM_group",
at2_group_order=None,
at2_group_colors=None,
unassigned_at2_label="Unassigned",
show_at2_counts=True,
legend_outside=True,
```

Validate `at2_display in {"all", "group"}` and require `at2_group_col` only for grouped display.

- [ ] **Step 4: Write failing age-model tests**

```python
def test_mhcii_hi_age_plot_uses_all_am_in_denominator(tmp_path):
    result = pipeline.plot_mhcii_hi_proportion_by_age(
        make_age_adata(), output_file=tmp_path / "age.pdf"
    )
    donor = result["donor_summary"].set_index("donor_id")
    assert donor.loc["D1", "n_all_am"] == 4
    assert donor.loc["D1", "n_mhcii_hi"] == 1
    assert result["output_file"].exists()


def test_mhcii_hi_age_model_rejects_inconsistent_donor_age():
    with pytest.raises(ValueError, match="inconsistent within donor"):
        pipeline.plot_mhcii_hi_proportion_by_age(make_age_adata(inconsistent=True))
```

- [ ] **Step 5: Run age tests and verify RED**

```powershell
& $python -m pytest tests/test_stage3_supporting_plots.py -k age -q
```

Expected: age function is missing.

- [ ] **Step 6: Add the quasibinomial fit and age plot**

Port the supplied functions under `_fit_grouped_quasibinomial` and `plot_mhcii_hi_proportion_by_age`. Replace the normal-reference slope P value with a donor-residual t reference:

```python
from scipy.stats import t as student_t

residual_df = max(len(age) - X.shape[1], 1)
t_value = slope_per_year / slope_se_per_year
p_value = 2 * student_t.sf(abs(t_value), df=residual_df)
```

Keep all AM states in the denominator, validate age consistency within donor, retain core points as descriptive, and label the model exploratory in the docstring. Export the public plot function.

- [ ] **Step 7: Run Task 5 tests and commit**

```powershell
& $python -m pytest tests/test_stage3_supporting_plots.py -q
git add scripts/xty_am_pipeline.py tests/test_stage3_supporting_plots.py
git commit -m "Add Stage 3 spatial and age plots"
```

Expected: all supporting-plot tests pass.

### Task 6: README, proposal, complete API audit, and branch delivery

**Files:**
- Modify: `README.md`
- Modify: `XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md`
- Modify: `scripts/xty_am_pipeline.py`
- Modify: `tests/test_stage3_vim_pipeline.py`
- Modify: `tests/test_reusable_plots_and_scoring.py`
- Modify: `tests/test_stage3_supporting_plots.py`

**Interfaces:**
- Consumes: all public functions delivered by Tasks 1–5.
- Produces: discoverable API documentation, updated methodology, clean full-suite validation, and merged/pushed Git history.

- [ ] **Step 1: Write failing API/documentation contract tests**

```python
PUBLIC_STAGE3 = (
    "plot_anndata_group_umap",
    "score_and_assign_two_signatures",
    "calculate_stage3_balanced_extremes_by_core",
    "calculate_stage3_knn_continuum_by_core",
    "calculate_stage3_radius_continuum_by_core",
    "calculate_stage3_nearest_at2_by_core",
    "run_stage3_all_methods",
    "run_stage3_multicore",
    "summarize_stage3_by_donor_and_tissue",
    "plot_stage3_primary",
    "plot_stage3_scale_sensitivity",
    "calculate_stage3_categorical_pair_enrichment_by_core",
    "calculate_stage3_categorical_knn_by_core",
    "calculate_stage3_categorical_radius_by_core",
    "calculate_stage3_categorical_nearest_at2_by_core",
    "plot_stage3_categorical_primary",
    "plot_stage3_categorical_scale_sensitivity",
    "plot_stage3_categorical_pair_heatmap",
    "plot_mhcii_hi_proportion_by_age",
)


def test_stage3_public_api_is_exported_documented_and_has_docstrings():
    readme = Path("README.md").read_text(encoding="utf-8")
    for name in PUBLIC_STAGE3:
        assert name in pipeline.__all__
        assert "Parameters" in inspect.getdoc(getattr(pipeline, name))
        assert "Returns" in inspect.getdoc(getattr(pipeline, name))
        assert f"`{name}`" in readme
```

- [ ] **Step 2: Run contract test and verify RED**

```powershell
& $python -m pytest tests/test_stage3_vim_pipeline.py -k public_api -q
```

Expected: README coverage and any incomplete docstrings fail.

- [ ] **Step 3: Complete exports and docstrings**

Add every public name to the existing organized `__all__` list. Ensure each public function has NumPy-style `Parameters`, `Returns`, and applicable `Raises`/`Notes` sections. Keep private helpers out of `__all__`.

- [ ] **Step 4: Update README**

Add a Stage 3 subsection to the existing function table. For each public function, describe its input concept, output, biological sign, and whether it is primary, sensitivity, diagnostic, or exploratory. State the local and HPC data locations already used by the project. Do not turn README into a methods manuscript.

- [ ] **Step 5: Update the proposal**

Add Stage 3 continuous and categorical workflows, donor inference, manuscript-derived negative-coupling hypothesis, age-model limitation, validation status, and a dated changelog entry. Preserve the distinction between observed manuscript results and the untested VIM-specific spatial hypothesis.

- [ ] **Step 6: Run focused and full verification**

```powershell
& $python -m py_compile scripts/xty_am_pipeline.py
& $python -m pytest tests/test_reusable_plots_and_scoring.py tests/test_stage3_vim_pipeline.py tests/test_stage3_supporting_plots.py -q
& $python -m pytest tests -q
git diff --check
git status --short
```

Expected: compilation succeeds, focused tests pass, the complete suite has zero failures, `git diff --check` is clean, and only intended source/test/documentation files are modified.

- [ ] **Step 7: Commit documentation and final audit**

```powershell
git add README.md XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md scripts/xty_am_pipeline.py tests/test_reusable_plots_and_scoring.py tests/test_stage3_vim_pipeline.py tests/test_stage3_supporting_plots.py
git commit -m "Document and validate Stage 3 VIM pipeline"
```

- [ ] **Step 8: Re-run the full suite on the committed feature branch**

```powershell
& $python -m pytest tests -q
git status --short --branch
```

Expected: zero failures and a clean feature branch.

- [ ] **Step 9: Fast-forward merge and verify main**

From the main checkout:

```powershell
git status --short --branch
git merge --ff-only codex/stage3-vim-pipeline
& $python -m pytest tests -q
```

Expected: clean pre-merge main, a fast-forward merge, and zero failures on merged main.

- [ ] **Step 10: Push and verify the remote-tracking branch**

```powershell
git push origin main
git status --short --branch
git rev-parse main
git rev-parse origin/main
```

Expected: `main` is not ahead of `origin/main` and both revisions match. If GitHub is unreachable, retain the local commits and report the exact network error without claiming the push succeeded.
