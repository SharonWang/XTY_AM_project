# First Annotation Validation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and locally execute the first reader-facing notebook showing the published lung cell-type UMAP, AM/AT1/AT2 definitions, marker dotplots, and representative whole-core spatial maps.

**Architecture:** One self-contained Python notebook reads `xenium.h5ad` in backed mode, validates the published `celltype_final` labels, and produces bounded Cell-style figures with an explicit macaron palette. Local validation uses complete cells from a deterministic set of whole cores so spatial relationships remain valid; full HPC mode uses every eligible core with the same functions.

**Tech Stack:** Python 3, Jupyter/nbformat/nbclient, anndata, h5py, pandas, NumPy, SciPy sparse matrices, Matplotlib, Seaborn, pytest.

**Spec:** `XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md` version 0.3

## Global Constraints

- Modify files only under `D:\Xiaonan\CODEX_projects\Xiaotong_AM\XTY_AM_project`.
- Read local data from `D:\Xiaonan\CODEX_projects\Xiaotong_AM\Spatial\Spatial`.
- Default HPC analysis path is `/dssg/home/acct-svetoslav_chakarov/svetoslav_chakarov/Data/External_Data/Xu_NC2026_human/analysis_v1`.
- Default HPC data path is `/dssg/home/acct-svetoslav_chakarov/svetoslav_chakarov/Data/External_Data/Xu_NC2026_human/data/Spatial`.
- Scientific analysis must live in Jupyter notebooks with explanatory Markdown before each step.
- Preserve all cells from selected test cores; never subsample within a core for spatial validation.
- Treat `celltype_final == "Macrophages"`, `"AT1"`, and `"AT2"` as published source labels, not newly inferred labels.
- Do not assign MHCII-high/low states in this notebook.
- Use explicit Cell-style macaron colors, white backgrounds, charcoal labels, rasterized points, and vector text.
- Commit every verified change. Do not push internal paths to GitHub without explicit authorization.

---

### Task 1: Establish the executable notebook contract

**Files:**
- Create: `tests/test_notebook_00_contract.py`
- Create: `notebooks/00_methodology_data_audit.ipynb`

**Interfaces:**
- Consumes: proposal version 0.3 and the notebook path.
- Produces: a structurally valid notebook with a tagged `parameters` cell and the required reader-facing sections.

- [ ] **Step 1: Write the failing contract test**

```python
from pathlib import Path
import nbformat

NOTEBOOK = Path("notebooks/00_methodology_data_audit.ipynb")

def test_notebook_has_required_structure():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    markdown = "\n".join(
        cell.source for cell in nb.cells if cell.cell_type == "markdown"
    )
    for heading in [
        "## tl;dr",
        "## Context & Methods",
        "## Data",
        "## Results",
        "## Takeaways",
    ]:
        assert heading in markdown
    assert any("parameters" in cell.metadata.get("tags", []) for cell in nb.cells)
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run: `pytest tests/test_notebook_00_contract.py -q`

Expected: fail because `notebooks/00_methodology_data_audit.ipynb` does not exist.

- [ ] **Step 3: Create the minimal notebook scaffold with nbformat**

Create Markdown sections in the required order and a tagged parameter cell containing `RUN_MODE`, local/HPC paths, `RANDOM_SEED = 20260912`, output paths, figure formats, and deterministic test-core settings.

- [ ] **Step 4: Re-run the contract test**

Run: `pytest tests/test_notebook_00_contract.py -q`

Expected: one passing test.

### Task 2: Add input, annotation, and matrix validation

**Files:**
- Modify: `notebooks/00_methodology_data_audit.ipynb`
- Modify: `tests/test_notebook_00_contract.py`

**Interfaces:**
- Consumes: `xenium.h5ad`.
- Produces: validated observation metadata, declared expression matrix, full/local-test core selection, and summary tables.

- [ ] **Step 1: Add a failing execution test**

```python
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "00_methodology_data_audit.ipynb"

def execute_local_notebook():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO)}},
    )
    client.execute()
    summary_path = Path(os.environ["XTY_AM_OUTPUT_ROOT"]) / "notebook_00_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))

def test_local_notebook_executes_and_writes_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("XTY_AM_RUN_MODE", "local_test")
    monkeypatch.setenv("XTY_AM_OUTPUT_ROOT", str(tmp_path))
    executed = execute_local_notebook()
    assert executed["n_cells"] > 0
    assert executed["celltype_counts"]["Macrophages"] > 0
    assert executed["celltype_counts"]["AT1"] > 0
    assert executed["celltype_counts"]["AT2"] > 0
    assert executed["duplicate_cell_ids"] == 0
```

Expected RED result: execution helper or notebook summary is missing.

- [ ] **Step 2: Implement visible preflight checks**

The notebook must assert:

```python
required_obs = {
    "donor_id", "core_id", "tissue_annotation", "celltype_final",
    "x_centroid", "y_centroid", "n_counts", "n_genes"
}
assert required_obs.issubset(adata.obs.columns)
assert adata.obs_names.is_unique
assert adata.obsm["X_umap"].shape == (adata.n_obs, 2)
assert adata.obsm["spatial"].shape == (adata.n_obs, 2)
```

The notebook will report matrix/layer semantics and use the normalized/log-transformed `X` for visualization while preserving `raw.X` for later count-aware work.

- [ ] **Step 3: Implement spatially valid local-test selection**

Choose a deterministic small set of complete cores containing macrophages, AT1, and AT2, prioritizing alveolar cores and including one non-alveolar edge case when available. Print selected cores, donors, regions, total cells, and target-cell counts. State that these outputs validate software only.

- [ ] **Step 4: Re-run the test**

Expected: the notebook executes through the audit cells and the summary values satisfy the assertions.

### Task 3: Validate published cell definitions with UMAP and markers

**Files:**
- Modify: `notebooks/00_methodology_data_audit.ipynb`
- Modify: `tests/test_notebook_00_contract.py`

**Interfaces:**
- Consumes: validated labels, `X_umap`, and normalized expression.
- Produces: full cell-type UMAP, focused AM/AT1/AT2 UMAP, marker-coverage table, and marker dotplot.

- [ ] **Step 1: Add failing figure-contract assertions**

Assert that the executed notebook exposes figure captions/titles for:

- all annotated lung cell types on UMAP;
- macrophages, AT1, and AT2 highlighted on UMAP;
- marker expression prevalence and mean expression by cell type.

- [ ] **Step 2: Define annotation evidence**

Use the authors' released Figure 5 marker pairs:

```python
AUTHOR_MARKERS = {
    "AT1": ["AGER", "SCEL"],
    "AT2": ["SFTPD", "PLA2G4F"],
    "Macrophages": ["MARCO", "APOE"],
}
```

Add panel-available supporting markers only after printing their availability and biological role. Show missing canonical markers such as `SFTPC` and `FABP4` in a separate availability table; never plot them as zeros.

- [ ] **Step 3: Build Cell-style figures**

Use a white background, Arial/DejaVu Sans fallback, 300 dpi, thin axes, rasterized scatter points, and explicit macaron colors. Reserve distinct focal colors for AM, AT1, and AT2 and render other cells in light grey in the focused UMAP. Use color plus direct labels/faceting so interpretation does not rely on color alone.

For the dotplot:

- dot area = fraction of cells with expression > 0;
- dot color = mean log-normalized expression among all cells in the group;
- rows = published cell types in a fixed biological order;
- columns = marker genes grouped by AM, AT1, and AT2;
- missing markers appear only in the adjacent coverage table.

- [ ] **Step 4: Validate figure data**

Independently reconcile plotted group counts with `obs.celltype_final.value_counts()`; assert prevalence is within [0, 1], group/gene combinations are unique, and all plotted genes are present.

### Task 4: Add whole-core spatial maps

**Files:**
- Modify: `notebooks/00_methodology_data_audit.ipynb`
- Modify: `tests/test_notebook_00_contract.py`

**Interfaces:**
- Consumes: full cells and centroids for each selected core.
- Produces: all-cell spatial maps and AM/AT1/AT2-focused maps with equal aspect ratio.

- [ ] **Step 1: Add failing spatial assertions**

Check that every plotted core contains exactly the source rows for that core and that no coordinate from another core enters the panel.

- [ ] **Step 2: Plot all cell types**

Plot one panel per selected core using the same fixed cell-type palette as UMAP. Use `axis("equal")`, invert the y-axis only if required by the Xenium coordinate convention and document the choice, suppress meaningless tick labels, and include core/donor/region/cell-count subtitles.

- [ ] **Step 3: Plot focused populations**

Render non-target cells in pale grey and overlay AT1, AT2, and macrophages with larger outlined points. Preserve the full core extent so apparent proximity is not created by zooming.

- [ ] **Step 4: Inspect exported PNG and PDF**

Verify labels, legends, point visibility, palette consistency, aspect ratio, and absence of clipping. The notebook must save figures under `outputs/local_test/figures/notebook_00/`.

### Task 5: Execute, validate, document, and commit

**Files:**
- Modify: `notebooks/00_methodology_data_audit.ipynb`
- Modify: `XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md`
- Create: `outputs/local_test/notebook_00_summary.json`

**Interfaces:**
- Consumes: completed notebook and its generated outputs.
- Produces: executed notebook, validation status, proposal change entry, and Git commit.

- [ ] **Step 1: Execute top-to-bottom**

Run the notebook with the explicit bundled Python kernel and D:-resident cache/temp directories. Use `nbclient` or:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/00_methodology_data_audit.ipynb
```

- [ ] **Step 2: Validate outputs**

Run:

```bash
pytest tests/test_notebook_00_contract.py -q
git diff --check
```

Confirm no traceback outputs, bounded table sizes, non-empty figures, expected cell labels, and summary counts reconciled to the source.

- [ ] **Step 3: Update notebook conclusions**

Write the `tl;dr` and `Takeaways` from executed results only. Separate verified annotation evidence, limitations, and unresolved questions. Do not infer MHCII-high/low states.

- [ ] **Step 4: Update proposal change control**

Increment the proposal version and record the notebook, cell-definition details, local validation mode, and any method corrections.

- [ ] **Step 5: Commit**

```bash
git add XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md \
  notebooks/00_methodology_data_audit.ipynb \
  tests/test_notebook_00_contract.py \
  outputs/local_test/notebook_00_summary.json
git commit -m "analysis: add lung annotation validation notebook"
```

Push only after explicit authorization to publish the internal paths contained in the repository.
