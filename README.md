# Human Lung Xenium AM AT2 Analysis

This repository contains a reproducible Python pipeline for validating human
lung Xenium cell annotations and studying alveolar macrophage heterogeneity,
spatial organization, and relationships with alveolar type 2 epithelial cells.

The deposited source label **Macrophages** is a broad candidate pool, not an
automatic alveolar-macrophage definition. Interstitial-macrophage and
monocyte-like evidence must be reviewed first. Cycling AMs are outside scope.

## Current status

Notebook 00 provides the complete deposited UMAP, focused macrophage/AT1/AT2
views, donor-balanced marker statistics, competing macrophage evidence scores,
and complete-core spatial maps. It does not assign final AM or MHCII states.

## Repository structure

~~~text
XTY_AM_project/
├── README.md
├── scripts/
│   └── xty_am_pipeline.py
├── notebooks/
│   └── 00_methodology_data_audit.ipynb
├── tests/
│   ├── test_xty_am_pipeline.py
│   └── test_notebook_00_contract.py
├── outputs/local_test/
└── XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md
~~~

The project uses one reusable source script:
**scripts/xty_am_pipeline.py**. The Jupyter notebook is the step-wise execution
and review layer and imports functions from that script.

## Execution modes

- **local_test**: expression-heavy steps use four complete alveolar cores.
  Every cell within a selected core is retained.
- **hpc_full**: marker summaries use all cells. Spatial display stays bounded
  to four deterministic complete alveolar cores.

Default data locations:

| Mode | Data directory |
|---|---|
| Local | D:/Xiaonan/CODEX_projects/Xiaotong_AM/Spatial/Spatial |
| HPC | /dssg/home/acct-svetoslav_chakarov/svetoslav_chakarov/Data/External_Data/Xu_NC2026_human/data/Spatial |

Paths can be overridden without editing scientific code:

~~~powershell
$env:XTY_AM_RUN_MODE = "local_test"
$env:XTY_AM_DATA_DIR = "D:/path/to/Spatial"
$env:XTY_AM_OUTPUT_ROOT = "D:/path/to/outputs"
jupyter lab notebooks/00_methodology_data_audit.ipynb
~~~

~~~bash
export XTY_AM_RUN_MODE=hpc_full
export XTY_AM_DATA_DIR=/dssg/home/acct-svetoslav_chakarov/svetoslav_chakarov/Data/External_Data/Xu_NC2026_human/data/Spatial
export XTY_AM_OUTPUT_ROOT=/dssg/home/acct-svetoslav_chakarov/svetoslav_chakarov/Data/External_Data/Xu_NC2026_human/analysis_v1/outputs/hpc_full
jupyter lab notebooks/00_methodology_data_audit.ipynb
~~~

## Python dependencies

Python 3.12 with anndata, h5py, numpy, pandas, scipy, matplotlib, seaborn,
nbformat, nbclient, ipykernel, and pytest.

## Function catalogue

All public functions have NumPy-style docstrings describing inputs, outputs,
assumptions, and errors.

### Validation and selection

| Function | Purpose | Main output |
|---|---|---|
| **validate_xenium_metadata** | Verifies metadata, cell identifiers, UMAP, and spatial coordinates. | Audit dictionary |
| **select_representative_cores** | Selects complete alveolar cores across the core-size distribution. | Core IDs and summary |
| **marker_availability_table** | Separates measured markers from genes absent from the panel. | Availability outputs |
| **extract_marker_matrices** | Aligns transformed and raw values in explicit gene order and validates zero patterns. | Two matrices and concordance |

### Marker summaries and program scores

| Function | Purpose | Main output |
|---|---|---|
| **summarize_markers** | Calculates core and donor summaries, then averages donors equally. | Three summary tables |
| **compute_program_scores** | Standardizes genes within candidate cells and averages genes into evidence programs. | Cell, core, and donor scores |

Program scores are exploratory visualization aids, not classifiers, statistical
tests, or final AM definitions.

### Plotting and output

| Function | Purpose |
|---|---|
| **configure_plot_style** | Applies shared Cell-style typography and settings. |
| **plot_full_umap** | Plots every cell in the deposited UMAP. |
| **plot_focus_umap** | Highlights macrophage candidates, AT1, and AT2. |
| **plot_marker_dotplot** | Shows donor-balanced expression and detection. |
| **plot_program_umap** | Displays standardized macrophage evidence programs. |
| **plot_spatial_celltypes** | Maps all published cell types in complete cores. |
| **plot_spatial_focus** | Maps macrophage candidates, AT1, and AT2. |
| **plot_spatial_programs** | Maps macrophage evidence programs. |
| **save_figure** | Writes matching PNG and PDF files. |

## Marker definitions

Definitions are version controlled in **MARKER_MODULES**:

| Evidence | Measured markers |
|---|---|
| AT1 | AGER, SCEL |
| AT2 | SFTPD, PLA2G4F |
| Alveolar-macrophage evidence | MARCO, APOE |
| Interstitial-macrophage alternative | LYVE1, CD163, FCGR3A, MS4A4A |
| Monocyte or inflammatory alternative | FCN1, S100A12, IL1B, CLEC4E |
| Pan-macrophage support | CD68, AIF1, MPEG1, TYROBP |

Genes absent from the panel, such as FABP4, PPARG, C1QA, and SFTPC, are
reported as unavailable rather than interpreted as zero expression.

## Reusable palettes

Palettes are stored in **scripts/xty_am_pipeline.py**:

- **CELLTYPE_PALETTE**: full source-cell-type palette;
- **FOCUS_PALETTE**: macrophage, AT1, and AT2 colors;
- **PROGRAM_CMAP**: blue-white-coral evidence-score gradient;
- **EXPRESSION_CMAP**: pale-lavender-to-coral expression gradient;
- **CONTEXT_GREY**: background tissue context.

New palettes must be added to the same registry and documented here.

## Minimal Python example

~~~python
import anndata as ad
import numpy as np

from scripts.xty_am_pipeline import (
    MARKER_MODULES,
    extract_marker_matrices,
    marker_availability_table,
    select_representative_cores,
    summarize_markers,
)

adata = ad.read_h5ad("xenium.h5ad", backed="r")
selected_cores, core_summary = select_representative_cores(
    adata.obs, n_cores=4, tissue_code="A"
)
availability, availability_map, genes = marker_availability_table(
    adata.var_names
)
mask = adata.obs["core_id"].astype(str).isin(selected_cores).to_numpy()
indices = np.flatnonzero(mask)
transformed, raw, concordance = extract_marker_matrices(
    adata, row_indices=indices, genes=genes
)
summary, by_core, by_donor = summarize_markers(
    transformed,
    raw,
    adata.obs.iloc[indices],
    genes=genes,
    marker_modules=MARKER_MODULES,
)
~~~

## Statistical rules

- Donor is the biological replicate.
- Core is a spatial field nested within donor.
- Cells are not independent biological replicates.
- Spatial distances and neighborhoods never cross core boundaries.
- Local testing retains every cell in a selected core.
- Donor-balanced summaries prevent unequal cell counts dominating results.
- MHCII analysis starts only after the AM candidate rule is approved.

## Function documentation and maintenance

Whenever reusable functions are added or changed:

1. Write a failing behavioral test before production code.
2. Implement the function in **scripts/xty_am_pipeline.py**.
3. Add a detailed Python docstring covering scientific purpose, every input,
   every returned object, assumptions, coordinate/replicate boundaries, and
   possible errors.
4. Update this README function catalogue and examples.
5. Update and execute affected notebook cells top to bottom.
6. Update the living proposal and change log.
7. Regenerate bounded local-test outputs.
8. Run tests, inspect figures, commit, and push verified changes.

If a procedural Python script is supplied, convert reusable logic into
functions in the single source script. If R functions are added later, document
them with Roxygen comments covering parameters, returns, examples, and export.

## Validation

~~~powershell
python -m pytest -p no:cacheprovider -q
~~~

Tests cover function behavior, donor-balanced statistics, raw/transformed gene
alignment, intact-core selection, plotting, docstrings, top-to-bottom notebook
execution, and required outputs.

## Outputs

Bounded review outputs are stored in **outputs/local_test**:

- **figures**: matching PNG and PDF files;
- **tables**: marker availability, donor/core summaries, and scores;
- **notebook_00_summary.json**: machine-readable execution record.

Unrestricted HPC outputs remain outside Git under the configured HPC analysis
directory.
