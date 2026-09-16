# Human lung Xenium AM–AT2 analysis

## Dataset and paper

This repository analyses the human lung Xenium dataset accompanying:

> Xu et al. *Cellular hallmarks and aging clock of the human lung
> parenchyma*. Nature Communications (2026). PMID: 42457688.

The deposited master object contains 332,063 segmented cells, 389 measured
genes, 22 donors, and 70 tissue cores. It includes expression values, cell
metadata, published cell-type labels, UMAP coordinates, and spatial centroid
coordinates. The published `Macrophages` label is treated as a broad
macrophage candidate pool, not as an automatic alveolar-macrophage definition.
AT1, AT2, alveolar-macrophage, interstitial-macrophage, and monocyte marker
evidence must be checked before MHCII AM subtyping.

Important metadata fields are:

- `donor_id`: biological donor and primary replication unit;
- `core_id`: one spatially independent tissue core;
- `tma_id`: tissue-microarray identifier;
- `tissue_annotation`: anatomical annotation (`A`, `B`, `V`, or `None` in
  the deposited object);
- `celltype_final`: published source cell-type label;
- `X_umap`: deposited two-dimensional UMAP coordinates;
- `spatial`: cell-centroid coordinates, interpreted only within a core.

Data locations:

| Environment | Xenium data directory |
|---|---|
| Local | `D:/Xiaonan/CODEX_projects/Xiaotong_AM/Spatial/Spatial` |
| HPC | `/dssg/home/acct-svetoslav_chakarov/svetoslav_chakarov/Data/External_Data/Xu_NC2026_human/data/Spatial` |

The main input is `xenium.h5ad`. The complete mouse differential-expression
table is `BD_DEgenes.csv`; Cycling AM is excluded from the MHCII-high/low AM
question.

## Notebooks

All notebooks are stored in `notebooks/`. They are the step-wise scientific
record; reusable calculations and plotting implementations live in the single
source module described below.

| Notebook | Status | Description |
|---|---|---|
| `notebooks/00_methodology_data_audit.ipynb` | Implemented and locally executed | Audits the H5AD schema and metadata; displays the complete deposited UMAP; checks macrophage, AT1, and AT2 marker evidence; calculates donor-balanced marker summaries; and maps source labels and macrophage identity evidence in intact tissue cores. It does not finalize AM membership or MHCII states. |
| `notebooks/01_mouse_MHCII_signature_mapping.ipynb` | Planned | Audits the complete mouse DE table, excludes Cycling AM, maps mouse genes to human orthologues, and records Xenium-panel signature coverage. |
| `notebooks/02_human_AM_MHCII_states.ipynb` | Planned | Defines the reviewed AM population and evaluates continuous and categorical MHCII-high/low state evidence across donors and cores. |
| `notebooks/03_AM_spatial_neighborhoods.ipynb` | Planned | Tests AM-state locations, nearest cell types, AT2 distances, multi-radius neighborhoods, and within-core spatial null models. |
| `notebooks/04_AM_AT2_communication.ipynb` | Planned | Evaluates panel-observable, spatially supported AM–AT2 ligand–receptor candidates in both directions. |
| `notebooks/05_integrated_statistics_figures.ipynb` | Planned | Performs donor-aware models, sensitivity analyses, final statistical checks, and manuscript figure/source-data assembly. |
| `notebooks/06_manuscript_methods_results.ipynb` | Planned | Produces reproducible Methods, Results, legends, limitations, and executed manuscript numbers. |

## Source module

The only reusable Python source file is `scripts/xty_am_pipeline.py`.
Every public function has a NumPy-style docstring describing parameters,
returns, errors, and relevant scientific interpretation.

### Validation, selection, and metadata transfer

| Function | Description |
|---|---|
| `validate_xenium_metadata` | Validates required observation columns, unique cell IDs, UMAP/spatial dimensions, coordinate agreement, and donor/core counts. |
| `select_representative_cores` | Deterministically selects complete alveolar cores containing macrophage, AT1, and AT2 source labels across the core-size distribution. |
| `merge_obs_to_main` | Transfers `.obs` columns from a subset AnnData into a main object by `obs_names`, with duplicate-ID, subset, and conflict safeguards plus a merge report. |

### Expression and marker summaries

| Function | Description |
|---|---|
| `marker_availability_table` | Reports which declared marker genes are measured or absent from the Xenium panel. |
| `extract_marker_matrices` | Extracts transformed and raw marker matrices using their own gene indices and verifies matching zero patterns. |
| `summarize_markers` | Calculates core- and donor-level expression/detection summaries and an equal-weight donor summary for dotplots. |
| `cluster_expression_summary` | Returns mean expression, percentage detected, and cell count for each requested gene and group from `X` or a named layer. |
| `gene_detection_by_group` | Calculates raw-count gene-detection percentages by a metadata group. |
| `compute_program_scores` | Standardizes measured genes within a candidate population and averages them into exploratory evidence programs with core/donor summaries. |

### Mouse-to-human mapping and MHCII annotation

| Function | Description |
|---|---|
| `add_human_gene_name` | Maps mouse DE genes to one or more human orthologues and records whether each result came from the orthologue table or an uppercase-symbol fallback. |
| `assign_mhcii_single_signature` | Calculates a deterministic positive MHCII signature score and labels score/detection-discordant cells as `Ambiguous`. This is exploratory, not the final two-state AM definition. Processed `X` is the default scoring matrix; raw counts are used for direct core-gene detection when available. |

### Cohort and abundance plots

| Function | Description |
|---|---|
| `plot_metadata_summary` | Creates a macaron-style cohort overview and returns unique core, donor, consistency, tissue, and TMA summary tables. |
| `plot_macrophage_pct_by_tissue` | Plots donor-level macrophage abundance by tissue after averaging multiple cores per donor/tissue; performs paired Wilcoxon tests and Benjamini–Hochberg correction. |

### UMAP, dotplot, and spatial figures

| Function | Description |
|---|---|
| `configure_plot_style` | Applies the shared publication-style Matplotlib and Seaborn theme. |
| `plot_full_umap` | Plots every cell in the deposited UMAP with published cell-type colors. |
| `plot_focus_umap` | Highlights Macrophages, AT1, and AT2 over pale context cells. |
| `plot_marker_dotplot` | Displays donor-balanced mean expression as color and raw detection as dot area. |
| `plot_program_umap` | Maps exploratory macrophage identity-program scores on the deposited UMAP. |
| `plot_spatial_celltypes` | Facets intact cores and maps all published cell types with an explicit legend. |
| `plot_spatial_focus` | Facets intact cores and highlights Macrophages, AT1, and AT2 with an explicit legend. |
| `plot_spatial_programs` | Maps macrophage identity-program scores within intact cores without mixing coordinate systems. |
| `save_figure` | Saves a figure as matching PNG and PDF files. |

## Reusable palettes

The source module stores these version-controlled palettes:

- `CELLTYPE_PALETTE`: published lung cell types;
- `FOCUS_PALETTE`: Macrophages, AT1, and AT2;
- `TISSUE_PALETTE`: alveolar, bronchial, vascular, and unannotated cores;
- `SEX_PALETTE`: donor sex colors;
- `TMA_PALETTE`: tissue-microarray colors;
- `MACROPHAGE_SUBTYPE_PALETTE`: AM, AM-like, LYVE1+ IM, and MMP2+ IM;
- `PROGRAM_CMAP`: diverging macrophage-program scores;
- `EXPRESSION_CMAP`: marker-expression dotplots.

The full analysis rationale, statistical plan, and change history are maintained
in `XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md`.
