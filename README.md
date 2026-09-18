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
| `assign_balanced_mhcii_score_groups` | Selects equal-sized high- and low-score AM tails within each core (or another declared grouping unit), leaves the middle as ambiguous, and returns group and cutoff audit tables. These are relative score groups rather than inferred biological clusters. |

### Cohort and abundance plots

| Function | Description |
|---|---|
| `plot_metadata_summary` | Creates a macaron-style cohort overview and returns unique core, donor, consistency, tissue, and TMA summary tables. |
| `plot_macrophage_pct_by_tissue` | Plots donor-level macrophage abundance by tissue after averaging multiple cores per donor/tissue; performs paired Wilcoxon tests and Benjamini–Hochberg correction. |
| `plot_am_at2_pct_by_donor` | Calculates the percentage of all cells that are AM or AT2 in each core and plots core-level values grouped by donor and colored by tissue. |

### Spatial neighborhood analysis

| Function | Description |
|---|---|
| `test_continuous_mhcii_at2_proximity` | Relates the continuous AM MHCII-high score to nearest-AT2 distance within each core, evaluates a within-core permutation null, aggregates correlations within donor, and performs tissue-level donor tests. |
| `calculate_nhood_enrichment_by_core` | Runs Squidpy radius-graph neighborhood enrichment separately in each core and returns focus-to-neighbor enrichment z-scores and counts; Squidpy is required only when this function is called. |
| `calculate_multitype_nhood_enrichment_by_core` | Tests undirected focal-target contact counts at multiple radii against within-core cell-label permutations. Symmetric pairs are deduplicated, excluded labels remain spatial background, and target hypotheses receive within-core FDR correction. |
| `calculate_multitype_knn_niche_by_core` | Tests directional target-cell fractions among each focal cell's k nearest neighbors against within-core target-label randomization. |
| `calculate_multitype_radius_niche_by_core` | Tests directional target-cell fractions in fixed-radius focal-cell neighborhoods and reports target exposure and empty-neighborhood diagnostics. |
| `calculate_multitype_nearest_distance_by_core` | Tests directional median nearest-target distances against within-core target-label randomization; positive effects consistently indicate closer-than-expected targets. |
| `summarize_stage1_by_donor_and_tissue` | Averages core effects within donor and tissue, then performs one-sided donor-level Wilcoxon tests with FDR correction across target types. |
| `run_spatial_function_multicore` | Runs a supported core-level spatial function with seeds derived from the master seed and stable core identity, so core order/subsetting does not change a retained core's null draws. Only required metadata and spatial coordinates are passed to workers rather than copying the expression matrix. |
| `calculate_stage2_balanced_extremes_by_core` | Compares local AT2 fractions between equal-sized high- and low-MHCII-score AM tails within each core and radius. Tied score boundaries are omitted rather than split arbitrarily, and AMs without any neighbor at that radius are excluded. |
| `calculate_stage2_knn_continuum_by_core` | Correlates the continuous AM MHCII score with the fraction of k nearest cells that are AT2, using within-core score permutations. |
| `calculate_stage2_radius_continuum_by_core` | Correlates the continuous AM MHCII score with fixed-radius AT2 fraction. It reports how many AMs were analyzed or excluded because their radius contained no measured cells. |
| `calculate_stage2_nearest_at2_by_core` | Correlates the continuous AM MHCII score with nearest-AT2 distance and reverses the sign so positive effects consistently mean that higher-score AMs are closer to AT2. |
| `run_stage2_multicore` | Compatibility wrapper around `run_spatial_function_multicore` for the Stage 2 functions. Core-level permutation P values and `FDR_core` are diagnostic rather than final biological-replicate inference. |
| `summarize_stage2_by_donor_and_tissue` | Averages Stage 2 effects across cores within donor/tissue, then performs one-sided donor-level Wilcoxon tests and Benjamini–Hochberg correction within declared method/scale/tissue families. |
| `rank_stage2_core_contributions` | Performs leave-one-core-out and leave-one-donor-out sensitivity analysis for one prespecified tissue/radius/hypothesis. Core correlations are combined equally on the Fisher-z scale within donor; AM cell counts are descriptive, and rows explicitly flag when omitting a core also removes a single-core donor. |
| `select_supportive_cores` | Selects supportive cores from distinct donors plus optional typical and discordant/near-null examples for transparent spatial illustration. The returned `selection_is_inferential=False` flag records that these examples cannot replace all-donor inference. |
| `summarize_nhood_by_donor` | Summarizes core-level neighborhood enrichment within donor, tissue, radius, and neighbor type. Mean/median z-scores are descriptive rather than a formal meta-analysis. |
| `calculate_knn_niche_continuum` | Quantifies each AM's k-nearest-neighbor composition within core, calculates within-core score correlations, combines core correlations within donor using Fisher z, and tests donor correlations by tissue. |
| `calculate_radius_niche_continuum` | Quantifies fixed-radius neighbor count, fraction, density, or presence around each AM within core and performs core-, donor-, and tissue-level correlation summaries. |

### Spatial ligand–receptor and CellChat preparation

Communication analysis has two deliberately separate tracks. The export
functions prepare validated sparse inputs for formal Spatial CellChat in R.
The Python functions calculate exploratory spatial expression co-occurrence;
their correlations are not CellChat probabilities and do not establish causal
signalling.

| Function | Description |
|---|---|
| `assign_balanced_mhcii_extremes` | Assigns equal MHCII-high and MHCII-low AM tails within each core for secondary CellChat contrasts. If either cutoff would split tied scores, all scored AMs in that core remain `Ambiguous`; scored AMs in underpowered cores are also `Ambiguous`, while unscored and non-AM cells remain `Unassigned`. |
| `add_cellchat_groups` | Creates `CellType_CCC`: MHCII AM tails and middle/unassigned AMs receive distinct labels, while every non-AM cell type is preserved. |
| `export_spatial_cellchat_inputs` | Requires an explicit normalized/log1p expression declaration and coordinate-unit contract, validates unique cell/gene IDs, complete metadata, finite micrometre-converted coordinates, and finite nonnegative expression, then exports compressed sparse inputs and a JSON manifest recording scale and units. Raw counts and unscaled non-micrometre coordinates are rejected. |
| `audit_lr_panel` | Normalizes ligand/receptor symbols, reports panel coverage, and distinguishes simple single-gene pairs from unsupported complex notation such as underscore-delimited receptor complexes. |
| `radius_weighted_mean` | Calculates uniform or Gaussian-weighted local target expression and neighbor counts inside a physical radius. |
| `calculate_lr_for_core_arrays` | For one core, correlates the continuous AM MHCII score with AM-gene expression multiplied by local AT2 partner-gene expression in both AM-to-AT2 and AT2-to-AM directions. Reports prevalence, spatial coverage, permutation P values, and overlap of the AM-side gene with the MHCII signature. |
| `calculate_continuous_spatial_lr` | Runs the bounded array calculation by core and radius without copying full AnnData objects to workers. It requires complete core/donor/tissue provenance and a positive finite coordinate scale, supports serial, process (`loky`), or thread execution, and derives seeds from stable core/pair identity. Diagnostic FDR is controlled within core × direction × radius across LR pairs. |
| `summarize_lr_by_donor` | Combines valid core correlations within donor using an equal-core Fisher-z mean, so donors with larger cores do not receive extra biological weight. |
| `summarise_lr_by_donor` | British-spelling alias of `summarize_lr_by_donor`. |
| `test_lr_across_donors` | Performs two-sided one-sample Wilcoxon tests on independent donor correlations and controls FDR within tissue × direction × radius across LR pairs. |

Untestable constant-expression pairs retain missing correlations and FDR rather
than being labelled nonsignificant. LR pairs whose AM-side gene contributed to
the MHCII score are explicitly flagged because those correlations can be
partly circular and require sensitivity analysis excluding overlapping genes.

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
| `plot_am_at2_spatial` | Maps selected MHCII AM groups or continuous AM scores together with AT2 cells inside one core. |
| `plot_nhood_enrichment_donor_tissue` | Produces donor-wise and tissue-summary plots for neighborhood-enrichment z-scores. |
| `plot_radius_core_correlations` | Plots core-level score-versus-radius-neighborhood correlations, colored by tissue, with median summaries. |
| `plot_knn_niche_continuum` | Plots tissue-stratified donor-level k-nearest-neighbor continuum correlations and FDR significance labels. |
| `plot_stage1A_niche_dotmap` | Creates one donor-level niche-discovery dot map per focal cell type. Color is median donor effect, size is donor-level FDR evidence, and outlines mark the selected FDR threshold. |
| `plot_stage1B_primary` | Filters full multitype summaries to the unordered AM–AT2 pair, then shows individual donor effects, medians, interquartile ranges, and donor-level tissue-test annotations for selected Stage 1 methods and scales. Both directional orientations are retained. |
| `plot_stage2_primary` | Shows donor-level effects, median diamonds, donor IQRs, and donor-level tissue-test annotations for prespecified continuous kNN, radius, and nearest-AT2 analyses plus a clearly marked secondary balanced-tail panel. Duplicate donor or tissue-test rows are rejected before plotting. |
| `plot_stage2_scale_sensitivity` | Displays the median and IQR of donor effects across k or radius values by tissue. This is a descriptive sensitivity plot based on unique donor rows, not a cell- or core-level inferential analysis. |
| `plot_core_contributions` | Preserves the supplied two-panel macaron plot of core Spearman correlations and leave-one-core-out influence, with optional top-influence filtering and file export. |
| `plot_mhcii_at2_spatial_core` | Exports a single-core PDF showing continuous AM MHCII score, AT2 cells, and background cells using a shared cross-core AM score scale. |
| `save_figure` | Saves a figure as matching PNG and PDF files. |

Stage 2 uses the continuous MHCII score as the primary exposure. Balanced
high/low tails are a sensitivity contrast and must not be described as newly
discovered discrete AM populations. For radius analyses, an AM with no measured
neighbor at a radius is not assigned an AT2 fraction of zero; it is excluded at
that scale and counted in the output diagnostics. Final tissue-level statements
must use `summarize_stage2_by_donor_and_tissue`, because donors—not cells or
cores—are the biological replicates.
