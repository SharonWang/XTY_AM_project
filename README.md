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
| Local | `D:/Xiaonan/CODEX_projects/Xiaotong_AM/Spatial` |
| HPC | `/dssg/home/acct-svetoslav_chakarov/svetoslav_chakarov/Data/External_Data/Xu_NC2026_human/data/Spatial` |

The main input is `xenium.h5ad`. The complete mouse differential-expression
table is `BD_DEgenes.csv`; Cycling AM is excluded from the MHCII-high/low AM
question.

## Notebooks

The 14 supplied notebooks are version-controlled exactly as received on
2026-09-24. They form the step-wise scientific record; reusable calculations
and plotting implementations live in the single source module described below.
Notebook outputs were generated in the project/HPC environment and are retained
for immediate review. Repository validation checks notebook structure and source
references without rerunning the full Xenium dataset locally.

The notebooks currently append the HPC analysis directory to `sys.path` and use
`import source`. The version-controlled canonical module is
`scripts/xty_am_pipeline.py`; the HPC `source.py` copy must correspond to the same
Git commit before execution.

| Execution order | Notebook | Description |
|---:|---|---|
| 1 | `notebooks/01_Metadata_Summary.ipynb` | Cohort and Xenium metadata audit: donors, cores, tissue/TMA annotations, cell counts, and Cell-style metadata summaries. |
| 2 | `notebooks/02_Myeloid_Refinement.ipynb` | Refines the broad deposited myeloid labels using marker-expression and detection summaries, including separation of alveolar macrophages from interstitial macrophages and monocytes. |
| 3 | `notebooks/02_MA_Refinement.ipynb` | Refines macrophage states with UMAP, abundance, and two-signature evidence after the myeloid review. |
| 4 | `notebooks/02_1_AM_MHCII_Scoring.ipynb` | Maps the mouse MHCII differential-expression evidence to human genes, audits Xenium-panel detection, and assigns the AM MHCII score/state used downstream; Cycling AM is outside the analysis question. |
| 5 | `notebooks/03_Epi_Refinement.ipynb` | Reviews and refines epithelial annotations before AT2-state analysis. |
| 6 | `notebooks/03_1_AT2_Vim_Scoring.ipynb` | Audits AT2 VIM-state signatures, panel coverage, UMAP placement, and continuous/categorical AT2 VIM scoring. |
| 7 | `notebooks/04_Merge_Obs_To_Main.ipynb` | Merges reviewed myeloid, AM, epithelial, MHCII, and VIM metadata back into the master AnnData object and checks AM/AT2 abundance by donor. |
| 8 | `notebooks/05_1A_Unbiased_Niche_Discovery.ipynb` | Stage 1A multitype neighborhood discovery using contact, kNN, radius, and nearest-distance analyses with donor-level summaries. |
| 9 | `notebooks/05_1B_Focused_AMAT2_Validation.ipynb` | Stage 1B prespecified AM-AT2 validation using the same core-local spatial methods and donor-level inference. |
| 10 | `notebooks/06_AM_MHCII_AT2_Spatial_Association.ipynb` | Stage 2 continuous AM MHCII-to-AT2 spatial association, balanced-tail sensitivity analysis, donor inference, and scale-sensitivity plots. |
| 11 | `notebooks/07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb` | Primary Stage 3 continuous coupling between AM MHCII score and local or nearest-AT2 VIM score. |
| 12 | `notebooks/07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb` | Secondary Stage 3 categorical MHCIIhi/MHCIIlo and VIMhi/VIMlo sensitivity analyses with explicit exclusion and balance diagnostics. |
| 13 | `notebooks/08_Spatial_Umap.ipynb` | Ranks core/donor influence, selects clearly labelled non-inferential example cores, and plots MHCII-AM and AT2/VIM spatial distributions. |
| 14 | `notebooks/09_MHCprop_Age_Correlation.ipynb` | Exploratory donor-level relationship between age and the proportion of all AMs assigned MHCII-high. |

## Source module

The only reusable Python source file is `scripts/xty_am_pipeline.py`.
Every public function has a NumPy-style docstring describing parameters,
returns, errors, and relevant scientific interpretation.

## Function-to-notebook index

This index covers every public function exported by `xty_am_pipeline.py`.
Detailed behavior, parameters, outputs, and scientific caveats remain in the
categorized descriptions below and in each function's NumPy-style docstring.

| Function | Used in notebook(s) |
|---|---|
| `add_cellchat_groups` | Reusable library function; not called directly by the current notebooks |
| `add_human_gene_name` | `02_1_AM_MHCII_Scoring.ipynb`<br>`03_1_AT2_Vim_Scoring.ipynb` |
| `assign_balanced_mhcii_extremes` | Reusable library function; not called directly by the current notebooks |
| `assign_balanced_mhcii_score_groups` | Reusable library function; not called directly by the current notebooks |
| `assign_mhcii_single_signature` | `02_1_AM_MHCII_Scoring.ipynb` |
| `audit_lr_panel` | Reusable library function; not called directly by the current notebooks |
| `calculate_continuous_spatial_lr` | Reusable library function; not called directly by the current notebooks |
| `calculate_knn_niche_continuum` | Reusable library function; not called directly by the current notebooks |
| `calculate_lr_for_core_arrays` | Reusable library function; not called directly by the current notebooks |
| `calculate_multitype_knn_niche_by_core` | `05_1A_Unbiased_Niche_Discovery.ipynb`<br>`05_1B_Focused_AMAT2_Validation.ipynb` |
| `calculate_multitype_nearest_distance_by_core` | `05_1A_Unbiased_Niche_Discovery.ipynb`<br>`05_1B_Focused_AMAT2_Validation.ipynb` |
| `calculate_multitype_nhood_enrichment_by_core` | `05_1A_Unbiased_Niche_Discovery.ipynb`<br>`05_1B_Focused_AMAT2_Validation.ipynb` |
| `calculate_multitype_radius_niche_by_core` | `05_1A_Unbiased_Niche_Discovery.ipynb`<br>`05_1B_Focused_AMAT2_Validation.ipynb` |
| `calculate_nhood_enrichment_by_core` | Reusable library function; not called directly by the current notebooks |
| `calculate_radius_niche_continuum` | Reusable library function; not called directly by the current notebooks |
| `calculate_stage2_balanced_extremes_by_core` | `06_AM_MHCII_AT2_Spatial_Association.ipynb` |
| `calculate_stage2_knn_continuum_by_core` | `06_AM_MHCII_AT2_Spatial_Association.ipynb` |
| `calculate_stage2_nearest_at2_by_core` | `06_AM_MHCII_AT2_Spatial_Association.ipynb` |
| `calculate_stage2_radius_continuum_by_core` | `06_AM_MHCII_AT2_Spatial_Association.ipynb` |
| `calculate_stage3_balanced_extremes_by_core` | `07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb` |
| `calculate_stage3_categorical_knn_by_core` | `07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb` |
| `calculate_stage3_categorical_nearest_at2_by_core` | `07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb` |
| `calculate_stage3_categorical_pair_enrichment_by_core` | `07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb` |
| `calculate_stage3_categorical_radius_by_core` | `07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb` |
| `calculate_stage3_knn_continuum_by_core` | `07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb` |
| `calculate_stage3_nearest_at2_by_core` | `07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb` |
| `calculate_stage3_radius_continuum_by_core` | `07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb` |
| `cluster_expression_summary` | `02_Myeloid_Refinement.ipynb` |
| `compute_program_scores` | Reusable library function; not called directly by the current notebooks |
| `configure_plot_style` | Reusable library function; not called directly by the current notebooks |
| `export_spatial_cellchat_inputs` | Reusable library function; not called directly by the current notebooks |
| `extract_marker_matrices` | Reusable library function; not called directly by the current notebooks |
| `gene_detection_by_group` | `02_1_AM_MHCII_Scoring.ipynb`<br>`03_1_AT2_Vim_Scoring.ipynb` |
| `marker_availability_table` | Reusable library function; not called directly by the current notebooks |
| `merge_obs_to_main` | `04_Merge_Obs_To_Main.ipynb`<br>`07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb`<br>`07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb`<br>`08_Spatial_Umap.ipynb` |
| `plot_am_at2_pct_by_donor` | `04_Merge_Obs_To_Main.ipynb` |
| `plot_am_at2_spatial` | Reusable library function; not called directly by the current notebooks |
| `plot_anndata_group_umap` | `02_1_AM_MHCII_Scoring.ipynb`<br>`02_MA_Refinement.ipynb`<br>`03_1_AT2_Vim_Scoring.ipynb` |
| `plot_core_contributions` | Reusable library function; not called directly by the current notebooks |
| `plot_focus_umap` | Reusable library function; not called directly by the current notebooks |
| `plot_full_umap` | Reusable library function; not called directly by the current notebooks |
| `plot_knn_niche_continuum` | Reusable library function; not called directly by the current notebooks |
| `plot_macrophage_pct_by_tissue` | `02_MA_Refinement.ipynb`<br>`02_Myeloid_Refinement.ipynb`<br>`03_Epi_Refinement.ipynb` |
| `plot_marker_dotplot` | Reusable library function; not called directly by the current notebooks |
| `plot_metadata_summary` | `01_Metadata_Summary.ipynb` |
| `plot_mhcii_at2_spatial_core` | `08_Spatial_Umap.ipynb` |
| `plot_mhcii_hi_proportion_by_age` | `09_MHCprop_Age_Correlation.ipynb` |
| `plot_nhood_enrichment_donor_tissue` | Reusable library function; not called directly by the current notebooks |
| `plot_program_umap` | Reusable library function; not called directly by the current notebooks |
| `plot_radius_core_correlations` | Reusable library function; not called directly by the current notebooks |
| `plot_spatial_celltypes` | Reusable library function; not called directly by the current notebooks |
| `plot_spatial_focus` | Reusable library function; not called directly by the current notebooks |
| `plot_spatial_programs` | Reusable library function; not called directly by the current notebooks |
| `plot_stage1A_niche_dotmap` | `05_1A_Unbiased_Niche_Discovery.ipynb` |
| `plot_stage1B_primary` | `05_1B_Focused_AMAT2_Validation.ipynb` |
| `plot_stage2_primary` | `06_AM_MHCII_AT2_Spatial_Association.ipynb` |
| `plot_stage2_scale_sensitivity` | `06_AM_MHCII_AT2_Spatial_Association.ipynb` |
| `plot_stage3_categorical_pair_heatmap` | `07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb` |
| `plot_stage3_categorical_primary` | `07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb` |
| `plot_stage3_categorical_scale_sensitivity` | `07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb` |
| `plot_stage3_primary` | `07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb` |
| `plot_stage3_scale_sensitivity` | `07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb` |
| `radius_weighted_mean` | Reusable library function; not called directly by the current notebooks |
| `rank_stage2_core_contributions` | `08_Spatial_Umap.ipynb` |
| `run_spatial_function_multicore` | `05_1A_Unbiased_Niche_Discovery.ipynb`<br>`05_1B_Focused_AMAT2_Validation.ipynb` |
| `run_stage2_multicore` | `06_AM_MHCII_AT2_Spatial_Association.ipynb` |
| `run_stage3_all_methods` | Reusable library function; not called directly by the current notebooks |
| `run_stage3_multicore` | `07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb`<br>`07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb` |
| `save_figure` | Reusable library function; not called directly by the current notebooks |
| `score_and_assign_two_signatures` | `02_MA_Refinement.ipynb`<br>`03_1_AT2_Vim_Scoring.ipynb` |
| `select_representative_cores` | Reusable library function; not called directly by the current notebooks |
| `select_supportive_cores` | `08_Spatial_Umap.ipynb` |
| `summarise_lr_by_donor` | Reusable library function; not called directly by the current notebooks |
| `summarize_lr_by_donor` | Reusable library function; not called directly by the current notebooks |
| `summarize_markers` | Reusable library function; not called directly by the current notebooks |
| `summarize_nhood_by_donor` | Reusable library function; not called directly by the current notebooks |
| `summarize_stage1_by_donor_and_tissue` | `05_1A_Unbiased_Niche_Discovery.ipynb`<br>`05_1B_Focused_AMAT2_Validation.ipynb` |
| `summarize_stage2_by_donor_and_tissue` | `06_AM_MHCII_AT2_Spatial_Association.ipynb` |
| `summarize_stage3_by_donor_and_tissue` | `07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb`<br>`07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb` |
| `test_continuous_mhcii_at2_proximity` | Reusable library function; not called directly by the current notebooks |
| `test_lr_across_donors` | Reusable library function; not called directly by the current notebooks |
| `validate_xenium_metadata` | Reusable library function; not called directly by the current notebooks |

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
| `rank_stage2_core_contributions` | Performs leave-one-core-out and leave-one-donor-out sensitivity analysis for one prespecified tissue/radius/hypothesis. It defaults to the primary Stage 2 estimator (equal-core arithmetic averaging within donor and a one-sided greater Wilcoxon test), with Fisher-z aggregation and two-sided testing available as explicit sensitivities. AM cell counts are descriptive, and rows flag when omitting a core also removes a single-core donor. |
| `select_supportive_cores` | Selects supportive cores from distinct donors plus optional typical and discordant/near-null examples for transparent spatial illustration. The returned `selection_is_inferential=False` flag records that these examples cannot replace all-donor inference. |
| `summarize_nhood_by_donor` | Summarizes core-level neighborhood enrichment within donor, tissue, radius, and neighbor type. Mean/median z-scores are descriptive rather than a formal meta-analysis. |
| `calculate_knn_niche_continuum` | Quantifies each AM's k-nearest-neighbor composition within core, calculates within-core score correlations, combines core correlations within donor using Fisher z, and tests donor correlations by tissue. |
| `calculate_radius_niche_continuum` | Quantifies fixed-radius neighbor count, fraction, density, or presence around each AM within core and performs core-, donor-, and tissue-level correlation summaries. |

### Stage 3 AM MHCII–AT2 VIM-state analysis

Stage 3 asks whether continuous or categorical AM MHCII state is spatially
coupled to AT2 VIM state. Continuous analyses are primary; categorical states
and balanced tails are sensitivity analyses. Effects retain their natural
sign: a negative effect means higher-MHCII AMs are near lower-VIM AT2 states,
consistent with the working hypothesis that adapted MHCII-high AMs associate
with canonical/VIM-low rather than transitional/VIM-high AT2. The Xu paper did
not directly test this VIM-specific distance hypothesis, so tissue tests are
two-sided by default. Core permutation P values are diagnostic; final
inference uses one equal-core summary per donor.

| Function | Role and output |
|---|---|
| `plot_anndata_group_umap` | Reusable single-panel, split, or highlighted categorical UMAP with explicit palettes and vector labels. |
| `score_and_assign_two_signatures` | Scores two gene sets and assigns the larger score, with optional low-score or small-margin ambiguity. Independently controlled Scanpy scores are heuristic and require marker/distribution review. |
| `calculate_stage3_balanced_extremes_by_core` | Secondary tie-safe comparison of local AT2 VIM score around equal MHCII-score tails at fixed radii. |
| `calculate_stage3_knn_continuum_by_core` | Primary within-core continuous MHCII–VIM kNN coupling; focal AM self-neighbors are excluded. |
| `calculate_stage3_radius_continuum_by_core` | Primary continuous MHCII–VIM coupling inside physical radii, with coverage diagnostics. |
| `calculate_stage3_nearest_at2_by_core` | Primary continuous MHCII coupling to the VIM score of the nearest eligible AT2. |
| `run_stage3_all_methods` | Runs the four continuous methods and returns tidy core tables. |
| `run_stage3_multicore` | Runs one Stage 3 method by donor × tissue × core with stable identity-derived seeds. |
| `summarize_stage3_by_donor_and_tissue` | Equal-core donor aggregation, donor signed-rank tests, and method/effect/scale-scoped FDR across tissues. |
| `plot_stage3_primary` | Donor forest plot at prespecified scales with medians, IQRs, and tissue-test annotations. |
| `plot_stage3_scale_sensitivity` | Descriptive donor median/IQR trajectories across k or radius values. |
| `calculate_stage3_categorical_pair_enrichment_by_core` | Four-state edge-count concordance log odds ratio; interpret with per-AM analyses. |
| `calculate_stage3_categorical_knn_by_core` | Balanced MHCIIhi-minus-MHCIIlo local VIMhi-fraction difference across kNN scales. |
| `calculate_stage3_categorical_radius_by_core` | Balanced categorical local-VIMhi difference across physical radii. |
| `calculate_stage3_categorical_nearest_at2_by_core` | Nearest-AT2 VIM-state odds ratio by AM MHCII group, requiring both AM states. |
| `plot_stage3_categorical_primary` | Primary-scale donor forest wrapper for categorical sensitivity methods. |
| `plot_stage3_categorical_scale_sensitivity` | Median/IQR scale sensitivity for categorical pair, kNN, and radius effects. |
| `plot_stage3_categorical_pair_heatmap` | Tissue panels of donor-median pair log2 observed/expected values or z-scores. |
| `plot_mhcii_hi_proportion_by_age` | Exploratory donor grouped-quasibinomial age trend; all AMs form the denominator. |

`plot_mhcii_at2_spatial_core` also supports `at2_display="group"` with
`AT2_VIM_group`; its historical default still draws all AT2 as one class. All
Stage 3 spatial functions group by donor, tissue, and core, convert declared
coordinates to micrometres once, and never construct cross-core neighborhoods.

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
