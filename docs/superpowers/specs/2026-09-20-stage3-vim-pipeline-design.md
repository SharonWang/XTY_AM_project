# Stage 3 AM MHCII and AT2 VIM Pipeline Design

## Goal

Add the supplied reusable UMAP, signature-scoring, Stage 3 spatial-coupling,
spatial-core, and age-trend functions to the single source module
`scripts/xty_am_pipeline.py`. Document every public function in `README.md` and
record the scientific workflow in `XENIUM_AM_AT2_ANALYSIS_PROPOSAL.md`.

The longer Stage 3 attachment is the authoritative version. Its shorter
continuous-only predecessor is not added separately. The existing
`plot_mhcii_at2_spatial_core` function is extended rather than duplicated.

## Biological interpretation

MHCII-high AMs represent the adapted AM program. The manuscript reports that
adapted AMs are closer to total SP-C-positive AT2 cells, whereas VIM-high AT2
cells are a transitional state that accumulates when the adapted-AM program or
AM-derived fibronectin is lost. The Stage 3 analysis therefore retains the
natural sign of every effect:

- positive continuous coupling means higher AM MHCII scores occur near higher
  AT2 VIM scores;
- negative continuous coupling means higher AM MHCII scores occur near lower
  AT2 VIM scores;
- positive categorical concordance means enrichment of MHCII-high–VIM-high and
  MHCII-low–VIM-low pairs;
- negative categorical concordance means enrichment of MHCII-high–VIM-low and
  MHCII-low–VIM-high pairs.

The manuscript motivates a negative MHCII–VIM coupling hypothesis, but does not
directly test MHCII-state-specific distance to VIM-high AT2 cells. Consequently,
the default tissue-level test remains two-sided. A directional `less`
sensitivity is allowed only when declared before viewing the result.

## Public API

The source module will expose:

- `plot_anndata_group_umap`
- `score_and_assign_two_signatures`
- `calculate_stage3_balanced_extremes_by_core`
- `calculate_stage3_knn_continuum_by_core`
- `calculate_stage3_radius_continuum_by_core`
- `calculate_stage3_nearest_at2_by_core`
- `run_stage3_all_methods`
- `run_stage3_multicore`
- `summarize_stage3_by_donor_and_tissue`
- `plot_stage3_primary`
- `plot_stage3_scale_sensitivity`
- `calculate_stage3_categorical_pair_enrichment_by_core`
- `calculate_stage3_categorical_knn_by_core`
- `calculate_stage3_categorical_radius_by_core`
- `calculate_stage3_categorical_nearest_at2_by_core`
- `plot_stage3_categorical_primary`
- `plot_stage3_categorical_scale_sensitivity`
- `plot_stage3_categorical_pair_heatmap`
- `plot_mhcii_hi_proportion_by_age`

`plot_mhcii_at2_spatial_core` remains public and gains optional AT2-state
colouring while preserving its current default display.

## Signature scoring and visualization

`plot_anndata_group_umap` creates unsplit, subset-split, or highlighted UMAP
panels with exact physical dimensions and a shared vector legend. It performs
no biological inference and does not mutate the AnnData palette.

`score_and_assign_two_signatures` copies the selected expression source before
optional scaling, reports missing genes, computes both Scanpy scores against
the same expression background, and writes only score/group columns back to
the original AnnData object. Comparing two independently controlled Scanpy
scores is a heuristic classification rather than a fitted mixture model.
Categorical calls are therefore secondary to continuous scores. The function
will support an optional minimum score-margin criterion for ambiguous cells in
addition to the supplied low-score ambiguity rule.

## Continuous Stage 3 analysis

Continuous MHCII and VIM scores are primary. Each spatial core is analyzed
independently using four complementary views:

1. balanced high/low MHCII score tails within a radius;
2. correlation between MHCII score and local AT2 VIM score among the nearest
   cells;
3. correlation between MHCII score and local AT2 VIM score within fixed
   micrometre radii;
4. correlation between MHCII score and the nearest eligible AT2 cell's VIM
   score.

All-cell kNN excludes the focal AM itself before selecting `k` neighbors. Fixed
geometry is retained during permutation and AT2 VIM values are shuffled only
within the same donor, tissue, and core. Balanced tails are equal-sized and do
not split tied boundaries. Core grouping always includes donor and tissue so
reused core identifiers cannot combine samples. Coordinate values must be in
micrometres or be converted by an explicit positive scale factor.

Core-level permutation P values and FDR are diagnostic. Stable seeds derive
from core identity and analysis settings, making results independent of input
row order, core order, job count, or subsetting of unrelated cores.

## Categorical Stage 3 sensitivity analysis

Categorical MHCII-high/low and VIM-high/low analyses are secondary sensitivity
analyses because thresholding discards information and may exclude ambiguous
cells. Outputs report retention fractions for both cell types.

The four-state pair method reports a log concordance odds ratio and
pair-specific observed/expected summaries. kNN and radius methods report the
difference in local VIM-high AT2 fraction around MHCII-high versus MHCII-low
AMs. The nearest-AT2 method reports a log odds ratio. Pair-count methods are
interpreted alongside local-fraction methods because raw edge counts give more
influence to AMs in denser neighborhoods.

## Donor-level inference

Cores are technical/spatial sampling units and donors are biological
replicates. Valid core correlations are Fisher-z transformed and averaged with
equal core weight within donor; differences and log odds ratios are averaged
on their native additive scales. Cell counts are QC/precision descriptors and
do not become biological-replicate weights.

Tissue tests use one donor estimate per tissue. The default signed-rank test is
two-sided, with `greater` and `less` available as prespecified alternatives.
FDR is controlled within declared method, effect, and scale families across
tissues. Primary fixed scales are separated from multiscale sensitivity
results. Figures display donors, medians, and interquartile ranges rather than
cell- or core-level pseudo-replicates.

## Age trend

`plot_mhcii_hi_proportion_by_age` summarizes all AMs in the denominator,
including ambiguous and unassigned AMs. Cores are shown descriptively and the
model is fitted to donor-level grouped counts. The grouped quasibinomial model
is labelled exploratory because it does not replace the planned adjusted or
mixed-effects age analysis. Small-sample inference uses a t reference with
donor residual degrees of freedom, and tissue-specific filtering is supported.

## Source organization and compatibility

All code remains in `scripts/xty_am_pipeline.py`, as requested. Stage 3 private
helpers use a `_stage3_` prefix to avoid overwriting existing helpers such as
`_safe_spearman` and `_permutation_summary`. Existing public behavior is
preserved unless the supplied extension explicitly adds an optional argument.
Plot colours and visual geometry from the supplied functions are retained;
changes are limited to validation, truthful labels, and generalization needed
for reusable data inputs.

## Validation

Implementation follows test-driven development with synthetic AnnData-like
objects and small local subsets. Tests will cover:

- duplicate attachment resolution and public exports;
- UMAP split/highlight behavior and file output;
- signature validation, score assignment, and ambiguity handling;
- donor/tissue/core isolation;
- self-exclusion from all-cell kNN;
- tie-safe balanced tails;
- stable serial/parallel seeds;
- continuous and categorical effect signs;
- donor aggregation, alternatives, and FDR families;
- extended spatial-core plotting;
- age-model summaries and file output;
- docstrings and README function coverage.

The focused tests and the full repository test suite must pass locally before
commit, fast-forward merge to `main`, and push to GitHub. Large Xenium inputs
are not loaded during local tests.
