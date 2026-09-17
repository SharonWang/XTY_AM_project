# Stage 2 AM–AT2 Spatial Analysis Design

## Purpose

Extend the single-file Xenium analysis pipeline with donor-aware Stage 1 plotting,
parallel per-core execution, and Stage 2 analyses testing whether the continuous
alveolar-macrophage (AM) MHCII score is associated with AT2 location. Cycling AMs
remain outside the analysis. Score-defined high and low groups are sensitivity
contrasts, not new biological clusters.

## Public interface

The additions remain in `scripts/xty_am_pipeline.py`:

- `plot_stage1A_niche_dotmap`
- `plot_stage1B_primary`
- `run_spatial_function_multicore`
- `calculate_stage2_balanced_extremes_by_core`
- `calculate_stage2_knn_continuum_by_core`
- `calculate_stage2_radius_continuum_by_core`
- `calculate_stage2_nearest_at2_by_core`
- `run_stage2_multicore`
- `summarize_stage2_by_donor_and_tissue`

`run_stage2_multicore` is a compatibility wrapper over the generic spatial
runner. One private worker implementation is used so expression matrices are not
copied to worker processes.

## Stage 1 figures

The two supplied plotting bodies are preserved. Stage 1A displays median donor
effects, donor-level FDR, and target ranking across methods. Stage 1B displays
individual donor effects for the AM–AT2 comparison. Plotting must use the
donor-level outputs of `summarize_stage1_by_donor_and_tissue`, not core-level
permutation P values.

## Stage 2 estimands

### Balanced extremes

Within each core, rank AMs by `MHCIIhi_score`, select equal-sized top and bottom
fractions, and compare their mean local AT2 fraction. The reported effect is:

`mean AT2 fraction around high-score AMs - mean AT2 fraction around low-score AMs`.

The permutation null reallocates the observed local-exposure values between two
equal-sized extreme groups. A positive effect means greater AT2 exposure around
higher-scoring AMs.

### Continuous exposure

Within each core, calculate Spearman correlation between AM MHCII score and the
fraction of neighboring cells that are AT2. Neighborhoods are defined either by
the k nearest cells or by a physical radius. A positive correlation means that
higher-scoring AMs have a more AT2-rich neighborhood.

AMs with zero total neighbors at a requested radius are excluded from that
radius's effect calculation. Treating these cells as AT2 fraction zero would
confound absence of AT2 with absence of all measured neighboring cells. Outputs
record the total, analyzed, and excluded AM counts.

### Nearest AT2

Within each core, calculate Spearman correlation between AM MHCII score and
distance to the nearest AT2. Report `effect = -rho`, so a positive effect means
higher-scoring AMs are closer to AT2.

## Statistical hierarchy

Permutations shuffle AM scores or extreme-group assignments only within a core.
Core-level P values and FDR are diagnostic. They are not the final biological
replicate-level inference.

For final tissue inference, core effects are averaged within donor and tissue.
One-sided Wilcoxon signed-rank tests then test donor effects against zero with the
alternative that the median association is positive. Benjamini–Hochberg FDR is
applied separately within method, effect type, scale type, scale, and tissue.

## Validation

Reject nonpositive scales or permutation counts, invalid extreme fractions,
nonpositive worker counts, nonfinite or malformed spatial coordinates,
nonpositive coordinate scaling, mixed donor/tissue annotations within a core,
insufficient AM/AT2 counts, insufficient analyzable AMs, and invariant scores.
Returned empty tables retain documented columns.

## Reproducibility and HPC execution

The parallel runner passes only required observation metadata and two-dimensional
spatial coordinates. `numpy.random.SeedSequence` creates a stable per-core seed.
Joblib is imported lazily and each process is limited to one inner numerical
thread. Local tests use small synthetic AnnData objects; production execution is
intended for the documented HPC analysis directory.
