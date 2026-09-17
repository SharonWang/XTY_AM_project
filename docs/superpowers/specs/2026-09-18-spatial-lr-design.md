# Spatial AM–AT2 Ligand–Receptor Design

## Goal

Add two separate, auditable communication workflows: validated sparse export for
formal Spatial CellChat analysis in R, and an exploratory Python analysis of
continuous AM MHCII score versus spatially weighted ligand–receptor expression
co-occurrence. The Python score is not called a CellChat probability and is not
interpreted as causal signaling.

## Cell groups and export

Balanced MHCII tails are core-relative sensitivity groups. Tied tail boundaries
are not split arbitrarily. CellChat labels preserve non-AM types and distinguish
MHCII-high, MHCII-low, middle, and unassigned AMs. Export requires unique cell
and gene identifiers, finite nonnegative normalized/log1p expression, complete
metadata, and finite two-dimensional micrometre coordinates. It writes sparse
genes-by-cells Matrix Market data, genes, cells, metadata, coordinates, and a
JSON manifest.

## Python spatial co-occurrence

Only simple single-gene ligand/receptor pairs are eligible. For AM-to-AT2, the
per-AM score is AM ligand expression multiplied by distance-weighted local AT2
receptor expression. The reverse direction uses AM receptor and local AT2 ligand.
Within each core, Spearman correlation relates this score to continuous MHCII
score. Identity-stable pair-specific permutations provide two-sided empirical P
values. Outputs include expression prevalence, neighbor coverage, and flags for
overlap between the AM-side gene and the MHCII signature.

Core FDR is controlled across LR pairs within core, direction, and radius. Core
correlations are combined with equal-weight Fisher-z within donor/tissue. Donor
tests are two-sided Wilcoxon signed-rank tests; FDR is controlled across LR pairs
within tissue, direction, and radius. Donors are biological replicates.

## HPC execution

The main AnnData object is never captured by process workers. Per-core arrays and
small expression mappings are prepared before dispatch. Serial, thread, and
process results use seeds derived from core and pair identities and therefore do
not depend on row, core, radius, or LR-pair order.

## Public API

- `assign_balanced_mhcii_extremes`
- `add_cellchat_groups`
- `export_spatial_cellchat_inputs`
- `audit_lr_panel`
- `radius_weighted_mean`
- `calculate_lr_for_core_arrays`
- `calculate_continuous_spatial_lr`
- `summarize_lr_by_donor`
- `summarise_lr_by_donor` compatibility alias
- `test_lr_across_donors`
