"""Reusable functions for the human lung Xenium AM and AT2 analysis.

The module performs no analysis at import time. Notebooks select paths, load
data, and call these functions explicitly. The published Macrophages label is
treated as a broad candidate pool, not an automatic alveolar-macrophage call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import re
from typing import Any
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    TwoSlopeNorm,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import false_discovery_control, spearmanr, wilcoxon

try:
    import squidpy as sq
except ImportError:
    class _MissingSquidpy:
        """Raise a focused error only when a Squidpy API is requested."""

        def __getattr__(self, name):
            raise ImportError(
                "calculate_nhood_enrichment_by_core requires squidpy. "
                "Install it in the HPC environment before calling this function."
            )

    sq = _MissingSquidpy()


def multipletests(pvals, alpha=0.05, method="fdr_bh"):
    """Return statsmodels-compatible Benjamini-Hochberg results via SciPy."""
    if method != "fdr_bh":
        raise ValueError("Only method='fdr_bh' is supported.")
    adjusted = false_discovery_control(np.asarray(pvals, dtype=float), method="bh")
    return adjusted <= alpha, adjusted, np.nan, np.nan


# Version-controlled palette registry for notebooks and future studies.
CELLTYPE_PALETTE: dict[str, str] = {
    "AT1": "#91A7FF", "AT2": "#FF8FA3", "Macrophages": "#F4B860",
    "Monocytes": "#B8A1E3", "DC": "#77C8C4", "Endothelial": "#73B9EE",
    "LymphaticEC": "#80CBC4", "Fibroblasts": "#A8D5BA",
    "SmoothMuscle": "#E4A0C4", "Pericytes": "#C8B6A6",
    "CD8T": "#9FA8DA", "CD4T": "#B39DDB", "NK": "#8ECAE6",
    "B": "#FFD166", "Plasma": "#F6BD60", "Mast": "#F28482",
    "Secretory": "#84A59D", "Multiciliated": "#90DBF4", "Basal": "#CDB4DB",
}
FOCUS_PALETTE = {
    "Macrophages": CELLTYPE_PALETTE["Macrophages"],
    "AT1": CELLTYPE_PALETTE["AT1"],
    "AT2": CELLTYPE_PALETTE["AT2"],
}
CONTEXT_GREY = "#D9D9D9"
DARK_TEXT = "#2B2B2B"
PROGRAM_CMAP = LinearSegmentedColormap.from_list(
    "macaron_diverging", ["#7EA6D8", "#F7F7F4", "#EE8B8B"]
)
EXPRESSION_CMAP = LinearSegmentedColormap.from_list(
    "macaron_expression", ["#F5F3F8", "#9BBFE0", "#E98A9B"]
)
TISSUE_PALETTE: dict[str, str] = {
    "A": "#F1B6B2", "B": "#CDB9DD", "V": "#AFCFE3",
    "None": "#D9D6D2",
}
SEX_PALETTE: dict[str, str] = {"F": "#D5B8DF", "M": "#A9D5CE"}
TMA_PALETTE: dict[str, str] = {
    "TMA1": "#F3C8A8", "TMA2": "#BFD8C2",
}
MACROPHAGE_SUBTYPE_PALETTE: dict[str, str] = {
    "AM": "#A9D6E5", "AM-like": "#F2A7A0",
    "LYVE1+ IM": "#BFD8B8", "MMP2+ IM": "#D2B7E5",
}

MARKER_MODULES: dict[str, tuple[str, ...]] = {
    "AT1 evidence": ("AGER", "SCEL"),
    "AT2 evidence": ("SFTPD", "PLA2G4F"),
    "AM evidence": ("MARCO", "APOE"),
    "IM alternative": ("LYVE1", "CD163", "FCGR3A", "MS4A4A"),
    "Monocyte alternative": ("FCN1", "S100A12", "IL1B", "CLEC4E"),
    "Pan-macrophage": ("CD68", "AIF1", "MPEG1", "TYROBP"),
}
CANONICAL_UNMEASURED_CHECKS = (
    "FABP4", "PPARG", "INHBA", "SIGLEC1", "ITGAX", "CD36", "ABCG1",
    "FOLR2", "MRC1", "C1QA", "C1QB", "C1QC", "SFTPC",
)
REQUIRED_OBS_COLUMNS = (
    "donor_id", "core_id", "tissue_annotation", "celltype_final",
    "x_centroid", "y_centroid", "n_counts", "n_genes",
)


def configure_plot_style() -> None:
    """Configure the shared Cell-style plotting theme.

    Parameters
    ----------
    None
        This function accepts no external parameters.

    Returns
    -------
    None
        Matplotlib and Seaborn settings are updated in the active process.
    """
    sns.set_theme(style="white", context="notebook")
    mpl.rcParams.update({
        "font.family": "Arial", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.edgecolor": DARK_TEXT, "axes.labelcolor": DARK_TEXT,
        "text.color": DARK_TEXT, "xtick.color": DARK_TEXT,
        "ytick.color": DARK_TEXT, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def validate_xenium_metadata(
    adata: Any,
    required_obs: Sequence[str] = REQUIRED_OBS_COLUMNS,
) -> dict[str, Any]:
    """Validate observation metadata and spatial embeddings.

    Parameters
    ----------
    adata
        AnnData-like object with obs, obs_names, obsm, n_obs and n_vars.
    required_obs
        Columns that must exist and contain no missing values.

    Returns
    -------
    dict
        Cell count, gene count, duplicate identifier count and missing-value
        counts for required columns.

    Raises
    ------
    ValueError
        If columns, unique identifiers, two-dimensional embeddings or complete
        required metadata are absent.
    """
    missing = sorted(set(required_obs).difference(adata.obs.columns))
    if missing:
        raise ValueError(f"Missing required obs columns: {missing}")
    if not adata.obs_names.is_unique:
        raise ValueError("Cell identifiers are not unique")
    for key in ("X_umap", "spatial"):
        if key not in adata.obsm or adata.obsm[key].shape != (adata.n_obs, 2):
            raise ValueError(f"{key} must exist with shape ({adata.n_obs}, 2)")
    missing_values = adata.obs[list(required_obs)].isna().sum()
    if int(missing_values.sum()):
        raise ValueError(
            f"Required metadata contain missing values: "
            f"{missing_values[missing_values.gt(0)].to_dict()}"
        )
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "duplicate_cell_ids": int(adata.n_obs - adata.obs_names.nunique()),
        "missing_required_values": {
            str(key): int(value) for key, value in missing_values.items()
        },
    }


def select_representative_cores(
    obs: pd.DataFrame,
    n_cores: int = 4,
    tissue_code: str = "A",
    required_celltypes: Sequence[str] = ("Macrophages", "AT1", "AT2"),
) -> tuple[list[str], pd.DataFrame]:
    """Select deterministic complete cores across the core-size distribution.

    Parameters
    ----------
    obs
        Cell metadata containing core_id, donor_id, tissue_annotation and
        celltype_final.
    n_cores
        Number of intact cores to select for bounded spatial display.
    tissue_code
        Required anatomical code. A denotes alveolar tissue in this dataset.
    required_celltypes
        Source labels that must each occur in an eligible core.

    Returns
    -------
    selected_cores
        Ordered list of selected core identifiers.
    core_summary
        Core-level donor, region, total-cell and required-cell-type counts.

    Raises
    ------
    ValueError
        If inputs are missing or too few complete cores are available.
    """
    needed = {"core_id", "donor_id", "tissue_annotation", "celltype_final"}
    missing = sorted(needed.difference(obs.columns))
    if missing:
        raise ValueError(f"Missing core-selection columns: {missing}")
    if n_cores < 1:
        raise ValueError("n_cores must be at least 1")
    work = obs.copy()
    work["core_id_string"] = work["core_id"].astype(str)
    work["donor_id_string"] = work["donor_id"].astype(str)
    work["celltype_string"] = work["celltype_final"].astype(str)
    counts = pd.crosstab(work["core_id_string"], work["celltype_string"])
    absent = [label for label in required_celltypes if label not in counts]
    if absent:
        raise ValueError(f"Required cell types absent from dataset: {absent}")
    metadata = work.groupby("core_id_string", observed=True).agg(
        donor_id=("donor_id_string", "first"),
        tissue_annotation=("tissue_annotation", "first"),
        n_cells=("celltype_string", "size"),
    )
    summary = metadata.join(counts[list(required_celltypes)], how="left").fillna(0)
    summary.index.name = "core_id"
    complete = summary["tissue_annotation"].astype(str).eq(str(tissue_code))
    for label in required_celltypes:
        complete &= summary[label].gt(0)
    eligible = summary.loc[complete].sort_values("n_cells")
    if len(eligible) < n_cores:
        raise ValueError(
            f"Requested {n_cores} cores but only {len(eligible)} are eligible"
        )
    positions = np.linspace(0, len(eligible) - 1, n_cores).round().astype(int)
    return eligible.iloc[positions].index.astype(str).tolist(), summary


def marker_availability_table(
    var_names: Sequence[str],
    marker_modules: Mapping[str, Sequence[str]] = MARKER_MODULES,
    additional_checks: Sequence[str] = CANONICAL_UNMEASURED_CHECKS,
) -> tuple[pd.DataFrame, dict[str, bool], list[str]]:
    """Report requested-marker availability in a targeted gene panel.

    Parameters
    ----------
    var_names
        Gene names measured in the expression object.
    marker_modules
        Ordered mapping from evidence module to requested genes.
    additional_checks
        Canonical genes to report even when absent from the panel.

    Returns
    -------
    availability_table
        Gene, module and Boolean availability table.
    availability
        Gene-to-availability mapping for summaries and JSON output.
    measured_marker_order
        Present module genes in intended display order.
    """
    available_names = set(map(str, var_names))
    module_genes = [gene for genes in marker_modules.values() for gene in genes]
    requested = list(dict.fromkeys(module_genes + list(additional_checks)))
    availability = {gene: gene in available_names for gene in requested}
    rows = []
    for gene in requested:
        module = next(
            (name for name, genes in marker_modules.items() if gene in genes),
            "canonical check",
        )
        rows.append({"gene": gene, "available": availability[gene], "module": module})
    return (
        pd.DataFrame(rows),
        availability,
        [gene for gene in module_genes if availability[gene]],
    )


def extract_marker_matrices(
    adata: Any,
    row_indices: Sequence[int] | np.ndarray,
    genes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract transformed and raw matrices in identical requested gene order.

    Parameters
    ----------
    adata
        AnnData-like object with transformed X and integer-like raw.X.
    row_indices
        Integer cell positions to extract.
    genes
        Ordered gene names required in transformed and raw matrices.

    Returns
    -------
    transformed_values
        Dense cell-by-gene transformed-expression array.
    raw_counts
        Dense cell-by-gene raw-count array in the same order.
    zero_pattern_concordance
        Fraction agreeing on zero versus non-zero status.

    Raises
    ------
    ValueError
        If raw data, genes, shapes, integer-like values or zero-pattern
        agreement are invalid.
    """
    if adata.raw is None:
        raise ValueError("adata.raw is required for transcript detection")
    genes = list(map(str, genes))
    missing_x = sorted(set(genes).difference(map(str, adata.var_names)))
    missing_raw = sorted(set(genes).difference(map(str, adata.raw.var_names)))
    if missing_x or missing_raw:
        raise ValueError(f"Missing transformed={missing_x}; missing raw={missing_raw}")
    rows = np.asarray(row_indices, dtype=int)
    transformed = adata[rows, genes].X
    raw_columns = pd.Index(adata.raw.var_names.astype(str)).get_indexer(genes)
    raw_values = adata.raw.X[rows, :][:, raw_columns]
    transformed = transformed.toarray() if sparse.issparse(transformed) else np.asarray(transformed)
    raw_values = raw_values.toarray() if sparse.issparse(raw_values) else np.asarray(raw_values)
    if transformed.shape != raw_values.shape:
        raise ValueError("Transformed and raw marker shapes differ")
    if not np.allclose(raw_values, np.round(raw_values)):
        raise ValueError("raw.X marker values are not integer-like")
    concordance = float(np.mean((transformed > 0) == (raw_values > 0)))
    if concordance <= 0.999:
        raise ValueError(f"Zero-pattern concordance is {concordance:.6f}")
    return transformed, raw_values, concordance


def summarize_markers(
    transformed_values: np.ndarray,
    raw_counts: np.ndarray,
    obs: pd.DataFrame,
    genes: Sequence[str],
    marker_modules: Mapping[str, Sequence[str]],
    groups: Sequence[str] = ("AT1", "AT2", "Macrophages", "Monocytes"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate core-, donor- and donor-balanced marker summaries.

    Parameters
    ----------
    transformed_values
        Cell-by-gene transformed-expression matrix.
    raw_counts
        Cell-by-gene raw-count matrix with identical row and gene order.
    obs
        Row-aligned metadata with donor_id, core_id and celltype_final.
    genes
        Gene names corresponding to matrix columns.
    marker_modules
        Mapping from module name to genes. Each gene must map once.
    groups
        Source cell-type groups to summarize.

    Returns
    -------
    donor_balanced
        Group-by-gene table in which every donor contributes equal weight.
    by_core
        Core-, group- and gene-level expression and detection summaries.
    by_donor
        Donor-, group- and gene-level summaries. Multiple cores within a
        donor are combined using cell-count weights.

    Raises
    ------
    ValueError
        If matrices, metadata, groups or module assignments are invalid.
    """
    transformed_values = np.asarray(transformed_values)
    raw_counts = np.asarray(raw_counts)
    genes = list(map(str, genes))
    if transformed_values.shape != raw_counts.shape:
        raise ValueError("Transformed and raw marker matrices must match")
    if transformed_values.shape != (len(obs), len(genes)):
        raise ValueError("Matrix shape does not match obs rows and genes")
    missing = sorted(
        {"donor_id", "core_id", "celltype_final"}.difference(obs.columns)
    )
    if missing:
        raise ValueError(f"Missing marker-summary metadata: {missing}")

    gene_modules: dict[str, str] = {}
    for module, module_genes in marker_modules.items():
        for gene in module_genes:
            if gene in gene_modules:
                raise ValueError(f"Gene {gene} occurs in multiple modules")
            gene_modules[str(gene)] = str(module)
    unmapped = sorted(set(genes).difference(gene_modules))
    if unmapped:
        raise ValueError(f"Genes lack module assignments: {unmapped}")

    work = obs.reset_index(drop=True).copy()
    work["donor_id"] = work["donor_id"].astype(str)
    work["core_id"] = work["core_id"].astype(str)
    work["source_celltype"] = work["celltype_final"].astype(str)
    rows: list[dict[str, Any]] = []
    for (core, donor, source_group), frame in work.groupby(
        ["core_id", "donor_id", "source_celltype"], observed=True
    ):
        if source_group not in groups:
            continue
        positions = frame.index.to_numpy()
        for column, gene in enumerate(genes):
            rows.append({
                "core_id": core,
                "donor_id": donor,
                "source_celltype": source_group,
                "gene": gene,
                "module": gene_modules[gene],
                "n_cells": int(len(positions)),
                "mean_log_normalized_expression": float(
                    transformed_values[positions, column].mean()
                ),
                "fraction_detected_raw_gt_0": float(
                    (raw_counts[positions, column] > 0).mean()
                ),
            })
    by_core = pd.DataFrame(rows)
    missing_groups = sorted(set(groups).difference(by_core["source_celltype"].unique()))
    if missing_groups:
        raise ValueError(f"No cells found for source groups: {missing_groups}")

    donor_rows: list[dict[str, Any]] = []
    for keys, frame in by_core.groupby(
        ["donor_id", "source_celltype", "gene", "module"], observed=True
    ):
        donor, source_group, gene, module = keys
        weights = frame["n_cells"].to_numpy()
        donor_rows.append({
            "donor_id": donor,
            "source_celltype": source_group,
            "gene": gene,
            "module": module,
            "n_cells": int(weights.sum()),
            "mean_log_normalized_expression": float(np.average(
                frame["mean_log_normalized_expression"], weights=weights
            )),
            "fraction_detected_raw_gt_0": float(np.average(
                frame["fraction_detected_raw_gt_0"], weights=weights
            )),
        })
    by_donor = pd.DataFrame(donor_rows)
    donor_balanced = (
        by_donor.groupby(
            ["source_celltype", "gene", "module"],
            as_index=False,
            observed=True,
        )
        .agg(
            n_donors=("donor_id", "nunique"),
            n_cells=("n_cells", "sum"),
            mean_log_normalized_expression=(
                "mean_log_normalized_expression", "mean"
            ),
            donor_sd_log_normalized_expression=(
                "mean_log_normalized_expression", "std"
            ),
            fraction_detected_raw_gt_0=("fraction_detected_raw_gt_0", "mean"),
            donor_sd_fraction_detected=("fraction_detected_raw_gt_0", "std"),
        )
    )
    return donor_balanced, by_core, by_donor


def compute_program_scores(
    expression: pd.DataFrame,
    obs: pd.DataFrame,
    program_modules: Mapping[str, Sequence[str]],
    candidate_label: str = "Macrophages",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate standardized evidence programs within a candidate cell pool.

    Parameters
    ----------
    expression
        Cell-by-gene transformed-expression DataFrame.
    obs
        Index-aligned metadata containing donor_id, core_id and celltype_final.
    program_modules
        Mapping from program name to measured genes. Genes are standardized
        across candidate cells before their values are averaged.
    candidate_label
        Source label defining the candidate pool. Macrophages is a broad
        candidate label and is not interpreted as a final AM call.

    Returns
    -------
    per_cell
        Candidate cells with donor, core and program-score columns.
    by_core
        Mean program scores for each donor and core.
    by_donor
        Mean program scores for each donor.

    Raises
    ------
    ValueError
        If metadata, indices, candidate cells or measured program genes are
        invalid.
    """
    missing = sorted(
        {"donor_id", "core_id", "celltype_final"}.difference(obs.columns)
    )
    if missing:
        raise ValueError(f"Missing program-score metadata: {missing}")
    if not expression.index.equals(obs.index):
        raise ValueError("Expression and obs indices must match exactly")
    candidate_mask = obs["celltype_final"].astype(str).eq(str(candidate_label))
    candidate_expression = expression.loc[candidate_mask].copy()
    if candidate_expression.empty:
        raise ValueError(f"No cells found for candidate label {candidate_label}")

    standard_deviations = candidate_expression.std(axis=0, ddof=0).replace(
        0, np.nan
    )
    standardized = (
        candidate_expression - candidate_expression.mean(axis=0)
    ) / standard_deviations
    scores = pd.DataFrame(index=candidate_expression.index)
    for program_name, requested_genes in program_modules.items():
        measured = [gene for gene in requested_genes if gene in standardized]
        if not measured:
            raise ValueError(f"No measured genes for program {program_name}")
        scores[str(program_name)] = standardized[measured].mean(
            axis=1, skipna=True
        )
    program_names = list(map(str, program_modules))
    per_cell = obs.loc[
        scores.index, ["donor_id", "core_id"]
    ].copy().join(scores)
    by_core = (
        per_cell.groupby(["donor_id", "core_id"], observed=True)[program_names]
        .mean()
        .reset_index()
    )
    by_donor = (
        per_cell.groupby("donor_id", observed=True)[program_names]
        .mean()
        .reset_index()
    )
    return per_cell, by_core, by_donor


def save_figure(
    figure: Figure,
    output_directory: str | Path,
    stem: str,
    dpi: int = 300,
    close: bool = True,
) -> tuple[Path, Path]:
    """Save a figure as a review-ready PNG and editable-text PDF.

    Parameters
    ----------
    figure
        Matplotlib figure to save.
    output_directory
        Destination directory, created when absent.
    stem
        Filename without an extension.
    dpi
        PNG resolution in dots per inch.
    close
        Whether to close the figure after saving.

    Returns
    -------
    png_path
        Path to the generated PNG.
    pdf_path
        Path to the generated PDF.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    png_path = output_directory / f"{stem}.png"
    pdf_path = output_directory / f"{stem}.pdf"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(figure)
    return png_path, pdf_path


def _clean_embedding_axis(
    axis: mpl.axes.Axes,
    x_label: str = "UMAP 1",
    y_label: str = "UMAP 2",
) -> None:
    """Apply the shared minimal formatting to an embedding axis."""
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[["top", "right"]].set_visible(False)


def plot_full_umap(
    umap: np.ndarray,
    labels: Sequence[str],
    palette: Mapping[str, str] = CELLTYPE_PALETTE,
    title: str = "Published human lung Xenium atlas",
    point_size: float = 1.4,
) -> Figure:
    """Plot every cell in a deposited two-dimensional UMAP.

    Parameters
    ----------
    umap
        Cell-by-two deposited UMAP coordinates.
    labels
        Source cell-type label for every coordinate row.
    palette
        Mapping from source label to plotting color.
    title
        Figure title.
    point_size
        Marker area passed to Matplotlib scatter.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing all input cells; the caller controls saving.

    Raises
    ------
    ValueError
        If coordinate and label lengths differ or UMAP is not two-dimensional.
    """
    coordinates = np.asarray(umap)
    labels = np.asarray(labels, dtype=str)
    if coordinates.shape != (len(labels), 2):
        raise ValueError(
            f"UMAP shape {coordinates.shape} is incompatible with "
            f"{len(labels)} labels"
        )
    figure, axis = plt.subplots(figsize=(10.5, 8.0))
    counts = pd.Series(labels).value_counts()
    for label in counts.index:
        positions = np.flatnonzero(labels == label)
        axis.scatter(
            coordinates[positions, 0],
            coordinates[positions, 1],
            s=point_size,
            c=palette.get(label, "#BDBDBD"),
            alpha=0.78,
            linewidths=0,
            rasterized=True,
            label=f"{label} ({counts[label]:,})",
        )
    _clean_embedding_axis(axis)
    axis.set_title(title, loc="left", fontsize=15, weight="bold", y=1.075)
    axis.text(
        0, 1.01, f"Source UMAP; all {len(labels):,} cells plotted",
        transform=axis.transAxes, fontsize=9, color="#666666",
    )
    axis.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
        markerscale=2.5, fontsize=8,
    )
    return figure


def plot_focus_umap(
    umap: np.ndarray,
    labels: Sequence[str],
    focus_labels: Sequence[str] = ("Macrophages", "AT1", "AT2"),
    focus_palette: Mapping[str, str] = FOCUS_PALETTE,
    context_grey: str = CONTEXT_GREY,
    title: str = "Macrophage, AT1, and AT2 source labels",
) -> Figure:
    """Highlight selected source labels on the deposited UMAP.

    Parameters
    ----------
    umap
        Cell-by-two deposited UMAP coordinates.
    labels
        Source cell-type label for every coordinate row.
    focus_labels
        Labels to plot in color; all other cells form pale-grey context.
    focus_palette
        Color mapping for focused labels.
    context_grey
        Color used for non-focused cells.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Focused UMAP figure containing every input cell.

    Raises
    ------
    ValueError
        If coordinates and labels are misaligned or a focus color is absent.
    """
    coordinates = np.asarray(umap)
    labels = np.asarray(labels, dtype=str)
    if coordinates.shape != (len(labels), 2):
        raise ValueError("UMAP coordinates and labels do not align")
    missing_colors = sorted(set(focus_labels).difference(focus_palette))
    if missing_colors:
        raise ValueError(f"Missing focus colors: {missing_colors}")
    figure, axis = plt.subplots(figsize=(9.2, 7.2))
    context = ~np.isin(labels, focus_labels)
    axis.scatter(
        coordinates[context, 0], coordinates[context, 1], s=2.2,
        c=context_grey, alpha=0.24, linewidths=0, rasterized=True,
    )
    for label in focus_labels:
        positions = np.flatnonzero(labels == label)
        axis.scatter(
            coordinates[positions, 0], coordinates[positions, 1], s=4.5,
            c=focus_palette[label], alpha=0.8, linewidths=0,
            rasterized=True, label=label,
        )
    _clean_embedding_axis(axis)
    axis.set_title(title, loc="left", fontsize=15, weight="bold", y=1.075)
    axis.text(
        0, 1.01,
        "Macrophages = candidate pool only; other labels in pale grey",
        transform=axis.transAxes, fontsize=9, color="#666666",
    )
    axis.legend(frameon=False, markerscale=2.8)
    return figure


def plot_marker_dotplot(
    marker_summary: pd.DataFrame,
    gene_order: Sequence[str],
    group_order: Sequence[str],
    title: str = "Donor-balanced marker evidence",
) -> Figure:
    """Plot donor-balanced marker means and detection fractions.

    Parameters
    ----------
    marker_summary
        First table returned by summarize_markers.
    gene_order
        Genes displayed from left to right.
    group_order
        Source groups displayed from top to bottom.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Dotplot whose color is transformed-expression mean and whose area is
        raw-transcript detection fraction.

    Raises
    ------
    ValueError
        If required marker-summary columns are absent.
    """
    required = {
        "source_celltype", "gene", "mean_log_normalized_expression",
        "fraction_detected_raw_gt_0",
    }
    missing = sorted(required.difference(marker_summary.columns))
    if missing:
        raise ValueError(f"Missing dotplot columns: {missing}")
    x_positions = {gene: index for index, gene in enumerate(gene_order)}
    reversed_groups = list(group_order)[::-1]
    y_positions = {group: index for index, group in enumerate(reversed_groups)}
    plot_data = marker_summary[
        marker_summary["gene"].isin(gene_order)
        & marker_summary["source_celltype"].isin(group_order)
    ].copy()
    figure, axis = plt.subplots(figsize=(14.5, 4.7))
    scatter = axis.scatter(
        plot_data["gene"].map(x_positions),
        plot_data["source_celltype"].map(y_positions),
        s=22 + 330 * plot_data["fraction_detected_raw_gt_0"],
        c=plot_data["mean_log_normalized_expression"],
        cmap=EXPRESSION_CMAP, linewidths=0.45, edgecolors="white",
    )
    axis.set_xticks(range(len(gene_order)), gene_order, rotation=55, ha="right")
    axis.set_yticks(range(len(reversed_groups)), reversed_groups)
    axis.set_xlim(-0.7, len(gene_order) - 0.3)
    axis.set_ylim(-0.7, len(reversed_groups) - 0.3)
    axis.grid(axis="x", color="#EEEEEE", linewidth=0.6)
    axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    axis.tick_params(length=0)
    axis.set_title(title, loc="left", fontsize=15, weight="bold", y=1.13)
    axis.text(
        0, 1.045,
        "Equal-weight mean across donors | Color: expression | Area: detection",
        transform=axis.transAxes, fontsize=9, color="#666666",
    )
    figure.colorbar(
        scatter, ax=axis, pad=0.02, fraction=0.025,
        label="Mean log-normalized expression",
    )
    return figure


def plot_program_umap(
    umap: np.ndarray,
    candidate_indices: Sequence[int],
    program_scores: pd.DataFrame,
    color_map: mpl.colors.Colormap = PROGRAM_CMAP,
    color_limit: float = 2.5,
    title: str = "Competing evidence inside the macrophage candidate pool",
) -> Figure:
    """Plot standardized candidate-cell program scores on UMAP.

    Parameters
    ----------
    umap
        Full cell-by-two UMAP coordinate array.
    candidate_indices
        Global integer positions corresponding row-for-row to program_scores.
    program_scores
        Candidate-cell DataFrame with one numeric column per program.
    color_map
        Diverging colormap for standardized scores.
    color_limit
        Symmetric clipping and color-normalization limit around zero.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        One UMAP facet per program plus a shared color bar.

    Raises
    ------
    ValueError
        If indices and score rows differ, no programs exist, or limits fail.
    """
    coordinates = np.asarray(umap)
    indices = np.asarray(candidate_indices, dtype=int)
    if len(indices) != len(program_scores):
        raise ValueError("candidate_indices must align with program_scores")
    if program_scores.shape[1] < 1:
        raise ValueError("At least one program score is required")
    if color_limit <= 0:
        raise ValueError("color_limit must be positive")
    figure, axes = plt.subplots(
        1, program_scores.shape[1],
        figsize=(4.7 * program_scores.shape[1], 4.7),
        squeeze=False, constrained_layout=True,
    )
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit)
    last_scatter = None
    for axis, program in zip(axes.flat, program_scores.columns):
        values = program_scores[program].clip(-color_limit, color_limit)
        last_scatter = axis.scatter(
            coordinates[indices, 0], coordinates[indices, 1],
            c=values, cmap=color_map, norm=norm, s=7, alpha=0.88,
            linewidths=0, rasterized=True,
        )
        _clean_embedding_axis(axis)
        axis.set_title(str(program), fontsize=11, weight="bold")
    figure.suptitle(title, x=0.01, ha="left", fontsize=15, weight="bold")
    figure.colorbar(
        last_scatter, ax=axes, shrink=0.72, pad=0.02,
        label="Within-pool standardized score",
    )
    return figure


def plot_spatial_celltypes(
    spatial: np.ndarray,
    obs: pd.DataFrame,
    core_ids: Sequence[str],
    palette: Mapping[str, str] = CELLTYPE_PALETTE,
    n_columns: int = 2,
    title: str = "Published cell types in complete alveolar Xenium cores",
) -> Figure:
    """Plot complete spatial cores without mixing coordinate systems.

    Parameters
    ----------
    spatial
        Cell-by-two spatial centroid coordinates.
    obs
        Row-aligned metadata with core_id and celltype_final.
    core_ids
        Cores to facet. Every cell from each selected core is retained.
    palette
        Mapping from source cell-type labels to colors.
    n_columns
        Number of facet columns.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Faceted spatial figure with equal aspect and image-coordinate y-axis.

    Raises
    ------
    ValueError
        If coordinates, metadata, core selection or column count are invalid.
    """
    spatial = np.asarray(spatial)
    if spatial.shape != (len(obs), 2):
        raise ValueError("Spatial coordinates must align with obs")
    missing = sorted({"core_id", "celltype_final"}.difference(obs.columns))
    if missing:
        raise ValueError(f"Missing spatial metadata: {missing}")
    if not core_ids or n_columns < 1:
        raise ValueError("At least one core and one facet column are required")
    n_rows = int(np.ceil(len(core_ids) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(6.4 * n_columns, 5.8 * n_rows),
        squeeze=False, constrained_layout=True,
    )
    core_labels = obs["core_id"].astype(str).to_numpy()
    celltypes = obs["celltype_final"].astype(str).to_numpy()
    order = pd.Series(celltypes).value_counts().index
    for axis, core in zip(axes.flat, map(str, core_ids)):
        positions = np.flatnonzero(core_labels == core)
        for label in order[::-1]:
            selected = positions[celltypes[positions] == label]
            if len(selected):
                axis.scatter(
                    spatial[selected, 0], spatial[selected, 1], s=2.2,
                    c=palette.get(label, "#BDBDBD"), alpha=0.82,
                    linewidths=0, rasterized=True,
                )
        axis.set_title(
            f"Core {core} | n={len(positions):,}",
            loc="left", fontsize=10, weight="bold",
        )
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
    for axis in axes.flat[len(core_ids):]:
        axis.set_visible(False)
    selected_core_mask = np.isin(core_labels, np.asarray(core_ids, dtype=str))
    present_labels = set(celltypes[selected_core_mask])
    legend_order = [label for label in palette if label in present_labels]
    legend_order.extend(sorted(present_labels.difference(legend_order)))
    handles = [
        Line2D(
            [], [], marker="o", linestyle="", markersize=5,
            markerfacecolor=palette.get(label, "#BDBDBD"),
            markeredgecolor="none", label=label,
        )
        for label in legend_order
    ]
    figure.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
        title="Published cell type",
        fontsize=8,
        title_fontsize=9,
    )
    figure.suptitle(title, x=0.01, ha="left", fontsize=16, weight="bold")
    return figure


def plot_spatial_focus(
    spatial: np.ndarray,
    obs: pd.DataFrame,
    core_ids: Sequence[str],
    focus_labels: Sequence[str] = ("Macrophages", "AT1", "AT2"),
    focus_palette: Mapping[str, str] = FOCUS_PALETTE,
    context_grey: str = CONTEXT_GREY,
    n_columns: int = 2,
    title: str = "Macrophage candidates and alveolar epithelial labels",
) -> Figure:
    """Highlight macrophage, AT1 and AT2 labels in complete spatial cores.

    Parameters
    ----------
    spatial
        Cell-by-two spatial centroid coordinates.
    obs
        Row-aligned metadata with core_id and celltype_final.
    core_ids
        Complete cores to facet.
    focus_labels
        Source labels plotted in color.
    focus_palette
        Color mapping for focused labels.
    context_grey
        Color for all non-focused tissue context.
    n_columns
        Number of facet columns.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Faceted complete-core spatial focus figure.
    """
    spatial = np.asarray(spatial)
    if spatial.shape != (len(obs), 2):
        raise ValueError("Spatial coordinates must align with obs")
    n_rows = int(np.ceil(len(core_ids) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(6.2 * n_columns, 5.7 * n_rows),
        squeeze=False, constrained_layout=True,
    )
    cores = obs["core_id"].astype(str).to_numpy()
    labels = obs["celltype_final"].astype(str).to_numpy()
    for axis, core in zip(axes.flat, map(str, core_ids)):
        positions = np.flatnonzero(cores == core)
        axis.scatter(
            spatial[positions, 0], spatial[positions, 1], s=1.6,
            c=context_grey, alpha=0.24, linewidths=0, rasterized=True,
        )
        for label in focus_labels:
            selected = positions[labels[positions] == label]
            axis.scatter(
                spatial[selected, 0], spatial[selected, 1], s=7,
                c=focus_palette[label], alpha=0.88, linewidths=0,
                rasterized=True, label=label,
            )
        axis.set_title(f"Core {core}", loc="left", fontsize=10, weight="bold")
        axis.set_aspect("equal")
        axis.invert_yaxis()
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
    for axis in axes.flat[len(core_ids):]:
        axis.set_visible(False)
    handles = [
        Line2D(
            [], [], marker="o", linestyle="", markersize=6,
            markerfacecolor=focus_palette[label], markeredgecolor="none",
            label=label,
        )
        for label in focus_labels
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=len(handles),
        frameon=False,
        title="Source label",
    )
    figure.suptitle(title, x=0.01, ha="left", fontsize=16, weight="bold")
    return figure


def plot_spatial_programs(
    spatial: np.ndarray,
    obs: pd.DataFrame,
    core_ids: Sequence[str],
    candidate_global_indices: Sequence[int],
    program_scores: pd.DataFrame,
    context_grey: str = CONTEXT_GREY,
    color_map: mpl.colors.Colormap = PROGRAM_CMAP,
    color_limit: float = 2.5,
    title: str = "Competing macrophage identity evidence in intact cores",
) -> Figure:
    """Plot candidate-cell program scores within complete spatial cores.

    Parameters
    ----------
    spatial
        Full cell-by-two spatial centroid coordinates.
    obs
        Row-aligned metadata with core_id.
    core_ids
        Complete cores displayed as rows.
    candidate_global_indices
        Global cell positions aligned row-for-row to program_scores.
    program_scores
        Candidate-cell DataFrame with one numeric column per program.
    context_grey
        Color for all tissue-context cells.
    color_map
        Diverging colormap for standardized program scores.
    color_limit
        Symmetric clipping and normalization limit.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Core-by-program spatial facet grid plus a shared color bar.

    Raises
    ------
    ValueError
        If coordinates, metadata, indices, scores or limits are invalid.
    """
    spatial = np.asarray(spatial)
    indices = np.asarray(candidate_global_indices, dtype=int)
    if spatial.shape != (len(obs), 2):
        raise ValueError("Spatial coordinates must align with obs")
    if len(indices) != len(program_scores):
        raise ValueError("Candidate indices must align with program scores")
    if not core_ids or program_scores.shape[1] < 1 or color_limit <= 0:
        raise ValueError("Cores, programs and a positive color limit are required")
    figure, axes = plt.subplots(
        len(core_ids), program_scores.shape[1],
        figsize=(4.7 * program_scores.shape[1], 4.0 * len(core_ids)),
        squeeze=False, constrained_layout=True,
    )
    core_labels = obs["core_id"].astype(str).to_numpy()
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit)
    last_scatter = None
    for row, core in enumerate(map(str, core_ids)):
        context = np.flatnonzero(core_labels == core)
        candidate_rows = np.flatnonzero(core_labels[indices] == core)
        selected_global = indices[candidate_rows]
        for column, program in enumerate(program_scores.columns):
            axis = axes[row, column]
            axis.scatter(
                spatial[context, 0], spatial[context, 1], s=1.3,
                c=context_grey, alpha=0.18, linewidths=0, rasterized=True,
            )
            last_scatter = axis.scatter(
                spatial[selected_global, 0], spatial[selected_global, 1],
                c=program_scores.iloc[candidate_rows][program].clip(
                    -color_limit, color_limit
                ),
                cmap=color_map, norm=norm, s=8, alpha=0.9,
                linewidths=0, rasterized=True,
            )
            if row == 0:
                axis.set_title(str(program), fontsize=11, weight="bold")
            if column == 0:
                axis.set_ylabel(f"Core {core}", fontsize=10, weight="bold")
            axis.set_aspect("equal")
            axis.invert_yaxis()
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[:].set_visible(False)
    figure.suptitle(title, x=0.01, ha="left", fontsize=16, weight="bold")
    figure.colorbar(
        last_scatter, ax=axes, shrink=0.45, pad=0.015,
        label="Within-pool standardized score",
    )
    return figure


# -----------------------------------------------------------------------------
# General expression, orthologue, and AnnData metadata utilities
# -----------------------------------------------------------------------------


def cluster_expression_summary(
    adata: Any,
    genes: Sequence[str],
    groupby: str,
    layer: str | None = None,
) -> pd.DataFrame:
    """Summarize expression and detection for genes within cell groups.

    This group-level summary is preferable to individual-cell marker gating
    for sparse targeted spatial-transcriptomics data. Genes absent from the
    panel are omitted, while a complete lack of overlap raises an error.

    Parameters
    ----------
    adata
        AnnData-like object whose rows are cells and columns are genes.
    genes
        Requested gene symbols. Duplicates are removed while input order is
        retained.
    groupby
        Column in ``adata.obs`` defining the groups to summarize.
    layer
        Optional ``adata.layers`` key. If ``None``, use ``adata.X``.

    Returns
    -------
    pandas.DataFrame
        Long table with the group column, gene, mean_expression,
        pct_expressing, and n_cells.

    Raises
    ------
    ValueError
        If the grouping column, layer, or all requested genes are absent.
    """
    if groupby not in adata.obs.columns:
        raise ValueError(f"'{groupby}' is absent from adata.obs.")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' is absent from adata.layers.")

    requested = list(dict.fromkeys(map(str, genes)))
    present = [gene for gene in requested if gene in adata.var_names]
    if not present:
        raise ValueError("None of the requested genes are in adata.var_names.")

    subset = adata[:, present]
    matrix = subset.layers[layer] if layer is not None else subset.X
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()

    groups = adata.obs[groupby].astype(str)
    records: list[pd.DataFrame] = []
    for group in sorted(groups.unique()):
        mask = groups.eq(group).to_numpy()
        group_matrix = matrix[mask]
        if sparse.issparse(group_matrix):
            mean_expression = np.asarray(group_matrix.mean(axis=0)).ravel()
            pct_expressing = (
                np.asarray((group_matrix > 0).mean(axis=0)).ravel() * 100
            )
        else:
            dense = np.asarray(group_matrix)
            mean_expression = dense.mean(axis=0)
            pct_expressing = (dense > 0).mean(axis=0) * 100
        records.append(pd.DataFrame({
            groupby: group,
            "gene": present,
            "mean_expression": mean_expression,
            "pct_expressing": pct_expressing,
            "n_cells": int(mask.sum()),
        }))
    return pd.concat(records, ignore_index=True)


def add_human_gene_name(
    de_table: pd.DataFrame,
    ortholog_table: pd.DataFrame | None = None,
    mouse_gene_col: str = "names",
    output_col: str = "Human_gene_name",
    add_mapping_method: bool = True,
) -> pd.DataFrame:
    """Add human gene symbols to a mouse differential-expression table.

    Mappings are case-insensitive on the mouse symbol. One-to-many human
    orthologues are retained as a sorted ``"; "``-separated value. When no
    mapping is available, the uppercase mouse symbol is preserved as an
    explicit fallback rather than silently dropping the gene.

    Parameters
    ----------
    de_table
        Mouse differential-expression table.
    ortholog_table
        Optional table containing ``external_gene_name`` and
        ``mmusculus_homolog_associated_gene_name``.
    mouse_gene_col
        Column in ``de_table`` containing mouse gene symbols.
    output_col
        Destination column for mapped human gene symbols.
    add_mapping_method
        Add ``Human_gene_mapping_method`` provenance when ``True``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``de_table`` with human symbols and optional provenance.

    Raises
    ------
    ValueError
        If required gene or orthologue columns are missing.
    """
    if mouse_gene_col not in de_table.columns:
        raise ValueError(f"'{mouse_gene_col}' is absent from the DE table.")
    de = de_table.copy()
    mouse_genes = de[mouse_gene_col].astype("string").str.strip()
    de["_mouse_gene_key"] = mouse_genes.str.upper()
    fallback = mouse_genes.str.upper()

    if ortholog_table is None:
        de[output_col] = fallback
        if add_mapping_method:
            de["Human_gene_mapping_method"] = "uppercase_fallback"
        return de.drop(columns="_mouse_gene_key")

    required = {
        "external_gene_name",
        "mmusculus_homolog_associated_gene_name",
    }
    missing = sorted(required.difference(ortholog_table.columns))
    if missing:
        raise ValueError(f"Missing orthologue columns: {missing}")

    ortholog = ortholog_table[list(required)].copy()
    ortholog["_mouse_gene_key"] = (
        ortholog["mmusculus_homolog_associated_gene_name"]
        .astype("string").str.strip().str.upper()
    )
    ortholog["external_gene_name"] = (
        ortholog["external_gene_name"].astype("string").str.strip()
    )
    ortholog = ortholog.dropna(
        subset=["_mouse_gene_key", "external_gene_name"]
    )
    ortholog = ortholog.loc[ortholog["external_gene_name"].ne("")]
    mouse_to_human = (
        ortholog.groupby("_mouse_gene_key")["external_gene_name"]
        .agg(lambda values: "; ".join(sorted(pd.unique(values.astype(str)))))
    )
    mapped = de["_mouse_gene_key"].map(mouse_to_human).astype("string")
    has_mapping = mapped.notna() & mapped.ne("")
    de[output_col] = mapped.where(has_mapping, fallback)
    if add_mapping_method:
        de["Human_gene_mapping_method"] = "uppercase_fallback"
        de.loc[has_mapping, "Human_gene_mapping_method"] = "orthologue_table"
        de["Human_gene_mapping_method"] = de[
            "Human_gene_mapping_method"
        ].astype("category")
    return de.drop(columns="_mouse_gene_key")


def gene_detection_by_group(
    adata: Any,
    genes: Sequence[str],
    groupby: str,
    use_raw: bool = True,
) -> pd.DataFrame:
    """Calculate the percentage of cells detecting each gene by group.

    Parameters
    ----------
    adata
        AnnData-like object with cell metadata in ``obs``.
    genes
        Gene symbols to evaluate. Genes absent from the selected matrix are
        omitted unless none overlap.
    groupby
        Column in ``adata.obs`` defining cell groups.
    use_raw
        Use ``adata.raw`` when available. This is recommended for detection
        because positivity should be defined from unscaled transcript counts.

    Returns
    -------
    pandas.DataFrame
        Long table with group, pct_expressing, and gene columns.

    Raises
    ------
    ValueError
        If ``groupby`` is absent or no requested genes overlap the matrix.
    """
    if groupby not in adata.obs.columns:
        raise ValueError(f"'{groupby}' is absent from adata.obs.")
    source = adata.raw if use_raw and adata.raw is not None else adata
    requested = list(dict.fromkeys(map(str, genes)))
    present = [gene for gene in requested if gene in source.var_names]
    if not present:
        raise ValueError("None of the requested genes are in the selected matrix.")

    matrix = source[:, present].X
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    groups = adata.obs[groupby].astype(str)
    records: list[pd.DataFrame] = []
    for group in sorted(groups.unique()):
        mask = groups.eq(group).to_numpy()
        group_matrix = matrix[mask]
        if sparse.issparse(group_matrix):
            pct = np.asarray((group_matrix > 0).mean(axis=0)).ravel() * 100
        else:
            pct = (np.asarray(group_matrix) > 0).mean(axis=0) * 100
        records.append(pd.DataFrame({
            "group": group,
            "pct_expressing": pct,
            "gene": present,
        }))
    return pd.concat(records, ignore_index=True)


def merge_obs_to_main(
    main_adata: Any,
    subset_adata: Any,
    columns: str | Sequence[str] | Mapping[str, str],
    *,
    copy: bool = False,
    overwrite: bool = False,
    strict_subset: bool = True,
    source_name: str = "subset",
) -> tuple[Any, pd.DataFrame]:
    """Merge ``.obs`` columns from a subset into a main AnnData by cell ID.

    Parameters
    ----------
    main_adata
        Main AnnData object to update.
    subset_adata
        Subset AnnData whose ``obs_names`` derive from the main object.
    columns
        Source column name, sequence of names, or mapping of source names to
        destination names.
    copy
        Return a copy instead of modifying ``main_adata`` in place.
    overwrite
        Replace conflicting existing non-missing values when ``True``.
    strict_subset
        Raise if any subset cell is absent from the main object.
    source_name
        Human-readable source recorded in the merge report.

    Returns
    -------
    tuple
        Updated AnnData object and one-row-per-column merge report.

    Raises
    ------
    TypeError
        If ``columns`` has an unsupported type.
    ValueError
        For duplicate cell IDs, missing columns, unmatched cells, unexpected
        subset-only cells, or conflicting values when overwrite is disabled.
    """
    if isinstance(columns, str):
        column_map = {columns: columns}
    elif isinstance(columns, Mapping):
        column_map = dict(columns)
    elif isinstance(columns, Sequence):
        column_map = {column: column for column in columns}
    else:
        raise TypeError("columns must be a string, sequence, or mapping.")
    if not column_map:
        raise ValueError("No columns were provided.")

    for label, obj in (("main_adata", main_adata), (source_name, subset_adata)):
        if not obj.obs_names.is_unique:
            duplicated = obj.obs_names[obj.obs_names.duplicated()].unique()
            raise ValueError(
                f"{label}.obs_names contains duplicates: "
                f"{duplicated[:5].tolist()}"
            )
    missing_columns = [
        column for column in column_map if column not in subset_adata.obs.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Columns missing from {source_name}.obs: {missing_columns}"
        )

    matched = main_adata.obs_names.intersection(
        subset_adata.obs_names, sort=False
    )
    source_only = subset_adata.obs_names.difference(
        main_adata.obs_names, sort=False
    )
    if strict_subset and len(source_only):
        raise ValueError(
            f"{source_name} contains {len(source_only):,} cells absent from "
            f"main_adata: {source_only[:5].tolist()}"
        )
    if not len(matched):
        raise ValueError(
            f"No matching obs_names between main_adata and {source_name}."
        )

    merged = main_adata.copy() if copy else main_adata
    # Stage every change in a detached metadata table. The AnnData object is
    # updated only after all columns pass conflict checks, so a failed merge
    # cannot leave earlier columns partially written.
    staged_obs = merged.obs.copy(deep=True)
    report_rows: list[dict[str, Any]] = []
    for source_column, destination_column in column_map.items():
        incoming = subset_adata.obs.loc[matched, source_column].copy()
        valid_cells = incoming.index[incoming.notna()]
        if destination_column not in staged_obs.columns:
            dtype = "float64" if pd.api.types.is_numeric_dtype(incoming) else "object"
            fill = np.nan if dtype == "float64" else pd.NA
            staged_obs[destination_column] = pd.Series(
                fill, index=staged_obs.index, dtype=dtype
            )
        if isinstance(staged_obs[destination_column].dtype, pd.CategoricalDtype):
            staged_obs[destination_column] = staged_obs[
                destination_column
            ].astype(object)

        existing = staged_obs.loc[valid_cells, destination_column]
        incoming_valid = incoming.loc[valid_cells]
        if not overwrite:
            both = existing.notna() & incoming_valid.notna()
            if both.any():
                left = existing.loc[both]
                right = incoming_valid.loc[both]
                if (
                    pd.api.types.is_numeric_dtype(left)
                    and pd.api.types.is_numeric_dtype(right)
                ):
                    equal = pd.Series(
                        np.isclose(
                            pd.to_numeric(left), pd.to_numeric(right),
                            equal_nan=True,
                        ),
                        index=left.index,
                    )
                else:
                    equal = left.astype("string").eq(right.astype("string"))
                conflicts = equal.index[~equal]
                if len(conflicts):
                    examples = pd.DataFrame({
                        "existing": left.loc[conflicts[:5]],
                        "incoming": right.loc[conflicts[:5]],
                    })
                    raise ValueError(
                        f"Found {len(conflicts):,} conflicting values for "
                        f"'{destination_column}' from {source_name}.\n{examples}\n"
                        "Set overwrite=True if replacement is intended."
                    )
        staged_obs.loc[valid_cells, destination_column] = incoming_valid.to_numpy()
        report_rows.append({
            "source": source_name,
            "source_column": source_column,
            "destination_column": destination_column,
            "n_source_cells": subset_adata.n_obs,
            "n_matched_cells": len(matched),
            "n_source_only_cells": len(source_only),
            "n_nonmissing_values": len(valid_cells),
            "n_values_written": len(valid_cells),
            "overwrite": overwrite,
        })
    merged.obs = staged_obs
    return merged, pd.DataFrame(report_rows)


# -----------------------------------------------------------------------------
# Exploratory MHCII scoring
# -----------------------------------------------------------------------------


def assign_mhcii_single_signature(
    adata: Any,
    signature_genes: Sequence[str],
    core_genes: Sequence[str] = ("CD74", "HLA-DQB1"),
    score_name: str = "MHCIIhi_score",
    output_col: str = "MHCII_group",
    score_cutoff: float = 0,
    require_core_detection: bool = True,
    use_raw: bool = False,
    scale: bool = True,
    max_scale_value: float = 10,
    min_signature_genes: int = 3,
    random_state: int = 0,
    copy: bool = True,
) -> Any:
    """Score a positive MHCII-high signature and flag discordant cells.

    The function computes a deterministic mean signature score. With
    ``scale=True``, each measured signature gene is standardized across the
    supplied cells before averaging. Direct MHC-II detection is evaluated
    from ``raw`` counts when available, independently of the scoring matrix.
    This is an exploratory annotation aid, not the final mouse-derived
    MHCII-high/MHCII-low AM definition.

    Parameters
    ----------
    adata
        AnnData-like object containing the candidate cells to score.
    signature_genes
        Positive-program human gene symbols, matched case-insensitively.
    core_genes
        Core MHC-II genes used as direct detection evidence.
    score_name
        ``obs`` column receiving the continuous signature score.
    output_col
        ``obs`` column receiving MHCIIlo, MHCIIhi, or Ambiguous.
    score_cutoff
        Strict lower boundary for a positive score.
    require_core_detection
        Require agreement between the score and direct core-gene detection.
    use_raw
        Score from ``adata.raw`` when available. The default uses processed
        ``adata.X``; raw Xenium counts should not be scored without an
        intentional normalization decision.
    scale
        Standardize each signature gene before averaging.
    max_scale_value
        Absolute clipping bound applied after standardization.
    min_signature_genes
        Minimum panel-overlapping signature genes required.
    random_state
        Retained for API/provenance compatibility. Scoring is deterministic
        and performs no random control-gene sampling.
    copy
        Return a copy when ``True``; otherwise modify ``adata`` in place.

    Returns
    -------
    AnnData
        Object with score, core-detection columns, categorical group, and an
        audit record in ``uns['MHCII_single_signature']``.

    Raises
    ------
    ValueError
        If too few signature genes or no required core genes are measured.
    """
    result = adata.copy() if copy else adata
    score_source = result.raw if use_raw and result.raw is not None else result
    score_lookup = {
        str(gene).upper(): str(gene) for gene in score_source.var_names
    }
    requested_signature = list(dict.fromkeys(
        str(gene).strip().upper() for gene in signature_genes
    ))
    available_signature = [
        score_lookup[gene] for gene in requested_signature if gene in score_lookup
    ]
    missing_signature = [
        gene for gene in requested_signature if gene not in score_lookup
    ]
    if len(available_signature) < min_signature_genes:
        raise ValueError(
            "Too few signature genes are present in the Xenium panel: "
            f"{len(available_signature)} available."
        )

    score_matrix = score_source[:, available_signature].X
    if sparse.issparse(score_matrix):
        score_matrix = score_matrix.astype(float).tocsr()
        if scale:
            means = np.asarray(score_matrix.mean(axis=0)).ravel()
            mean_squares = np.asarray(
                score_matrix.power(2).mean(axis=0)
            ).ravel()
            variances = np.maximum(mean_squares - means ** 2, 0)
            standard_deviations = np.sqrt(variances)
            standard_deviations[standard_deviations == 0] = 1.0

            # A centered sparse matrix is mathematically dense because every
            # stored zero receives a non-zero baseline. Preserve sparsity by
            # representing that baseline separately, and store only each
            # observed entry's deviation from it.
            zero_baseline = np.clip(
                -means / standard_deviations,
                -max_scale_value,
                max_scale_value,
            )
            scaled_delta = score_matrix.copy()
            columns = scaled_delta.indices
            observed_scaled = (
                scaled_delta.data - means[columns]
            ) / standard_deviations[columns]
            scaled_delta.data = (
                np.clip(
                    observed_scaled, -max_scale_value, max_scale_value
                )
                - zero_baseline[columns]
            )
            scores = (
                zero_baseline.mean()
                + np.asarray(scaled_delta.sum(axis=1)).ravel()
                / len(available_signature)
            )
        else:
            scores = np.asarray(score_matrix.mean(axis=1)).ravel()
    else:
        score_matrix = np.asarray(score_matrix, dtype=float)
        if scale:
            means = score_matrix.mean(axis=0)
            standard_deviations = score_matrix.std(axis=0, ddof=0)
            standard_deviations[standard_deviations == 0] = 1.0
            score_matrix = (score_matrix - means) / standard_deviations
            score_matrix = np.clip(
                score_matrix, -max_scale_value, max_scale_value
            )
        scores = score_matrix.mean(axis=1)

    detection_source = result.raw if result.raw is not None else result
    detection_lookup = {
        str(gene).upper(): str(gene) for gene in detection_source.var_names
    }
    requested_core = list(dict.fromkeys(
        str(gene).strip().upper() for gene in core_genes
    ))
    available_core = [
        detection_lookup[gene] for gene in requested_core
        if gene in detection_lookup
    ]
    missing_core = [gene for gene in requested_core if gene not in detection_lookup]
    if require_core_detection and not available_core:
        raise ValueError(
            "None of the core MHC-II genes are present. Set "
            "require_core_detection=False only if intentional."
        )
    if available_core:
        core_matrix = detection_source[:, available_core].X
        if sparse.issparse(core_matrix):
            core_n_detected = np.asarray(
                (core_matrix > 0).sum(axis=1)
            ).ravel()
        else:
            core_n_detected = (np.asarray(core_matrix) > 0).sum(axis=1)
    else:
        core_n_detected = np.zeros(result.n_obs, dtype=int)
    core_detected = core_n_detected > 0

    result.obs[score_name] = scores
    result.obs["MHCII_core_detected"] = core_detected
    result.obs["MHCII_core_n_detected"] = core_n_detected
    score_positive = scores > score_cutoff
    if require_core_detection:
        groups = np.select(
            [score_positive & core_detected, ~score_positive & ~core_detected],
            ["MHCIIhi", "MHCIIlo"],
            default="Ambiguous",
        )
        categories = ["MHCIIlo", "MHCIIhi", "Ambiguous"]
    else:
        groups = np.where(score_positive, "MHCIIhi", "MHCIIlo")
        categories = ["MHCIIlo", "MHCIIhi"]
    result.obs[output_col] = pd.Categorical(
        groups, categories=categories, ordered=True
    )
    result.uns["MHCII_single_signature"] = {
        "available_signature": available_signature,
        "missing_signature": missing_signature,
        "available_core": available_core,
        "missing_core": missing_core,
        "score_cutoff": score_cutoff,
        "require_core_detection": require_core_detection,
        "score_source": "raw" if score_source is result.raw else "X",
        "scale": scale,
        "max_scale_value": max_scale_value,
        "random_state_recorded_only": random_state,
        "score_method": "mean gene z-score" if scale else "mean expression",
    }
    return result


# -----------------------------------------------------------------------------
# Cohort and abundance plots
# -----------------------------------------------------------------------------


def plot_metadata_summary(
    unique_meta: pd.DataFrame,
    tissue_order: Sequence[str] = ("A", "B", "V", "None"),
    save: str | Path | None = None,
    dpi: int = 300,
) -> dict[str, Any]:
    """Create a macaron-style overview of the human lung Xenium cohort.

    Core-level variables are reduced to one row per core; age, sex, PMI, and
    TMA are checked for within-donor consistency before donor summaries are
    created. Inconsistencies are returned and warned about rather than printed
    with notebook-only display functions.

    Parameters
    ----------
    unique_meta
        Metadata containing donor_id, core_id, tma_id, age, pmi, sex, and
        tissue_annotation. It may contain repeated cell-level rows.
    tissue_order
        Preferred display order for tissue annotations.
    save
        Optional output path. The format is inferred from its extension.
    dpi
        Raster resolution when saving the figure.

    Returns
    -------
    dict
        Figure plus core, donor, inconsistency, cohort, tissue, TMA, and
        donor-by-tissue summary tables.

    Raises
    ------
    ValueError
        If required metadata columns are absent or no cores remain.
    """
    required = {
        "donor_id", "core_id", "tma_id", "age", "pmi", "sex",
        "tissue_annotation",
    }
    missing = sorted(required.difference(unique_meta.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    core_meta = unique_meta[list(required)].copy()
    for column in ("donor_id", "core_id", "tma_id"):
        core_meta[column] = core_meta[column].astype("string").str.strip()
    core_meta["sex"] = (
        core_meta["sex"].astype("string").str.strip().str.upper()
    )
    core_meta["tissue_annotation"] = (
        core_meta["tissue_annotation"].astype("string").str.strip()
        .fillna("None")
        .replace({"nan": "None", "NaN": "None", "<NA>": "None",
                  "NA": "None", "": "None"})
    )
    for column in ("age", "pmi"):
        core_meta[column] = pd.to_numeric(core_meta[column], errors="coerce")
    core_meta = core_meta.drop_duplicates(subset="core_id").reset_index(drop=True)
    if core_meta.empty:
        raise ValueError("No unique cores remain after metadata preparation.")

    donor_variables = ["age", "pmi", "sex", "tma_id"]
    inconsistent_mask = (
        core_meta.groupby("donor_id")[donor_variables]
        .nunique(dropna=False).gt(1)
    )
    donor_inconsistency = inconsistent_mask.loc[
        inconsistent_mask.any(axis=1)
    ]
    if not donor_inconsistency.empty:
        warnings.warn(
            "Inconsistent donor-level metadata detected; inspect the returned "
            "donor_inconsistency table.",
            RuntimeWarning,
            stacklevel=2,
        )
    donor_meta = (
        core_meta.sort_values(["donor_id", "core_id"])
        .groupby("donor_id", as_index=False)
        .agg(
            age=("age", "first"), pmi=("pmi", "first"),
            sex=("sex", "first"), tma_id=("tma_id", "first"),
            n_cores=("core_id", "nunique"),
        )
    )
    tissues_present = core_meta["tissue_annotation"].dropna().unique().tolist()
    final_tissue_order = [x for x in tissue_order if x in tissues_present]
    final_tissue_order.extend(x for x in tissues_present if x not in final_tissue_order)
    tma_order = sorted(core_meta["tma_id"].dropna().astype(str).unique())
    sex_order = [x for x in ("F", "M") if x in donor_meta["sex"].values]
    sex_order.extend(x for x in donor_meta["sex"].dropna().unique() if x not in sex_order)

    tissue_colors = dict(TISSUE_PALETTE)
    fallback = sns.color_palette("pastel", n_colors=max(len(final_tissue_order), 1))
    for index, tissue in enumerate(final_tissue_order):
        tissue_colors.setdefault(tissue, mpl.colors.to_hex(fallback[index]))
    tma_colors = dict(TMA_PALETTE)
    for index, tma in enumerate(tma_order):
        tma_colors.setdefault(tma, mpl.colors.to_hex(fallback[index % len(fallback)]))

    tissue_counts = (
        core_meta["tissue_annotation"].value_counts()
        .reindex(final_tissue_order, fill_value=0)
        .rename_axis("tissue_annotation").reset_index(name="n_cores")
    )
    tissue_by_tma_counts = pd.crosstab(
        core_meta["tma_id"], core_meta["tissue_annotation"]
    ).reindex(index=tma_order, columns=final_tissue_order, fill_value=0)
    tissue_by_tma_pct = tissue_by_tma_counts.div(
        tissue_by_tma_counts.sum(axis=1).replace(0, np.nan), axis=0
    ).mul(100).fillna(0)
    donor_tissue_counts = pd.crosstab(
        core_meta["donor_id"], core_meta["tissue_annotation"]
    ).reindex(columns=final_tissue_order, fill_value=0)
    n_female = int(donor_meta["sex"].eq("F").sum())
    n_male = int(donor_meta["sex"].eq("M").sum())
    cohort_summary = pd.DataFrame({
        "Metric": [
            "Donors", "Cores", "Female donors", "Male donors",
            "Median age", "Minimum age", "Maximum age", "Median PMI",
            "Minimum PMI", "Maximum PMI",
        ],
        "Value": [
            donor_meta["donor_id"].nunique(), core_meta["core_id"].nunique(),
            n_female, n_male, donor_meta["age"].median(),
            donor_meta["age"].min(), donor_meta["age"].max(),
            donor_meta["pmi"].median(), donor_meta["pmi"].min(),
            donor_meta["pmi"].max(),
        ],
    })

    configure_plot_style()
    figure = plt.figure(figsize=(17, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 4, height_ratios=[0.68, 2.1, 3.3])
    card_specs = [
        ("DONORS", str(len(donor_meta)), f"{n_female} female · {n_male} male", "#F1B6B2"),
        ("CORES", str(len(core_meta)), f"{len(tma_order)} tissue microarrays", "#BFD8C2"),
        ("AGE", f"{donor_meta['age'].median():g} years", "median", "#CDB9DD"),
        ("PMI", f"{donor_meta['pmi'].median():g} hours", "median", "#F3C8A8"),
    ]
    for column, (title, value, subtitle, color) in enumerate(card_specs):
        axis = figure.add_subplot(grid[0, column])
        axis.set_axis_off()
        axis.add_patch(FancyBboxPatch(
            (0.02, 0.08), 0.96, 0.84,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            edgecolor=color, facecolor=color, alpha=0.45,
            transform=axis.transAxes,
        ))
        axis.text(0.07, 0.71, title, transform=axis.transAxes,
                  fontsize=10, weight="bold", color="#555555")
        axis.text(0.07, 0.42, value, transform=axis.transAxes,
                  fontsize=21, weight="bold")
        axis.text(0.07, 0.18, subtitle, transform=axis.transAxes,
                  fontsize=9, color="#666666")

    ax1 = figure.add_subplot(grid[1, 0])
    bars = ax1.bar(
        tissue_counts["tissue_annotation"], tissue_counts["n_cores"],
        color=[tissue_colors[x] for x in tissue_counts["tissue_annotation"]],
        edgecolor=DARK_TEXT,
    )
    ax1.bar_label(bars, padding=3)
    ax1.set(title="Core distribution", xlabel="Tissue annotation",
            ylabel="Number of cores")

    ax2 = figure.add_subplot(grid[1, 1])
    bottom = np.zeros(len(tma_order))
    for tissue in final_tissue_order:
        values = tissue_by_tma_pct[tissue].to_numpy()
        ax2.bar(tma_order, values, bottom=bottom, color=tissue_colors[tissue],
                edgecolor="white", label=tissue)
        bottom += values
    ax2.set(title="Tissue composition by TMA", ylabel="Core composition (%)",
            ylim=(0, 100))
    ax2.legend(title="Tissue", frameon=False, fontsize=8)

    ax3 = figure.add_subplot(grid[1, 2])
    age_groups = [
        donor_meta.loc[donor_meta["sex"].eq(sex), "age"].dropna().to_numpy()
        for sex in sex_order
    ]
    boxplot = ax3.boxplot(
        age_groups, tick_labels=sex_order, patch_artist=True, widths=0.52,
        medianprops={"color": DARK_TEXT},
    )
    for patch, sex in zip(boxplot["boxes"], sex_order):
        patch.set_facecolor(SEX_PALETTE.get(sex, "#D9D6D2"))
        patch.set_edgecolor(DARK_TEXT)
    for position, (sex, values) in enumerate(zip(sex_order, age_groups), start=1):
        offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0]
        ax3.scatter(
            position + np.asarray(offsets), values,
            color=SEX_PALETTE.get(sex, "#D9D6D2"), edgecolor=DARK_TEXT,
            linewidth=0.6, s=34, zorder=3,
        )
    ax3.set(title="Donor age distribution", xlabel="Sex", ylabel="Age (years)")

    ax4 = figure.add_subplot(grid[1, 3])
    donor_plot = donor_meta.sort_values(["n_cores", "donor_id"], ascending=[False, True])
    ax4.bar(donor_plot["donor_id"], donor_plot["n_cores"],
            color=[tma_colors.get(str(x), "#D9D6D2") for x in donor_plot["tma_id"]])
    ax4.tick_params(axis="x", rotation=90)
    ax4.set(title="Sampling depth per donor", xlabel="Donor",
            ylabel="Number of cores")

    ax5 = figure.add_subplot(grid[2, :2])
    slots = core_meta["core_id"].str.extract(r"\.c(\d+)$", expand=False)
    fallback_slots = core_meta.groupby("donor_id").cumcount().add(1).astype(str)
    core_meta["_core_slot"] = slots.fillna(fallback_slots)
    slot_order = sorted(core_meta["_core_slot"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    donor_order = donor_meta.sort_values(["tma_id", "age", "donor_id"])["donor_id"]
    donor_core_grid = core_meta.pivot_table(
        index="donor_id", columns="_core_slot", values="tissue_annotation",
        aggfunc="first",
    ).reindex(index=donor_order, columns=slot_order)
    tissue_code = {tissue: index for index, tissue in enumerate(final_tissue_order)}
    numeric_grid = donor_core_grid.map(tissue_code.get).to_numpy(dtype=float)
    color_map = ListedColormap(
        [tissue_colors[x] for x in final_tissue_order]
    ).with_extremes(bad="white")
    ax5.imshow(
        np.ma.masked_invalid(numeric_grid), aspect="auto", cmap=color_map,
        norm=BoundaryNorm(np.arange(len(final_tissue_order) + 1) - 0.5,
                          color_map.N),
    )
    ax5.set_xticks(np.arange(len(slot_order)), labels=[f"Core {x}" for x in slot_order])
    ax5.set_yticks(np.arange(len(donor_order)), labels=donor_order)
    ax5.set_title("Donor–core tissue map", loc="left")
    ax5.legend(handles=[Patch(facecolor=tissue_colors[x], label=x)
                        for x in final_tissue_order], frameon=False,
               title="Tissue", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax6 = figure.add_subplot(grid[2, 2:])
    sns.scatterplot(data=donor_meta, x="age", y="pmi", hue="sex",
                    palette=SEX_PALETTE, style="tma_id", s=100, ax=ax6)
    for row in donor_meta.itertuples():
        ax6.annotate(row.donor_id, (row.age, row.pmi), xytext=(4, 4),
                     textcoords="offset points", fontsize=7)
    ax6.set(title="Age and post-mortem interval", xlabel="Age (years)",
            ylabel="PMI (hours)")
    ax6.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    for axis in (ax1, ax2, ax3, ax4, ax5, ax6):
        axis.grid(False)
        axis.set_facecolor("white")
    figure.suptitle("Human lung Xenium cohort overview", x=0.01, ha="left",
                    fontsize=18, weight="bold")
    if save is not None:
        figure.savefig(save, dpi=dpi, bbox_inches="tight", facecolor="white")
    return {
        "fig": figure,
        "core_meta": core_meta.drop(columns="_core_slot", errors="ignore"),
        "donor_meta": donor_meta,
        "donor_inconsistency": donor_inconsistency,
        "cohort_summary": cohort_summary,
        "tissue_counts": tissue_counts,
        "tissue_by_tma_counts": tissue_by_tma_counts,
        "tissue_by_tma_pct": tissue_by_tma_pct,
        "donor_tissue_counts": donor_tissue_counts,
    }


def plot_macrophage_pct_by_tissue(
    macrophage_pct_long: pd.DataFrame,
    tissue_order: Sequence[str] = ("A", "B", "V", "None"),
    macrophage_order: Sequence[str] = (
        "AM", "AM-like", "LYVE1+ IM", "MMP2+ IM",
    ),
    comparisons: Sequence[tuple[str, str]] = (
        ("A", "B"), ("A", "V"), ("B", "V"),
    ),
    donor_col: str = "donor_id",
    tissue_col: str = "tissue_annotation",
    celltype_col: str = "CellType",
    value_col: str = "percent_of_all_cells",
    ylabel: str = "Macrophages (% of all cells)",
    palette: Mapping[str, str] | None = None,
    show_ns: bool = True,
    save: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Figure, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Plot donor-level macrophage abundance across tissue annotations.

    Multiple cores from the same donor and tissue are averaged before any
    plotting or inference. Bars and SEM therefore summarize donors, not cells
    or cores. Paired two-sided Wilcoxon tests use donors represented in both
    tissues, followed by Benjamini-Hochberg correction across every requested
    subtype/comparison test.

    Parameters
    ----------
    macrophage_pct_long
        Long table containing donor, tissue, macrophage subtype, and abundance.
    tissue_order
        Tissue categories and display order.
    macrophage_order
        Macrophage subtype categories and facet order.
    comparisons
        Tissue pairs tested within each macrophage subtype.
    donor_col, tissue_col, celltype_col, value_col
        Column names in ``macrophage_pct_long``.
    ylabel
        Shared y-axis label.
    palette
        Subtype-to-color mapping; defaults to the reusable macaron palette.
    show_ns
        Display brackets for adjusted non-significant comparisons.
    save
        Optional output path.
    dpi
        Raster resolution when saving.

    Returns
    -------
    tuple
        Figure, axes array, donor-level summary, and statistics table.

    Raises
    ------
    ValueError
        If required columns are missing or filtering leaves no observations.
    """
    required = {donor_col, tissue_col, celltype_col, value_col}
    missing = sorted(required.difference(macrophage_pct_long.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = macrophage_pct_long.copy()
    short_names = {
        "Alveolar Macrophage": "AM", "SPP1+ Macrophage": "AM-like",
        "MARCO-low Macrophage": "AM-like", "AM-like Macrophage": "AM-like",
        "LYVE1+ Interstitial Macrophage": "LYVE1+ IM",
        "MMP2+ Interstitial Macrophage": "MMP2+ IM",
    }
    data[celltype_col] = data[celltype_col].astype(object).replace(short_names)
    data[tissue_col] = data[tissue_col].fillna("None").astype(str)
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.loc[
        data[celltype_col].isin(macrophage_order)
        & data[tissue_col].isin(tissue_order)
        & data[value_col].notna()
    ].copy()
    if data.empty:
        raise ValueError(
            "No observations remain after filtering macrophage and tissue categories."
        )
    donor_summary = (
        data.groupby([donor_col, tissue_col, celltype_col], observed=True,
                     as_index=False)[value_col].mean()
    )
    donor_summary[tissue_col] = pd.Categorical(
        donor_summary[tissue_col], categories=list(tissue_order), ordered=True
    )
    donor_summary[celltype_col] = pd.Categorical(
        donor_summary[celltype_col], categories=list(macrophage_order), ordered=True
    )

    rows: list[dict[str, Any]] = []
    for celltype in macrophage_order:
        cell_data = donor_summary.loc[
            donor_summary[celltype_col].astype(str).eq(celltype)
        ]
        wide = cell_data.pivot(index=donor_col, columns=tissue_col,
                               values=value_col)
        for tissue1, tissue2 in comparisons:
            paired = (
                wide[[tissue1, tissue2]].dropna()
                if tissue1 in wide and tissue2 in wide
                else pd.DataFrame()
            )
            n_paired = len(paired)
            p_value = np.nan
            if n_paired >= 3:
                differences = paired[tissue1].to_numpy() - paired[tissue2].to_numpy()
                if np.allclose(differences, 0):
                    p_value = 1.0
                else:
                    try:
                        p_value = wilcoxon(
                            paired[tissue1], paired[tissue2],
                            alternative="two-sided", zero_method="wilcox",
                        ).pvalue
                    except ValueError:
                        p_value = np.nan
            rows.append({
                celltype_col: celltype, "tissue1": tissue1,
                "tissue2": tissue2, "n_paired_donors": n_paired,
                "p_value": p_value,
            })
    statistics = pd.DataFrame(rows)
    statistics["p_adj"] = np.nan
    valid = statistics["p_value"].notna()
    if valid.any():
        statistics.loc[valid, "p_adj"] = false_discovery_control(
            statistics.loc[valid, "p_value"].to_numpy(), method="bh"
        )

    def significance_label(p_value: float) -> str:
        if pd.isna(p_value):
            return "n<3"
        for threshold, label in (
            (0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*"),
        ):
            if p_value < threshold:
                return label
        return "ns"

    statistics["significance"] = statistics["p_adj"].map(significance_label)
    colors = dict(MACROPHAGE_SUBTYPE_PALETTE)
    if palette is not None:
        colors.update(palette)
    present_tissues = [
        tissue for tissue in tissue_order
        if tissue in donor_summary[tissue_col].astype(str).unique()
    ]
    global_max = donor_summary[value_col].max()
    global_max = float(global_max) if np.isfinite(global_max) and global_max > 0 else 1.0
    bracket_step = max(global_max * 0.12, 0.05)
    bracket_start = global_max + bracket_step * 0.45
    y_upper = bracket_start + bracket_step * (len(comparisons) + 1.1)
    figure, axes = plt.subplots(
        1, len(macrophage_order), figsize=(3.3 * len(macrophage_order), 4.8),
        sharey=True, squeeze=False,
    )
    axes_array = axes.ravel()
    title_map = {
        "AM": "Alveolar macrophage", "AM-like": "AM-like macrophage",
        "LYVE1+ IM": "LYVE1+ interstitial macrophage",
        "MMP2+ IM": "MMP2+ interstitial macrophage",
    }
    for axis, celltype in zip(axes_array, macrophage_order):
        plot_data = donor_summary.loc[
            donor_summary[celltype_col].astype(str).eq(celltype)
        ]
        color = colors.get(celltype, "#BDBDBD")
        sns.barplot(
            data=plot_data, x=tissue_col, y=value_col, order=present_tissues,
            color=color, width=0.68, errorbar="se", capsize=0.12,
            edgecolor="black", linewidth=0.8, ax=axis,
        )
        sns.stripplot(
            data=plot_data, x=tissue_col, y=value_col, order=present_tissues,
            color=color, edgecolor="black", linewidth=0.5, size=4.2,
            jitter=0.13, alpha=0.9, ax=axis,
        )
        positions = {tissue: index for index, tissue in enumerate(present_tissues)}
        cell_statistics = statistics.loc[statistics[celltype_col].eq(celltype)]
        bracket_number = 0
        for row in cell_statistics.itertuples(index=False):
            label = row.significance
            if row.tissue1 not in positions or row.tissue2 not in positions:
                continue
            if not show_ns and label == "ns":
                continue
            x1, x2 = positions[row.tissue1], positions[row.tissue2]
            y = bracket_start + bracket_number * bracket_step
            height = bracket_step * 0.13
            axis.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
                      color="black", linewidth=0.8, clip_on=False)
            axis.text((x1 + x2) / 2, y + height * 1.7, label,
                      ha="center", va="bottom", fontsize=8.5)
            bracket_number += 1
        axis.set_title(title_map.get(celltype, celltype), fontsize=10,
                       weight="bold")
        axis.set(xlabel="", ylabel="", ylim=(0, y_upper))
        axis.grid(False)
        sns.despine(ax=axis)
    axes_array[0].set_ylabel(ylabel)
    figure.text(0.5, 0.02, "Tissue annotation", ha="center")
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.14, top=0.84,
                           wspace=0.20)
    if save is not None:
        figure.savefig(save, dpi=dpi, bbox_inches="tight", transparent=True)
    return figure, axes_array, donor_summary, statistics


def plot_am_at2_pct_by_donor(
    adata,
    *,
    celltype_col="CellType_refined",
    donor_col="donor_id",
    core_col="core_id",
    tissue_col="tissue_annotation",
    am_labels=("AM", "Alveolar Macrophage"),
    at2_labels=("AT2",),
    tissue_order=("A", "B", "V", "None"),
    tissue_colors=None,
    point_size=70,
    point_alpha=0.95,
    annotate_cores=True,
    core_label_size=7,
    shared_y=True,
    figsize=(16, 5),
    save=None,
    dpi=300,
):
    """
    Calculate and plot AM and AT2 percentages for every core.

    Percentage denominator:
        all cells within the same core_id.

    Each point represents one core.
    Points are grouped along the x-axis by donor and colored by
    tissue_annotation.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-level object containing annotations used to count AM and AT2.
    celltype_col, donor_col, core_col, tissue_col : str
        Observation columns defining cell type, donor, core, and tissue.
    am_labels, at2_labels : sequence of str
        Labels counted as alveolar macrophages and AT2 cells.
    tissue_order : sequence of str
        Tissue display order.
    tissue_colors : mapping or None
        Optional tissue-to-color mapping.
    point_size, point_alpha : float
        Core-point size and opacity.
    annotate_cores : bool
        Whether to label core points.
    core_label_size : float
        Core-label font size.
    shared_y : bool
        Whether AM and AT2 panels share a y-axis.
    figsize : tuple
        Figure size in inches.
    save : path-like or None
        Optional output path.
    dpi : int
        Resolution used when saving.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray
    summary_wide : pandas.DataFrame
        One row per core, with AM and AT2 counts and percentages.
    summary_long : pandas.DataFrame
        Long-format table with one row per core and cell type.
    """

    # =========================================================
    # 1. Validate columns
    # =========================================================
    required_columns = {
        celltype_col,
        donor_col,
        core_col,
        tissue_col,
    }

    missing_columns = required_columns.difference(
        adata.obs.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing adata.obs columns: {sorted(missing_columns)}"
        )

    # =========================================================
    # 2. Prepare cell-level metadata
    # =========================================================
    obs = adata.obs[
        [
            celltype_col,
            donor_col,
            core_col,
            tissue_col,
        ]
    ].copy()

    obs[celltype_col] = (
        obs[celltype_col]
        .astype("string")
        .str.strip()
    )

    obs[donor_col] = (
        obs[donor_col]
        .astype("string")
        .str.strip()
    )

    obs[core_col] = (
        obs[core_col]
        .astype("string")
        .str.strip()
    )

    obs[tissue_col] = (
        obs[tissue_col]
        .astype("string")
        .str.strip()
        .fillna("None")
        .replace(
            {
                "nan": "None",
                "NaN": "None",
                "<NA>": "None",
                "": "None",
            }
        )
    )

    am_labels = {
        str(label)
        for label in am_labels
    }

    at2_labels = {
        str(label)
        for label in at2_labels
    }

    obs["_is_AM"] = (
        obs[celltype_col]
        .isin(am_labels)
        .astype(int)
    )

    obs["_is_AT2"] = (
        obs[celltype_col]
        .isin(at2_labels)
        .astype(int)
    )

    obs["_one"] = 1

    # =========================================================
    # 3. Check that core metadata are consistent
    # =========================================================
    core_consistency = (
        obs.groupby(
            core_col,
            observed=True,
            dropna=False,
        )[
            [
                donor_col,
                tissue_col,
            ]
        ]
        .nunique(dropna=False)
    )

    inconsistent_cores = core_consistency.loc[
        core_consistency.gt(1).any(axis=1)
    ]

    if len(inconsistent_cores) > 0:
        raise ValueError(
            "Some core IDs have inconsistent donor or tissue "
            f"annotations:\n{inconsistent_cores}"
        )

    # =========================================================
    # 4. Summarize each core
    # =========================================================
    grouping_columns = [
        donor_col,
        core_col,
        tissue_col,
    ]

    summary_wide = (
        obs.groupby(
            grouping_columns,
            observed=True,
            dropna=False,
        )
        .agg(
            total_cells=("_one", "sum"),
            n_AM=("_is_AM", "sum"),
            n_AT2=("_is_AT2", "sum"),
        )
        .reset_index()
    )

    summary_wide["pct_AM"] = (
        100
        * summary_wide["n_AM"]
        / summary_wide["total_cells"]
    )

    summary_wide["pct_AT2"] = (
        100
        * summary_wide["n_AT2"]
        / summary_wide["total_cells"]
    )

    # =========================================================
    # 5. Long-format summary
    # =========================================================
    am_long = summary_wide[
        grouping_columns
        + [
            "total_cells",
            "n_AM",
            "pct_AM",
        ]
    ].rename(
        columns={
            "n_AM": "n_cells",
            "pct_AM": "pct",
        }
    )

    am_long["CellType"] = "AM"

    at2_long = summary_wide[
        grouping_columns
        + [
            "total_cells",
            "n_AT2",
            "pct_AT2",
        ]
    ].rename(
        columns={
            "n_AT2": "n_cells",
            "pct_AT2": "pct",
        }
    )

    at2_long["CellType"] = "AT2"

    summary_long = pd.concat(
        [
            am_long,
            at2_long,
        ],
        ignore_index=True,
    )

    summary_long = summary_long[
        grouping_columns
        + [
            "CellType",
            "n_cells",
            "total_cells",
            "pct",
        ]
    ]

    # =========================================================
    # 6. Natural donor/core ordering
    # =========================================================
    def natural_sort_key(value):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(
                r"(\d+)",
                str(value),
            )
        ]

    donor_order = sorted(
        summary_wide[donor_col]
        .dropna()
        .unique(),
        key=natural_sort_key,
    )

    donor_position = {
        donor: index
        for index, donor in enumerate(donor_order)
    }

    # Extract c1, c2, c3, etc.
    summary_wide["_core_slot"] = (
        summary_wide[core_col]
        .str.extract(
            r"\.c(\d+)$",
            expand=False,
        )
    )

    # Fallback for non-standard core names
    fallback_slot = (
        summary_wide
        .sort_values(
            [donor_col, core_col]
        )
        .groupby(donor_col)
        .cumcount()
        .add(1)
        .astype(str)
    )

    summary_wide["_core_slot"] = (
        summary_wide["_core_slot"]
        .fillna(fallback_slot)
        .astype(str)
    )

    summary_wide["_core_label"] = (
        "c" + summary_wide["_core_slot"]
    )

    slot_order = sorted(
        summary_wide["_core_slot"].unique(),
        key=natural_sort_key,
    )

    if len(slot_order) == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(
            -0.28,
            0.28,
            len(slot_order),
        )

    slot_offset = {
        slot: offset
        for slot, offset in zip(
            slot_order,
            offsets,
        )
    }

    summary_wide["_x_position"] = [
        donor_position[donor]
        + slot_offset[slot]
        for donor, slot in zip(
            summary_wide[donor_col],
            summary_wide["_core_slot"],
        )
    ]

    # Add plotting positions to long table
    summary_long = summary_long.merge(
        summary_wide[
            [
                core_col,
                "_core_slot",
                "_core_label",
                "_x_position",
            ]
        ],
        on=core_col,
        how="left",
        validate="many_to_one",
    )

    # =========================================================
    # 7. Tissue colors
    # =========================================================
    default_tissue_colors = {
        "A": "#D95F62",
        "B": "#8963B5",
        "V": "#2E86AB",
        "None": "#8D8D8D",
    }

    if tissue_colors is None:
        tissue_colors = default_tissue_colors
    else:
        tissue_colors = {
            **default_tissue_colors,
            **tissue_colors,
        }

    tissues_present = (
        summary_wide[tissue_col]
        .dropna()
        .unique()
        .tolist()
    )

    final_tissue_order = [
        tissue
        for tissue in tissue_order
        if tissue in tissues_present
    ]

    final_tissue_order += [
        tissue
        for tissue in tissues_present
        if tissue not in final_tissue_order
    ]

    fallback_colors = plt.get_cmap("Set2")(
        np.linspace(
            0,
            1,
            max(len(final_tissue_order), 1),
        )
    )

    for index, tissue in enumerate(
        final_tissue_order
    ):
        if tissue not in tissue_colors:
            tissue_colors[tissue] = (
                fallback_colors[index]
            )

    # =========================================================
    # 8. Plot
    # =========================================================
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=shared_y,
        constrained_layout=True,
        facecolor="white",
    )

    panel_information = [
        ("AM", "pct_AM"),
        ("AT2", "pct_AT2"),
    ]

    maximum_pct = max(
        summary_wide[
            [
                "pct_AM",
                "pct_AT2",
            ]
        ].to_numpy().max(),
        1,
    )

    y_upper = min(
        100,
        maximum_pct * 1.20 + 1,
    )

    for ax, (
        cell_type,
        percentage_column,
    ) in zip(
        axes,
        panel_information,
    ):
        for tissue in final_tissue_order:
            tissue_data = summary_wide.loc[
                summary_wide[tissue_col].eq(tissue)
            ]

            ax.scatter(
                tissue_data["_x_position"],
                tissue_data[percentage_column],
                s=point_size,
                color=tissue_colors[tissue],
                alpha=point_alpha,
                edgecolor="#222222",
                linewidth=0.65,
                zorder=3,
                label=tissue,
            )

        if annotate_cores:
            for _, row in summary_wide.iterrows():
                ax.annotate(
                    row["_core_label"],
                    (
                        row["_x_position"],
                        row[percentage_column],
                    ),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=core_label_size,
                    color="#333333",
                    clip_on=False,
                )

        ax.set_xticks(
            np.arange(len(donor_order)),
            labels=donor_order,
            rotation=90,
        )

        ax.set_xlim(
            -0.65,
            len(donor_order) - 0.35,
        )

        ax.set_ylim(0, y_upper)

        ax.yaxis.set_major_formatter(
            PercentFormatter(
                xmax=100,
                decimals=0,
            )
        )

        ax.set_title(
            f"{cell_type} abundance",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_xlabel("Donor")
        ax.grid(False)
        ax.set_facecolor("white")

        ax.tick_params(
            axis="both",
            labelsize=8,
            direction="out",
            length=3,
            width=0.8,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")

    axes[0].set_ylabel(
        "Cell abundance within core (%)"
    )

    if not shared_y:
        axes[1].set_ylabel(
            "Cell abundance within core (%)"
        )

    # =========================================================
    # 9. Shared legend
    # =========================================================
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=tissue_colors[tissue],
            markeredgecolor="#222222",
            markeredgewidth=0.6,
            markersize=8,
            label=tissue,
        )
        for tissue in final_tissue_order
    ]

    fig.legend(
        handles=legend_handles,
        title="Tissue annotation",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(final_tissue_order),
    )

    # fig.suptitle(
    #     "AM and AT2 abundance across donor cores",
    #     fontsize=15,
    #     fontweight="bold",
    # )

    for axis in fig.axes:
        axis.grid(False)

    # =========================================================
    # 10. Save
    # =========================================================
    if save is not None:
        fig.savefig(
            save,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    # Remove helper columns from the returned wide table
    output_wide = summary_wide.drop(
        columns=[
            "_core_slot",
            "_core_label",
            "_x_position",
        ],
        errors="ignore",
    )

    output_long = summary_long.drop(
        columns=[
            "_core_slot",
            "_core_label",
            "_x_position",
        ],
        errors="ignore",
    )

    return (
        fig,
        axes,
        output_wide,
        output_long,
    )


def plot_am_at2_spatial(
    adata,
    core_id,
    *,
    core_col="core_id",
    celltype_col="CellType_refined",
    mhcii_col="MHCII_group",
    am_score_col="MHCIIhi_score",
    spatial_key="spatial",
    am_labels=("AM", "Alveolar Macrophage"),
    at2_labels=("AT2",),
    am_groups=("MHCIIhi",),
    color_mode="group",
    am_group_colors=None,
    at2_color="#9FD3C7",
    background_color="#D9D9D9",
    score_cmap=None,
    score_vmin="p1",
    score_vmax="p99",
    score_center=0,
    background_size=4,
    am_size=13,
    at2_size=10,
    background_alpha=0.25,
    target_alpha=0.95,
    edgecolor="#333333",
    linewidth=0.20,
    show_separate_panels=True,
    show_axes=False,
    invert_y=False,
    rasterized=True,
    title=None,
    figsize=None,
    save=None,
    dpi=300,
):
    """
    Plot selected AM populations together with AT2 cells.

    Parameters
    ----------
    color_mode : {"group", "score"}
        "group":
            Color AM cells according to MHCII_group.

        "score":
            Color AM cells continuously using am_score_col.

    am_groups : str, sequence or None
        AM groups to include.

        Examples
        --------
        ("MHCIIhi",)
        ("MHCIIlo",)
        ("MHCIIhi", "MHCIIlo")
        None  -> include all AM cells

    score_vmin, score_vmax : float, percentile string or None
        Examples: "p1", "p99", 0, 3.

    Returns
    -------
    tuple
        Matplotlib Figure, Axes array, and the plotted cell-level table.
    """

    # =========================================================
    # 1. Validate inputs
    # =========================================================
    color_mode = str(color_mode).lower()

    if color_mode not in {"group", "score"}:
        raise ValueError(
            "color_mode must be 'group' or 'score'."
        )

    required_columns = {
        core_col,
        celltype_col,
    }

    if color_mode == "group" or am_groups is not None:
        required_columns.add(mhcii_col)

    if color_mode == "score":
        required_columns.add(am_score_col)

    missing_columns = required_columns.difference(
        adata.obs.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing adata.obs columns: {sorted(missing_columns)}"
        )

    if spatial_key not in adata.obsm:
        alternative_key = f"X_{spatial_key}"

        if alternative_key in adata.obsm:
            spatial_key = alternative_key
        else:
            raise ValueError(
                f"Neither adata.obsm['{spatial_key}'] nor "
                f"adata.obsm['{alternative_key}'] exists."
            )

    if isinstance(am_groups, str):
        am_groups = (am_groups,)

    elif am_groups is not None:
        am_groups = tuple(am_groups)

    # =========================================================
    # 2. Default macaron colors
    # =========================================================
    default_am_colors = {
        "MHCIIhi": "#E7B2B6",
        "MHCIIlo": "#B7C3E0",
        "Ambiguous": "#CFCFCF",
        "Unassigned": "#AFAFAF",
    }

    if am_group_colors is None:
        am_group_colors = default_am_colors
    else:
        am_group_colors = {
            **default_am_colors,
            **am_group_colors,
        }

    if score_cmap is None:
        score_cmap = LinearSegmentedColormap.from_list(
            "MHCIIhi_score_macaron",
            [
                "#B7C3E0",
                "#F5F3F1",
                "#E7B2B6",
            ],
        )

    elif isinstance(score_cmap, str):
        score_cmap = plt.get_cmap(score_cmap)

    # =========================================================
    # 3. Select core
    # =========================================================
    core_mask = (
        adata.obs[core_col]
        .astype("string")
        .eq(str(core_id))
        .fillna(False)
        .to_numpy()
    )

    n_core_cells = int(core_mask.sum())

    if n_core_cells == 0:
        raise ValueError(
            f"No cells found for {core_col}='{core_id}'."
        )

    selected_obs = adata.obs.loc[
        core_mask
    ].copy()

    coordinates = np.asarray(
        adata.obsm[spatial_key][core_mask]
    )

    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(
            "Spatial coordinates must have at least two columns."
        )

    # =========================================================
    # 4. Identify cell populations
    # =========================================================
    cell_types = (
        selected_obs[celltype_col]
        .astype("string")
    )

    is_am = cell_types.isin(
        [str(label) for label in am_labels]
    )

    is_at2 = cell_types.isin(
        [str(label) for label in at2_labels]
    )

    if mhcii_col in selected_obs.columns:
        mhcii_groups = (
            selected_obs[mhcii_col]
            .astype("string")
        )
    else:
        mhcii_groups = pd.Series(
            pd.NA,
            index=selected_obs.index,
            dtype="string",
        )

    # Include all AM or selected MHCII groups
    if am_groups is None:
        selected_am = is_am.copy()
    else:
        selected_am = (
            is_am
            & mhcii_groups.isin(am_groups)
        )

    if (selected_am & is_at2).any():
        raise ValueError(
            "Some cells are classified as both AM and AT2."
        )

    if color_mode == "score":
        am_scores = pd.to_numeric(
            selected_obs[am_score_col],
            errors="coerce",
        )

        selected_am_with_score = (
            selected_am
            & am_scores.notna()
        )
    else:
        am_scores = pd.Series(
            np.nan,
            index=selected_obs.index,
        )

        selected_am_with_score = selected_am

    # =========================================================
    # 5. Plotting dataframe
    # =========================================================
    spatial_data = pd.DataFrame(
        {
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "CellType": cell_types.to_numpy(),
            "MHCII_group": mhcii_groups.to_numpy(),
            "MHCIIhi_score": am_scores.to_numpy(),
            "is_AM": is_am.to_numpy(),
            "is_selected_AM": selected_am.to_numpy(),
            "is_AT2": is_at2.to_numpy(),
        },
        index=selected_obs.index,
    )

    n_at2 = int(is_at2.sum())
    n_am_total = int(is_am.sum())
    n_selected_am = int(selected_am.sum())
    n_scored_am = int(selected_am_with_score.sum())

    print(f"Core: {core_id}")
    print(f"Total cells: {n_core_cells:,}")
    print(f"AT2: {n_at2:,}")
    print(f"All AM: {n_am_total:,}")
    print(f"Selected AM: {n_selected_am:,}")

    if color_mode == "score":
        print(f"Selected AM with score: {n_scored_am:,}")

    # =========================================================
    # 6. Resolve score limits
    # =========================================================
    def resolve_limit(value, data, default):
        if value is None:
            return default

        if isinstance(value, str):
            value_lower = value.lower()

            if value_lower.startswith("p"):
                percentile = float(
                    value_lower[1:]
                )

                return float(
                    np.nanpercentile(
                        data,
                        percentile,
                    )
                )

            raise ValueError(
                f"Invalid limit specification: {value}"
            )

        return float(value)

    score_norm = None

    if color_mode == "score":
        plotted_scores = am_scores.loc[
            selected_am_with_score
        ].to_numpy(dtype=float)

        if len(plotted_scores) == 0:
            raise ValueError(
                "No selected AM cells have a non-missing "
                f"'{am_score_col}' value."
            )

        actual_vmin = resolve_limit(
            score_vmin,
            plotted_scores,
            np.nanmin(plotted_scores),
        )

        actual_vmax = resolve_limit(
            score_vmax,
            plotted_scores,
            np.nanmax(plotted_scores),
        )

        if actual_vmin >= actual_vmax:
            padding = max(
                abs(actual_vmin) * 0.01,
                1e-6,
            )

            actual_vmin -= padding
            actual_vmax += padding

        if (
            score_center is not None
            and actual_vmin < score_center < actual_vmax
        ):
            score_norm = TwoSlopeNorm(
                vmin=actual_vmin,
                vcenter=score_center,
                vmax=actual_vmax,
            )
        else:
            score_norm = Normalize(
                vmin=actual_vmin,
                vmax=actual_vmax,
            )

    # =========================================================
    # 7. Spatial limits
    # =========================================================
    x = spatial_data["x"].to_numpy()
    y = spatial_data["y"].to_numpy()

    finite = np.isfinite(x) & np.isfinite(y)

    x_min, x_max = (
        np.nanmin(x[finite]),
        np.nanmax(x[finite]),
    )

    y_min, y_max = (
        np.nanmin(y[finite]),
        np.nanmax(y[finite]),
    )

    x_padding = max(x_max - x_min, 1) * 0.025
    y_padding = max(y_max - y_min, 1) * 0.025

    x_limits = (
        x_min - x_padding,
        x_max + x_padding,
    )

    if invert_y:
        y_limits = (
            y_max + y_padding,
            y_min - y_padding,
        )
    else:
        y_limits = (
            y_min - y_padding,
            y_max + y_padding,
        )

    # =========================================================
    # 8. Figure
    # =========================================================
    n_panels = (
        3 if show_separate_panels else 1
    )

    if figsize is None:
        figsize = (
            4.6 * n_panels,
            4.8,
        )

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
        facecolor="white",
    )

    axes = axes.ravel()

    # =========================================================
    # 9. Plot helpers
    # =========================================================
    def plot_background(ax):
        ax.scatter(
            x,
            y,
            s=background_size,
            color=background_color,
            alpha=background_alpha,
            edgecolors="none",
            rasterized=rasterized,
            zorder=1,
        )

    def plot_at2(ax):
        data = spatial_data.loc[
            spatial_data["is_AT2"]
        ]

        ax.scatter(
            data["x"],
            data["y"],
            s=at2_size,
            color=at2_color,
            alpha=target_alpha,
            edgecolors=edgecolor,
            linewidths=linewidth,
            rasterized=rasterized,
            zorder=2,
        )

    def plot_am_groups(ax):
        if am_groups is None:
            groups_to_plot = (
                mhcii_groups.loc[is_am]
                .fillna("Unassigned")
                .drop_duplicates()
                .tolist()
            )
        else:
            groups_to_plot = list(am_groups)

        fallback_colors = plt.get_cmap(
            "Set2"
        )(
            np.linspace(
                0,
                1,
                max(len(groups_to_plot), 1),
            )
        )

        for group_index, group in enumerate(
            groups_to_plot
        ):
            group_mask = (
                selected_am
                & mhcii_groups.fillna(
                    "Unassigned"
                ).eq(group)
            )

            group_data = spatial_data.loc[
                group_mask
            ]

            color = am_group_colors.get(
                group,
                fallback_colors[group_index],
            )

            ax.scatter(
                group_data["x"],
                group_data["y"],
                s=am_size,
                color=color,
                alpha=target_alpha,
                edgecolors=edgecolor,
                linewidths=linewidth,
                rasterized=rasterized,
                zorder=3 + group_index,
            )

        return groups_to_plot

    def plot_am_scores(ax):
        data = spatial_data.loc[
            selected_am_with_score
        ].sort_values("MHCIIhi_score")

        scatter = ax.scatter(
            data["x"],
            data["y"],
            c=data["MHCIIhi_score"],
            cmap=score_cmap,
            norm=score_norm,
            s=am_size,
            alpha=target_alpha,
            edgecolors=edgecolor,
            linewidths=linewidth,
            rasterized=rasterized,
            zorder=3,
        )

        return scatter

    def format_axis(ax):
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("white")
        ax.grid(False)

        if show_axes:
            ax.set_xlabel("Spatial 1")
            ax.set_ylabel("Spatial 2")

            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=4)
            )

            ax.yaxis.set_major_locator(
                MaxNLocator(nbins=4)
            )

            ax.tick_params(
                labelsize=8,
                length=3,
                width=0.8,
                direction="out",
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

    # =========================================================
    # 10. Overlay panel
    # =========================================================
    overlay_ax = axes[0]

    plot_background(overlay_ax)
    plot_at2(overlay_ax)

    if color_mode == "group":
        plotted_groups = plot_am_groups(
            overlay_ax
        )
        score_scatter = None
    else:
        plotted_groups = None
        score_scatter = plot_am_scores(
            overlay_ax
        )

    overlay_ax.set_title(
        "AT2 and AM overlay",
        fontsize=12,
        fontweight="bold",
    )

    format_axis(overlay_ax)

    # =========================================================
    # 11. Separate panels
    # =========================================================
    if show_separate_panels:
        at2_ax = axes[1]

        plot_background(at2_ax)
        plot_at2(at2_ax)

        at2_ax.set_title(
            f"AT2\nn={n_at2:,}",
            fontsize=12,
            fontweight="bold",
        )

        format_axis(at2_ax)

        am_ax = axes[2]

        plot_background(am_ax)

        if color_mode == "group":
            plot_am_groups(am_ax)
        else:
            plot_am_scores(am_ax)

        if am_groups is None:
            am_selection_title = "All AM"
        else:
            am_selection_title = " + ".join(
                am_groups
            )

        am_ax.set_title(
            f"{am_selection_title} AM\n"
            f"n={n_selected_am:,}",
            fontsize=12,
            fontweight="bold",
        )

        format_axis(am_ax)

    # =========================================================
    # 12. Legend or continuous colorbar
    # =========================================================
    if color_mode == "group":
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=at2_color,
                markeredgecolor=edgecolor,
                markersize=7,
                label=f"AT2 (n={n_at2:,})",
            )
        ]

        for group in plotted_groups:
            group_count = int(
                (
                    selected_am
                    & mhcii_groups.fillna(
                        "Unassigned"
                    ).eq(group)
                ).sum()
            )

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=(
                        am_group_colors.get(
                            group,
                            "#999999",
                        )
                    ),
                    markeredgecolor=edgecolor,
                    markersize=7,
                    label=(
                        f"AM {group} "
                        f"(n={group_count:,})"
                    ),
                )
            )

        overlay_ax.legend(
            handles=legend_handles,
            frameon=False,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            borderaxespad=0,
        )

    else:
        overlay_ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=at2_color,
                    markeredgecolor=edgecolor,
                    markersize=7,
                    label=f"AT2 (n={n_at2:,})",
                )
            ],
            frameon=False,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            borderaxespad=0,
        )

        colorbar = fig.colorbar(
            score_scatter,
            ax=axes.tolist(),
            fraction=0.025,
            pad=0.02,
        )

        colorbar.set_label(
            am_score_col,
            fontsize=10,
        )

        colorbar.ax.grid(False)

    # =========================================================
    # 13. Overall title
    # =========================================================
    if title is None:
        title = f"{core_id} spatial distribution"

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    for axis in fig.axes:
        axis.grid(False)

    if save is not None:
        fig.savefig(
            save,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    return fig, axes, spatial_data


def test_continuous_mhcii_at2_proximity(
    adata,
    score_col="MHCIIhi_score",
    celltype_col="CellType_refined",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    am_labels=("AM", "Alveolar Macrophage"),
    at2_labels=("AT2",),
    spatial_key="spatial",
    min_am=10,
    min_at2=5,
    n_permutations=1000,
    coordinate_scale=1.0,
    random_state=123,
):
    """
    Test whether continuous MHCIIhi_score in AMs is associated with
    distance to the nearest AT2 cell.

    Analysis is performed separately within each core_id.

    Parameters
    ----------
    coordinate_scale
        Multiply spatial coordinates by this value.
        Use 1.0 when coordinates are already in micrometres.

    Returns
    -------
    dict containing:
        cell_distances : one row per AM
        core_results   : one correlation/permutation test per core
        donor_results  : correlations aggregated within donor and tissue
        group_tests    : donor-level tests against zero

    Notes
    -----
    Tissue-specific group tests use one donor-level value per tissue. The
    pooled ``All`` row can contain more than one donor-tissue value from the
    same donor and is therefore descriptive when donors contribute multiple
    tissues. The within-core score shuffle assumes exchangeability and is not
    a spatially constrained permutation of an autocorrelated score field.
    """

    required_obs = {
        score_col,
        celltype_col,
        core_col,
        donor_col,
        tissue_col,
    }

    missing = required_obs.difference(adata.obs.columns)
    if missing:
        raise KeyError(f"Missing adata.obs columns: {sorted(missing)}")

    if spatial_key not in adata.obsm:
        raise KeyError(f"{spatial_key!r} is not present in adata.obsm")

    rng = np.random.default_rng(random_state)

    obs = adata.obs.copy()
    coordinates = np.asarray(adata.obsm[spatial_key])[:, :2]
    coordinates = coordinates.astype(float) * coordinate_scale

    obs["_x"] = coordinates[:, 0]
    obs["_y"] = coordinates[:, 1]
    obs["_cell_id"] = obs.index.astype(str)

    am_labels = set(am_labels)
    at2_labels = set(at2_labels)

    cell_tables = []
    core_records = []

    for core_id, core_df in obs.groupby(core_col, observed=True):

        core_df = core_df.copy()

        valid_coordinates = (
            np.isfinite(core_df["_x"]) &
            np.isfinite(core_df["_y"])
        )
        core_df = core_df.loc[valid_coordinates]

        am_mask = (
            core_df[celltype_col].isin(am_labels) &
            pd.to_numeric(
                core_df[score_col],
                errors="coerce",
            ).notna()
        )

        at2_mask = core_df[celltype_col].isin(at2_labels)

        am_df = core_df.loc[am_mask].copy()
        at2_df = core_df.loc[at2_mask].copy()

        if len(am_df) < min_am or len(at2_df) < min_at2:
            continue

        scores = pd.to_numeric(
            am_df[score_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        if np.nanstd(scores) == 0:
            continue

        at2_tree = cKDTree(
            at2_df[["_x", "_y"]].to_numpy(dtype=float)
        )

        nearest_distance, nearest_index = at2_tree.query(
            am_df[["_x", "_y"]].to_numpy(dtype=float),
            k=1,
        )

        if np.nanstd(nearest_distance) == 0:
            continue

        rho, asymptotic_p = spearmanr(
            scores,
            nearest_distance,
            nan_policy="omit",
        )

        # Shuffle scores among AM positions while keeping AT2 positions fixed.
        null_rho = np.empty(n_permutations, dtype=float)

        for permutation in range(n_permutations):
            shuffled_scores = rng.permutation(scores)

            null_rho[permutation] = spearmanr(
                shuffled_scores,
                nearest_distance,
                nan_policy="omit",
            ).statistic

        # Directional alternative:
        # higher MHCIIhi_score is associated with shorter distance.
        p_closer = (
            1 + np.sum(null_rho <= rho)
        ) / (n_permutations + 1)

        p_two_sided = (
            1 + np.sum(np.abs(null_rho) >= abs(rho))
        ) / (n_permutations + 1)

        donor_values = am_df[donor_col].dropna().unique()
        tissue_values = am_df[tissue_col].dropna().unique()

        donor_id = (
            donor_values[0] if len(donor_values) == 1 else pd.NA
        )
        tissue = (
            tissue_values[0] if len(tissue_values) == 1 else pd.NA
        )

        core_records.append({
            core_col: core_id,
            donor_col: donor_id,
            tissue_col: tissue,
            "n_AM": len(am_df),
            "n_AT2": len(at2_df),
            "spearman_rho": rho,
            "asymptotic_p": asymptotic_p,
            "permutation_p_closer": p_closer,
            "permutation_p_two_sided": p_two_sided,
            "median_AT2_distance": np.median(nearest_distance),
        })

        nearest_at2_ids = (
            at2_df.iloc[nearest_index]["_cell_id"].to_numpy()
        )

        cell_tables.append(pd.DataFrame({
            "cell_id": am_df["_cell_id"].to_numpy(),
            core_col: core_id,
            donor_col: donor_id,
            tissue_col: tissue,
            score_col: scores,
            "nearest_AT2_distance": nearest_distance,
            "nearest_AT2_cell_id": nearest_at2_ids,
        }))

    if not core_records:
        raise ValueError(
            "No cores passed the minimum AM/AT2 requirements. "
            "Check cell-type labels and minimum-cell settings."
        )

    core_results = pd.DataFrame(core_records)

    for p_col in [
        "permutation_p_closer",
        "permutation_p_two_sided",
    ]:
        core_results[f"{p_col}_FDR"] = multipletests(
            core_results[p_col],
            method="fdr_bh",
        )[1]

    cell_distances = pd.concat(
        cell_tables,
        ignore_index=True,
    )

    # Fisher transformation before aggregating correlations.
    core_results["fisher_z"] = np.arctanh(
        core_results["spearman_rho"].clip(-0.999999, 0.999999)
    )

    donor_results = (
        core_results
        .groupby(
            [donor_col, tissue_col],
            observed=True,
            dropna=False,
        )
        .agg(
            n_cores=(core_col, "nunique"),
            n_AM=("n_AM", "sum"),
            n_AT2=("n_AT2", "sum"),
            mean_fisher_z=("fisher_z", "mean"),
        )
        .reset_index()
    )

    donor_results["mean_spearman_rho"] = np.tanh(
        donor_results["mean_fisher_z"]
    )

    # Donor-level inference; donors, rather than cells, are replicates.
    test_records = []

    all_tissues = ["All"] + list(
        donor_results[tissue_col].dropna().unique()
    )

    for tissue in all_tissues:

        if tissue == "All":
            values = donor_results["mean_fisher_z"].dropna()
        else:
            values = donor_results.loc[
                donor_results[tissue_col] == tissue,
                "mean_fisher_z",
            ].dropna()

        record = {
            tissue_col: tissue,
            "n_donors": len(values),
            "median_spearman_rho": (
                np.tanh(values.median()) if len(values) else np.nan
            ),
            "wilcoxon_statistic": np.nan,
            "p_closer": np.nan,
        }

        if len(values) >= 3 and not np.allclose(values, 0):
            result = wilcoxon(
                values,
                alternative="less",
                zero_method="wilcox",
            )

            record["wilcoxon_statistic"] = result.statistic
            record["p_closer"] = result.pvalue

        test_records.append(record)

    group_tests = pd.DataFrame(test_records)

    valid = group_tests["p_closer"].notna()
    group_tests["p_closer_FDR"] = np.nan

    if valid.any():
        group_tests.loc[valid, "p_closer_FDR"] = multipletests(
            group_tests.loc[valid, "p_closer"],
            method="fdr_bh",
        )[1]

    return {
        "cell_distances": cell_distances,
        "core_results": core_results,
        "donor_results": donor_results,
        "group_tests": group_tests,
    }


def calculate_nhood_enrichment_by_core(
    adata,
    cluster_key="CellType_MHCII",
    focus_label="AT2",
    neighbor_types=None,
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    radius=50,
    min_focus_cells=5,
    min_neighbor_cells=3,
    n_perms=1000,
    random_state=123,
    exclude_self=True,
):
    """
    Calculate Squidpy neighborhood enrichment separately for each core.

    Positive z-score:
        more spatial contacts than expected.

    Negative z-score:
        fewer spatial contacts than expected.

    Parameters
    ----------
    neighbor_types
        Cell types to extract. If None, extract every cell type.
    radius
        Spatial neighborhood radius. Assumes coordinates use the same
        spatial units, normally µm for Xenium.

    Returns
    -------
    pandas.DataFrame
        Core-level focus-to-neighbor enrichment z-scores and cell counts.
    """

    required = {
        cluster_key,
        core_col,
        donor_col,
        tissue_col,
    }

    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(
            f"Missing adata.obs columns: {sorted(missing)}"
        )

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"{spatial_key!r} not found in adata.obsm"
        )

    records = []

    grouped_indices = adata.obs.groupby(
        core_col,
        observed=True,
    ).groups

    for core_number, (core_id, indices) in enumerate(
        grouped_indices.items()
    ):
        adata_core = adata[list(indices)].copy()

        valid = adata_core.obs[cluster_key].notna()
        adata_core = adata_core[valid].copy()

        adata_core.obs[cluster_key] = (
            adata_core.obs[cluster_key]
            .astype(str)
            .astype("category")
        )

        counts = adata_core.obs[cluster_key].value_counts()

        if focus_label not in counts.index:
            continue

        if counts[focus_label] < min_focus_cells:
            continue

        donor_values = (
            adata_core.obs[donor_col]
            .dropna()
            .astype(str)
            .unique()
        )

        tissue_values = (
            adata_core.obs[tissue_col]
            .dropna()
            .astype(str)
            .unique()
        )

        donor_id = (
            donor_values[0]
            if len(donor_values) == 1
            else pd.NA
        )

        tissue = (
            tissue_values[0]
            if len(tissue_values) == 1
            else pd.NA
        )

        # Construct the spatial graph
        if hasattr(sq.gr, "spatial_neighbors_radius"):
            sq.gr.spatial_neighbors_radius(
                adata_core,
                radius=radius,
                spatial_key=spatial_key,
                key_added="spatial",
            )
        else:
            # Compatibility with older Squidpy
            sq.gr.spatial_neighbors(
                adata_core,
                coord_type="generic",
                radius=radius,
                spatial_key=spatial_key,
                key_added="spatial",
            )

        # Permutation-based neighborhood enrichment
        sq.gr.nhood_enrichment(
            adata_core,
            cluster_key=cluster_key,
            n_perms=n_perms,
            seed=random_state + core_number,
            show_progress_bar=False,
        )

        categories = list(
            adata_core.obs[cluster_key].cat.categories
        )

        result = adata_core.uns[
            f"{cluster_key}_nhood_enrichment"
        ]

        zscore_matrix = result["zscore"]
        count_matrix = result["count"]

        focus_index = categories.index(focus_label)

        if neighbor_types is None:
            selected_neighbors = categories
        else:
            selected_neighbors = [
                celltype
                for celltype in neighbor_types
                if celltype in categories
            ]

        for neighbor in selected_neighbors:

            if exclude_self and neighbor == focus_label:
                continue

            n_neighbor = int(counts.get(neighbor, 0))

            if n_neighbor < min_neighbor_cells:
                continue

            neighbor_index = categories.index(neighbor)

            records.append({
                core_col: core_id,
                donor_col: donor_id,
                tissue_col: tissue,
                "focus_celltype": focus_label,
                "neighbor_celltype": neighbor,
                "radius": radius,
                "n_focus": int(counts[focus_label]),
                "n_neighbor": n_neighbor,
                "observed_contacts": float(
                    count_matrix[
                        focus_index,
                        neighbor_index,
                    ]
                ),
                "enrichment_zscore": float(
                    zscore_matrix[
                        focus_index,
                        neighbor_index,
                    ]
                ),
            })

    results = pd.DataFrame(records)

    if results.empty:
        raise ValueError(
            "No cores passed the filtering criteria. Check the "
            "cell-type labels, radius and minimum cell thresholds."
        )

    return results


def plot_nhood_enrichment_donor_tissue(
    donor_summary,
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    value_col="mean_enrichment_zscore",
    neighbor_col="neighbor_celltype",
    tissue_order=("A", "B", "V"),
    neighbor_order=None,
    palette=None,
    save_prefix=None,
    dpi=300,
):
    """Plot donor-wise and tissue-wise neighborhood enrichment.

    Parameters
    ----------
    donor_summary : pandas.DataFrame
        Donor summary from summarize_nhood_by_donor.
    donor_col, tissue_col, value_col, neighbor_col : str
        Columns defining donor, tissue, enrichment value, and neighbor type.
    tissue_order : sequence of str
        Tissue display order.
    neighbor_order : sequence of str or None
        Neighbor display order; observed order is used when omitted.
    palette : mapping or None
        Optional neighbor-type color mapping.
    save_prefix : path-like or None
        Optional prefix used to save both figures.
    dpi : int
        Resolution used when saving.

    Returns
    -------
    tuple
        Donor-wise Matplotlib Figure and tissue-summary Figure.
    """

    plot_df = donor_summary.copy()

    plot_df = plot_df.loc[
        plot_df[tissue_col].isin(tissue_order)
    ].copy()

    if neighbor_order is None:
        neighbor_order = list(
            plot_df[neighbor_col].dropna().unique()
        )
    else:
        neighbor_order = [
            x for x in neighbor_order
            if x in plot_df[neighbor_col].unique()
        ]

    default_palette = {
        "MHCIIhi AM": "#D95C69",
        "MHCIIlo AM": "#5276B5",
        "Ambiguous AM": "#8E8E8E",
        "Unassigned AM": "#B8B8B8",
    }

    if palette is None:
        palette = default_palette.copy()

    fallback_colors = sns.color_palette(
        "Set2",
        n_colors=max(len(neighbor_order), 1),
    )

    for index, celltype in enumerate(neighbor_order):
        if celltype not in palette:
            palette[celltype] = fallback_colors[index]

    sns.set_theme(
        style="white",
        context="paper",
        font_scale=1.05,
    )

    # -----------------------------------------------------
    # Donor-wise figure, separated by tissue
    # -----------------------------------------------------
    available_tissues = [
        tissue
        for tissue in tissue_order
        if tissue in plot_df[tissue_col].unique()
    ]

    fig1, axes = plt.subplots(
        1,
        len(available_tissues),
        figsize=(5.2 * len(available_tissues), 4.3),
        sharey=True,
        squeeze=False,
    )

    axes = axes.ravel()

    for ax, tissue in zip(axes, available_tissues):

        tissue_df = plot_df.loc[
            plot_df[tissue_col] == tissue
        ].copy()

        donor_order = (
            tissue_df.groupby(donor_col, observed=True)[value_col]
            .mean()
            .sort_values()
            .index
            .tolist()
        )

        sns.stripplot(
            data=tissue_df,
            x=donor_col,
            y=value_col,
            hue=neighbor_col,
            order=donor_order,
            hue_order=neighbor_order,
            palette=palette,
            dodge=True,
            jitter=False,
            size=6,
            edgecolor="black",
            linewidth=0.55,
            ax=ax,
        )

        ax.axhline(
            0,
            color="#555555",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )

        ax.set_title(
            f"Tissue {tissue}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Donor")
        ax.tick_params(
            axis="x",
            rotation=90,
            labelsize=8,
        )

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax is axes[0]:
            ax.set_ylabel("AT2 neighborhood enrichment z-score")
        else:
            ax.set_ylabel("")

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[0].get_legend_handles_labels()

    fig1.legend(
        handles,
        labels,
        title="Cell type near AT2",
        frameon=False,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
    )

    fig1.tight_layout(rect=(0, 0, 0.88, 1))

    # -----------------------------------------------------
    # Tissue-wise summary
    # -----------------------------------------------------
    fig2, ax = plt.subplots(figsize=(7.3, 4.8))

    sns.barplot(
        data=plot_df,
        x=tissue_col,
        y=value_col,
        hue=neighbor_col,
        order=available_tissues,
        hue_order=neighbor_order,
        palette=palette,
        errorbar="se",
        capsize=0.12,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.72,
        ax=ax,
    )

    sns.stripplot(
        data=plot_df,
        x=tissue_col,
        y=value_col,
        hue=neighbor_col,
        order=available_tissues,
        hue_order=neighbor_order,
        palette=palette,
        dodge=True,
        jitter=0.12,
        size=4.5,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.9,
        ax=ax,
    )

    ax.axhline(
        0,
        color="#555555",
        linewidth=0.8,
        linestyle="--",
        zorder=0,
    )

    ax.set_xlabel("Tissue annotation")
    ax.set_ylabel("AT2 neighborhood enrichment z-score")
    ax.set_title(
        "Cell-type enrichment in the AT2 neighborhood",
        fontsize=12,
        fontweight="bold",
    )

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove duplicated legends from the two seaborn layers
    handles, labels = ax.get_legend_handles_labels()
    n_groups = len(neighbor_order)

    ax.legend(
        handles[:n_groups],
        labels[:n_groups],
        title="Cell type near AT2",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig2.tight_layout()

    if save_prefix is not None:
        fig1.savefig(
            f"{save_prefix}_donor_wise.pdf",
            dpi=dpi,
            bbox_inches="tight",
        )
        fig2.savefig(
            f"{save_prefix}_tissue_wise.pdf",
            dpi=dpi,
            bbox_inches="tight",
        )

    return fig1, fig2


def assign_balanced_mhcii_score_groups(
    adata,
    score_col="MHCIIhi_score",
    celltype_col="CellType_refined",
    am_labels=("AM", "Alveolar Macrophage"),
    groupby="core_id",
    tail_fraction=0.25,
    n_per_tail=None,
    min_am_cells=20,
    output_col="MHCII_score_group",
    combined_col="CellType_MHCII_balanced",
    hi_label="MHCIIhi",
    lo_label="MHCIIlo",
    ambiguous_label="Ambiguous",
    unassigned_label="Unassigned",
    copy=False,
):
    """
    Assign balanced high- and low-MHCIIhi-score AM groups.

    Within every groupby unit:
      - highest scores -> MHCIIhi
      - lowest scores  -> MHCIIlo
      - middle scores  -> Ambiguous

    High and low groups always contain the same number of cells.

    Parameters
    ----------
    tail_fraction
        Fraction assigned to each tail. For example, 0.25 gives:
        bottom 25% low, middle 50% ambiguous, top 25% high.

    n_per_tail
        Optionally specify an exact number of cells per tail.
        If provided, this overrides tail_fraction.

    groupby
        Usually "core_id". Can instead be "donor_id" or None for
        global assignment.

    Returns
    -------
    tuple
        With copy=False, returns the group summary and cutoff table after
        modifying adata.obs. With copy=True, returns the copied AnnData first,
        followed by those two tables.

    Notes
    -----
    These are relative within-group score tails, not evidence of two discrete
    biological clusters. Cells tied at a tail boundary are ordered by the
    stable input order, so alternative tail fractions and the continuous score
    should be retained as sensitivity and primary analyses, respectively.
    """

    if copy:
        adata = adata.copy()

    required = {score_col, celltype_col}

    if groupby is not None:
        if isinstance(groupby, str):
            groupby_cols = [groupby]
        else:
            groupby_cols = list(groupby)

        required.update(groupby_cols)
    else:
        groupby_cols = []

    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(
            f"Missing adata.obs columns: {sorted(missing)}"
        )

    if not 0 < tail_fraction < 0.5:
        raise ValueError(
            "tail_fraction must be greater than 0 and less than 0.5."
        )

    obs = adata.obs
    am_mask = obs[celltype_col].isin(am_labels)

    numeric_score = pd.to_numeric(
        obs[score_col],
        errors="coerce",
    )

    # AMs initially remain unassigned
    group_assignment = pd.Series(
        pd.NA,
        index=obs.index,
        dtype="string",
    )

    group_assignment.loc[am_mask] = unassigned_label

    valid_am_mask = (
        am_mask &
        numeric_score.notna() &
        np.isfinite(numeric_score)
    )

    cutoff_records = []

    if groupby_cols:
        grouped_indices = (
            obs.loc[valid_am_mask]
            .groupby(
                groupby_cols,
                observed=True,
                dropna=False,
            )
            .groups
        )
    else:
        grouped_indices = {
            "All": obs.index[valid_am_mask]
        }

    for group_name, indices in grouped_indices.items():

        indices = pd.Index(indices)
        scores = numeric_score.loc[indices]

        n_cells = len(scores)

        if n_cells < min_am_cells:
            continue

        if scores.nunique() < 2:
            continue

        if n_per_tail is None:
            n_tail = int(np.floor(n_cells * tail_fraction))
        else:
            n_tail = int(n_per_tail)

        n_tail = min(n_tail, n_cells // 2)

        if n_tail < 1:
            continue

        # Stable sorting ensures exactly equal tail sizes
        sorted_indices = scores.sort_values(
            kind="mergesort"
        ).index

        low_indices = sorted_indices[:n_tail]
        high_indices = sorted_indices[-n_tail:]
        middle_indices = sorted_indices[
            n_tail:(n_cells - n_tail)
        ]

        group_assignment.loc[low_indices] = lo_label
        group_assignment.loc[high_indices] = hi_label
        group_assignment.loc[middle_indices] = ambiguous_label

        low_cutoff = scores.loc[low_indices].max()
        high_cutoff = scores.loc[high_indices].min()

        record = {
            "group": group_name,
            "n_AM": n_cells,
            "n_per_tail": n_tail,
            "n_MHCIIhi": len(high_indices),
            "n_MHCIIlo": len(low_indices),
            "n_Ambiguous": len(middle_indices),
            "low_score_max": low_cutoff,
            "high_score_min": high_cutoff,
            "score_gap": high_cutoff - low_cutoff,
        }

        cutoff_records.append(record)

    group_order = [
        hi_label,
        lo_label,
        ambiguous_label,
        unassigned_label,
    ]

    adata.obs[output_col] = pd.Categorical(
        group_assignment,
        categories=group_order,
        ordered=True,
    )

    # Create combined cell-type annotation
    combined_annotation = (
        adata.obs[celltype_col]
        .astype("string")
        .copy()
    )

    for group_label in group_order:
        mask = (
            am_mask &
            adata.obs[output_col].astype("string").eq(
                group_label
            )
        )

        combined_annotation.loc[mask] = (
            f"{group_label} AM"
        )

    adata.obs[combined_col] = (
        combined_annotation
        .astype("category")
    )

    cutoff_table = pd.DataFrame(cutoff_records)

    summary_columns = groupby_cols + [output_col]

    group_summary = (
        adata.obs.loc[am_mask]
        .groupby(
            summary_columns,
            observed=True,
            dropna=False,
        )
        .size()
        .rename("n_cells")
        .reset_index()
    )

    if copy:
        return adata, group_summary, cutoff_table

    return group_summary, cutoff_table


def calculate_knn_niche_continuum(
    adata,
    query_celltypes=("AM", "Alveolar Macrophage"),
    query_celltype_col="CellType_refined",
    neighbor_celltype_col="CellType_refined",
    score_col="MHCIIhi_score",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    target_types=None,
    k_neighbors=15,
    min_query_cells=10,
):
    """
    Test associations between a continuous AM score and the cellular
    composition of each AM's k-nearest neighborhood.

    Neighbors are calculated separately within every core_id.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-level object with annotations and spatial coordinates.
    query_celltypes : sequence of str
        Labels identifying focal alveolar macrophages.
    query_celltype_col, neighbor_celltype_col : str
        Observation columns defining focal and neighboring cell types.
    score_col : str
        Continuous MHCII-high program score in focal AMs.
    core_col, donor_col, tissue_col : str
        Observation columns defining spatial core, donor, and tissue.
    spatial_key : str
        adata.obsm key containing two-dimensional coordinates.
    target_types : sequence of str or None
        Neighbor types to quantify; all observed types when omitted.
    k_neighbors : int
        Number of non-self nearest neighbors per focal AM.
    min_query_cells : int
        Minimum focal AM count for a core-level correlation.

    Returns
    -------
    dict containing:
        neighbor_long
        neighborhood_comp
        core_correlations
        donor_correlations
        tissue_tests
    """

    required = {
        query_celltype_col,
        neighbor_celltype_col,
        score_col,
        core_col,
        donor_col,
        tissue_col,
    }

    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(
            f"Missing adata.obs columns: {sorted(missing)}"
        )

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"{spatial_key!r} not found in adata.obsm"
        )

    if not adata.obs_names.is_unique:
        raise ValueError(
            "adata.obs_names must be unique."
        )

    obs = adata.obs.copy()

    coordinates = np.asarray(
        adata.obsm[spatial_key]
    )[:, :2].astype(float)

    obs["_x"] = coordinates[:, 0]
    obs["_y"] = coordinates[:, 1]
    obs["_cell_id"] = obs.index.astype(str)

    query_celltypes = set(query_celltypes)

    if target_types is None:
        target_types = sorted(
            obs[neighbor_celltype_col]
            .dropna()
            .astype(str)
            .unique()
        )
    else:
        target_types = list(target_types)

    neighbor_records = []
    query_metadata = []

    for core_id, core_df in obs.groupby(
        core_col,
        observed=True,
    ):
        core_df = core_df.copy()

        valid_reference = (
            core_df[neighbor_celltype_col].notna() &
            np.isfinite(core_df["_x"]) &
            np.isfinite(core_df["_y"])
        )

        reference = core_df.loc[valid_reference].copy()

        valid_query = (
            reference[query_celltype_col].isin(
                query_celltypes
            ) &
            pd.to_numeric(
                reference[score_col],
                errors="coerce",
            ).notna()
        )

        query = reference.loc[valid_query].copy()

        if len(query) < min_query_cells:
            continue

        if len(reference) <= 1:
            continue

        reference_coordinates = reference[
            ["_x", "_y"]
        ].to_numpy(dtype=float)

        query_coordinates = query[
            ["_x", "_y"]
        ].to_numpy(dtype=float)

        reference_ids = reference["_cell_id"].to_numpy()
        reference_types = (
            reference[neighbor_celltype_col]
            .astype(str)
            .to_numpy()
        )

        tree = cKDTree(reference_coordinates)

        # Request one extra neighbor because the query AM itself
        # will normally be returned at distance zero.
        requested_k = min(
            k_neighbors + 1,
            len(reference),
        )

        distances, indices = tree.query(
            query_coordinates,
            k=requested_k,
        )

        if requested_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        for row_number, (
            query_index,
            query_row,
        ) in enumerate(query.iterrows()):

            candidate_indices = np.atleast_1d(
                indices[row_number]
            )
            candidate_distances = np.atleast_1d(
                distances[row_number]
            )

            query_id = str(query_index)

            # Remove only the query cell itself.
            keep = (
                reference_ids[candidate_indices]
                != query_id
            )

            candidate_indices = candidate_indices[keep]
            candidate_distances = candidate_distances[keep]

            candidate_indices = candidate_indices[
                :k_neighbors
            ]
            candidate_distances = candidate_distances[
                :k_neighbors
            ]

            n_observed = len(candidate_indices)

            if n_observed == 0:
                continue

            query_metadata.append({
                "cell_id": query_id,
                core_col: core_id,
                donor_col: query_row[donor_col],
                tissue_col: query_row[tissue_col],
                score_col: float(query_row[score_col]),
                "n_neighbors_observed": n_observed,
                "kth_neighbor_distance": float(
                    candidate_distances[-1]
                ),
            })

            for rank, (
                neighbor_index,
                neighbor_distance,
            ) in enumerate(
                zip(
                    candidate_indices,
                    candidate_distances,
                ),
                start=1,
            ):
                neighbor_records.append({
                    "cell_id": query_id,
                    core_col: core_id,
                    donor_col: query_row[donor_col],
                    tissue_col: query_row[tissue_col],
                    score_col: float(query_row[score_col]),
                    "neighbor_rank": rank,
                    "neighbor_cell_id": (
                        reference_ids[neighbor_index]
                    ),
                    "neighbor_type": (
                        reference_types[neighbor_index]
                    ),
                    "neighbor_distance": float(
                        neighbor_distance
                    ),
                })

    if not neighbor_records:
        raise ValueError(
            "No valid neighborhoods were found."
        )

    neighbor_long = pd.DataFrame(neighbor_records)

    query_metadata = (
        pd.DataFrame(query_metadata)
        .drop_duplicates("cell_id")
    )

    # Count neighbor types for each AM
    neighbor_counts = (
        neighbor_long.loc[
            neighbor_long["neighbor_type"].isin(
                target_types
            )
        ]
        .groupby(
            ["cell_id", "neighbor_type"],
            observed=True,
        )
        .size()
        .rename("n_neighbor")
    )

    # Include zero counts for missing cell types
    complete_index = pd.MultiIndex.from_product(
        [
            query_metadata["cell_id"],
            target_types,
        ],
        names=[
            "cell_id",
            "neighbor_type",
        ],
    )

    neighborhood_comp = (
        neighbor_counts
        .reindex(
            complete_index,
            fill_value=0,
        )
        .reset_index()
        .merge(
            query_metadata,
            on="cell_id",
            how="left",
            validate="many_to_one",
        )
    )

    neighborhood_comp["neighbor_fraction"] = (
        neighborhood_comp["n_neighbor"] /
        neighborhood_comp["n_neighbors_observed"]
    )

    neighborhood_comp["neighbor_pct"] = (
        100 * neighborhood_comp["neighbor_fraction"]
    )

    # Correlation separately within each core
    correlation_records = []

    grouped = neighborhood_comp.groupby(
        [core_col, "neighbor_type"],
        observed=True,
    )

    for (
        core_id,
        neighbor_type,
    ), group_df in grouped:

        score = group_df[score_col].to_numpy(dtype=float)
        fraction = group_df[
            "neighbor_fraction"
        ].to_numpy(dtype=float)

        donor_values = (
            group_df[donor_col]
            .dropna()
            .unique()
        )

        tissue_values = (
            group_df[tissue_col]
            .dropna()
            .unique()
        )

        result = {
            core_col: core_id,
            donor_col: (
                donor_values[0]
                if len(donor_values) == 1
                else pd.NA
            ),
            tissue_col: (
                tissue_values[0]
                if len(tissue_values) == 1
                else pd.NA
            ),
            "neighbor_type": neighbor_type,
            "n_AM": len(group_df),
            "rho": np.nan,
            "p_value": np.nan,
        }

        if (
            len(group_df) >= min_query_cells and
            np.nanstd(score) > 0 and
            np.nanstd(fraction) > 0
        ):
            test = spearmanr(
                score,
                fraction,
                nan_policy="omit",
            )

            result["rho"] = test.statistic
            result["p_value"] = test.pvalue

        correlation_records.append(result)

    core_correlations = pd.DataFrame(
        correlation_records
    )

    # BH correction separately within every core
    core_correlations["FDR_within_core"] = np.nan

    for core_id, indices in core_correlations.groupby(
        core_col,
        observed=True,
    ).groups.items():

        indices = list(indices)

        valid = core_correlations.loc[
            indices,
            "p_value",
        ].notna()

        valid_indices = (
            core_correlations.loc[indices]
            .index[valid]
        )

        if len(valid_indices):
            core_correlations.loc[
                valid_indices,
                "FDR_within_core",
            ] = multipletests(
                core_correlations.loc[
                    valid_indices,
                    "p_value",
                ],
                method="fdr_bh",
            )[1]

    # Aggregate core correlations within donor and tissue
    valid_core = core_correlations.dropna(
        subset=["rho"]
    ).copy()

    valid_core["fisher_z"] = np.arctanh(
        valid_core["rho"].clip(
            -0.999999,
            0.999999,
        )
    )

    donor_correlations = (
        valid_core
        .groupby(
            [
                donor_col,
                tissue_col,
                "neighbor_type",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            n_cores=(core_col, "nunique"),
            mean_fisher_z=("fisher_z", "mean"),
        )
        .reset_index()
    )

    donor_correlations["mean_rho"] = np.tanh(
        donor_correlations["mean_fisher_z"]
    )

    # Tissue-level tests using donors as replicates
    tissue_records = []

    for (
        tissue,
        neighbor_type,
    ), group_df in donor_correlations.groupby(
        [tissue_col, "neighbor_type"],
        observed=True,
        dropna=False,
    ):

        values = (
            group_df["mean_fisher_z"]
            .dropna()
            .to_numpy()
        )

        record = {
            tissue_col: tissue,
            "neighbor_type": neighbor_type,
            "n_donors": len(values),
            "median_rho": (
                np.tanh(np.median(values))
                if len(values)
                else np.nan
            ),
            "p_value": np.nan,
        }

        if len(values) >= 3 and not np.allclose(
            values,
            0,
        ):
            test = wilcoxon(
                values,
                alternative="two-sided",
                zero_method="wilcox",
            )

            record["p_value"] = test.pvalue

        tissue_records.append(record)

    tissue_tests = pd.DataFrame(tissue_records)
    tissue_tests["FDR"] = np.nan

    # BH correction across cell types within each tissue
    for tissue, indices in tissue_tests.groupby(
        tissue_col,
        observed=True,
    ).groups.items():

        indices = list(indices)

        valid = tissue_tests.loc[
            indices,
            "p_value",
        ].notna()

        valid_indices = (
            tissue_tests.loc[indices]
            .index[valid]
        )

        if len(valid_indices):
            tissue_tests.loc[
                valid_indices,
                "FDR",
            ] = multipletests(
                tissue_tests.loc[
                    valid_indices,
                    "p_value",
                ],
                method="fdr_bh",
            )[1]

    return {
        "neighbor_long": neighbor_long,
        "neighborhood_comp": neighborhood_comp,
        "core_correlations": core_correlations,
        "donor_correlations": donor_correlations,
        "tissue_tests": tissue_tests,
    }


def calculate_radius_niche_continuum(
    adata,
    query_celltypes=("AM", "Alveolar Macrophage"),
    query_celltype_col="CellType_refined",
    neighbor_celltype_col="CellType_refined",
    score_col="MHCIIhi_score",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    target_types=None,
    radii=(25, 50, 100),
    exposure_metric="fraction",
    coordinate_scale=1.0,
    min_query_cells=10,
    alternative="two-sided",
):
    """
    Test associations between a continuous score in query cells and
    the composition of their fixed-radius spatial neighborhoods.

    Neighborhoods are calculated separately within every core_id.

    Parameters
    ----------
    adata
        AnnData containing cell annotations and spatial coordinates.

    query_celltypes
        Cell-type labels identifying the focal cells, usually AMs.

    query_celltype_col
        Column used to identify the focal/query cells.

    neighbor_celltype_col
        Column defining the neighboring cell types.

    score_col
        Continuous score measured in the query cells.

    radii
        One or more spatial radii. These must use the same units as
        adata.obsm[spatial_key] after applying coordinate_scale.

    exposure_metric
        Metric used for correlation with score_col:

        - "fraction": fraction of all neighbors belonging to each type
        - "count": number of neighboring cells of each type
        - "density": count divided by pi * radius^2
        - "presence": whether at least one neighboring cell is present

    coordinate_scale
        Multiplier applied to the spatial coordinates. Use 1.0 if the
        coordinates are already in micrometres.

    min_query_cells
        Minimum number of valid query cells needed per core.

    alternative
        Alternative hypothesis for the donor-level Wilcoxon test:
        "two-sided", "greater", or "less".

        "greater" tests whether correlations are consistently positive.

    Returns
    -------
    dict with:
        neighborhood_comp
        core_correlations
        donor_correlations
        tissue_tests
    """

    valid_metrics = {
        "fraction": "neighbor_fraction",
        "count": "n_neighbor",
        "density": "neighbor_density",
        "presence": "neighbor_presence",
    }

    if exposure_metric not in valid_metrics:
        raise ValueError(
            "exposure_metric must be one of: "
            "'fraction', 'count', 'density', or 'presence'."
        )

    metric_col = valid_metrics[exposure_metric]

    radii = sorted(
        {
            float(radius)
            for radius in np.atleast_1d(radii)
        }
    )

    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("All radii must be greater than zero.")

    required_columns = {
        query_celltype_col,
        neighbor_celltype_col,
        score_col,
        core_col,
        donor_col,
        tissue_col,
    }

    missing_columns = required_columns.difference(
        adata.obs.columns
    )

    if missing_columns:
        raise KeyError(
            "Missing adata.obs columns: "
            f"{sorted(missing_columns)}"
        )

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"{spatial_key!r} was not found in adata.obsm."
        )

    if not adata.obs_names.is_unique:
        raise ValueError(
            "adata.obs_names must be unique."
        )

    query_celltypes = set(query_celltypes)

    obs = adata.obs.copy()

    coordinates = np.asarray(
        adata.obsm[spatial_key]
    )[:, :2].astype(float)

    coordinates = coordinates * coordinate_scale

    obs["_spatial_x"] = coordinates[:, 0]
    obs["_spatial_y"] = coordinates[:, 1]
    obs["_cell_id"] = obs.index.astype(str)

    if target_types is None:
        target_types = sorted(
            obs[neighbor_celltype_col]
            .dropna()
            .astype(str)
            .unique()
        )
    else:
        target_types = list(dict.fromkeys(target_types))

    maximum_radius = max(radii)

    composition_records = []

    for core_id, core_df in obs.groupby(
        core_col,
        observed=True,
        dropna=False,
    ):
        core_df = core_df.copy()

        valid_coordinates = (
            np.isfinite(core_df["_spatial_x"]) &
            np.isfinite(core_df["_spatial_y"])
        )

        valid_reference = (
            valid_coordinates &
            core_df[neighbor_celltype_col].notna()
        )

        reference_df = core_df.loc[
            valid_reference
        ].copy()

        if len(reference_df) < 2:
            continue

        numeric_score = pd.to_numeric(
            reference_df[score_col],
            errors="coerce",
        )

        query_mask = (
            reference_df[query_celltype_col].isin(
                query_celltypes
            ) &
            numeric_score.notna() &
            np.isfinite(numeric_score)
        )

        query_df = reference_df.loc[
            query_mask
        ].copy()

        query_scores = numeric_score.loc[
            query_df.index
        ]

        if len(query_df) < min_query_cells:
            continue

        donor_values = (
            query_df[donor_col]
            .dropna()
            .astype(str)
            .unique()
        )

        tissue_values = (
            query_df[tissue_col]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(donor_values) != 1:
            raise ValueError(
                f"Core {core_id!r} contains multiple donor IDs: "
                f"{donor_values.tolist()}"
            )

        if len(tissue_values) != 1:
            raise ValueError(
                f"Core {core_id!r} contains multiple tissue "
                f"annotations: {tissue_values.tolist()}"
            )

        donor_id = donor_values[0]
        tissue = tissue_values[0]

        reference_coordinates = reference_df[
            ["_spatial_x", "_spatial_y"]
        ].to_numpy(dtype=float)

        query_coordinates = query_df[
            ["_spatial_x", "_spatial_y"]
        ].to_numpy(dtype=float)

        reference_ids = (
            reference_df["_cell_id"]
            .astype(str)
            .to_numpy()
        )

        reference_types = (
            reference_df[neighbor_celltype_col]
            .astype(str)
            .to_numpy()
        )

        tree = cKDTree(reference_coordinates)

        # Find every candidate neighbor within the largest radius.
        candidate_lists = tree.query_ball_point(
            query_coordinates,
            r=maximum_radius,
        )

        for query_position, (
            query_index,
            query_row,
        ) in enumerate(query_df.iterrows()):

            query_id = str(query_index)
            query_score = float(
                query_scores.loc[query_index]
            )

            candidate_indices = np.asarray(
                candidate_lists[query_position],
                dtype=int,
            )

            # Remove the focal AM itself, but retain other cells at the
            # same coordinates if they have a different cell ID.
            keep = (
                reference_ids[candidate_indices]
                != query_id
            )

            candidate_indices = candidate_indices[keep]

            if len(candidate_indices):
                candidate_coordinates = (
                    reference_coordinates[
                        candidate_indices
                    ]
                )

                candidate_distances = np.sqrt(
                    np.sum(
                        (
                            candidate_coordinates -
                            query_coordinates[
                                query_position
                            ]
                        ) ** 2,
                        axis=1,
                    )
                )

                candidate_types = reference_types[
                    candidate_indices
                ]

            else:
                candidate_distances = np.array(
                    [],
                    dtype=float,
                )

                candidate_types = np.array(
                    [],
                    dtype=str,
                )

            for radius in radii:

                within_radius = (
                    candidate_distances <= radius
                )

                local_types = candidate_types[
                    within_radius
                ]

                local_distances = candidate_distances[
                    within_radius
                ]

                n_total_neighbors = len(local_types)
                neighborhood_area = np.pi * radius ** 2

                for neighbor_type in target_types:

                    type_mask = (
                        local_types == neighbor_type
                    )

                    n_neighbor = int(
                        np.sum(type_mask)
                    )

                    if n_total_neighbors > 0:
                        neighbor_fraction = (
                            n_neighbor /
                            n_total_neighbors
                        )
                    else:
                        neighbor_fraction = np.nan

                    neighbor_density = (
                        n_neighbor /
                        neighborhood_area
                    )

                    neighbor_presence = float(
                        n_neighbor > 0
                    )

                    if n_neighbor > 0:
                        nearest_type_distance = float(
                            np.min(
                                local_distances[type_mask]
                            )
                        )

                        mean_type_distance = float(
                            np.mean(
                                local_distances[type_mask]
                            )
                        )
                    else:
                        nearest_type_distance = np.nan
                        mean_type_distance = np.nan

                    composition_records.append({
                        "cell_id": query_id,
                        core_col: core_id,
                        donor_col: donor_id,
                        tissue_col: tissue,
                        score_col: query_score,
                        "radius": radius,
                        "neighbor_type": neighbor_type,
                        "n_total_neighbors": (
                            n_total_neighbors
                        ),
                        "n_neighbor": n_neighbor,
                        "neighbor_fraction": (
                            neighbor_fraction
                        ),
                        "neighbor_pct": (
                            100 * neighbor_fraction
                            if np.isfinite(
                                neighbor_fraction
                            )
                            else np.nan
                        ),
                        "neighbor_density": (
                            neighbor_density
                        ),
                        "neighbor_presence": (
                            neighbor_presence
                        ),
                        "nearest_type_distance": (
                            nearest_type_distance
                        ),
                        "mean_type_distance": (
                            mean_type_distance
                        ),
                    })

    neighborhood_comp = pd.DataFrame(
        composition_records
    )

    if neighborhood_comp.empty:
        raise ValueError(
            "No valid query-cell neighborhoods were found. "
            "Check cell-type labels, radii and spatial coordinates."
        )

    # ============================================================
    # Within-core correlations
    # ============================================================

    correlation_records = []

    grouped = neighborhood_comp.groupby(
        [
            core_col,
            "radius",
            "neighbor_type",
        ],
        observed=True,
        dropna=False,
    )

    for (
        core_id,
        radius,
        neighbor_type,
    ), group_df in grouped:

        score_values = pd.to_numeric(
            group_df[score_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        exposure_values = pd.to_numeric(
            group_df[metric_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        valid = (
            np.isfinite(score_values) &
            np.isfinite(exposure_values)
        )

        score_values = score_values[valid]
        exposure_values = exposure_values[valid]

        donor_values = (
            group_df[donor_col]
            .dropna()
            .unique()
        )

        tissue_values = (
            group_df[tissue_col]
            .dropna()
            .unique()
        )

        result = {
            core_col: core_id,
            donor_col: (
                donor_values[0]
                if len(donor_values) == 1
                else pd.NA
            ),
            tissue_col: (
                tissue_values[0]
                if len(tissue_values) == 1
                else pd.NA
            ),
            "radius": radius,
            "neighbor_type": neighbor_type,
            "exposure_metric": exposure_metric,
            "metric_column": metric_col,
            "n_query_cells": len(score_values),
            "rho": np.nan,
            "p_value": np.nan,
        }

        if (
            len(score_values) >= min_query_cells and
            np.nanstd(score_values) > 0 and
            np.nanstd(exposure_values) > 0
        ):
            correlation = spearmanr(
                score_values,
                exposure_values,
                nan_policy="omit",
            )

            result["rho"] = correlation.statistic
            result["p_value"] = correlation.pvalue

        correlation_records.append(result)

    core_correlations = pd.DataFrame(
        correlation_records
    )

    core_correlations["FDR_within_core"] = np.nan

    # Correct across neighboring cell types within each core/radius.
    correction_groups = core_correlations.groupby(
        [core_col, "radius"],
        observed=True,
        dropna=False,
    ).groups

    for _, indices in correction_groups.items():

        indices = pd.Index(indices)

        valid_indices = indices[
            core_correlations.loc[
                indices,
                "p_value",
            ].notna()
        ]

        if len(valid_indices):
            core_correlations.loc[
                valid_indices,
                "FDR_within_core",
            ] = multipletests(
                core_correlations.loc[
                    valid_indices,
                    "p_value",
                ],
                method="fdr_bh",
            )[1]

    # ============================================================
    # Aggregate core correlations within donor and tissue
    # ============================================================

    valid_core_results = core_correlations.dropna(
        subset=["rho"]
    ).copy()

    if valid_core_results.empty:
        donor_correlations = pd.DataFrame()
        tissue_tests = pd.DataFrame()

        return {
            "neighborhood_comp": neighborhood_comp,
            "core_correlations": core_correlations,
            "donor_correlations": donor_correlations,
            "tissue_tests": tissue_tests,
        }

    valid_core_results["fisher_z"] = np.arctanh(
        valid_core_results["rho"].clip(
            -0.999999,
            0.999999,
        )
    )

    donor_correlations = (
        valid_core_results
        .groupby(
            [
                donor_col,
                tissue_col,
                "radius",
                "neighbor_type",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            n_cores=(core_col, "nunique"),
            total_query_cells=(
                "n_query_cells",
                "sum",
            ),
            mean_fisher_z=(
                "fisher_z",
                "mean",
            ),
        )
        .reset_index()
    )

    donor_correlations["mean_rho"] = np.tanh(
        donor_correlations["mean_fisher_z"]
    )

    # ============================================================
    # Tissue-level inference using donors as replicates
    # ============================================================

    tissue_records = []

    tissue_groups = donor_correlations.groupby(
        [
            tissue_col,
            "radius",
            "neighbor_type",
        ],
        observed=True,
        dropna=False,
    )

    for (
        tissue,
        radius,
        neighbor_type,
    ), group_df in tissue_groups:

        donor_values = (
            group_df["mean_fisher_z"]
            .dropna()
            .to_numpy(dtype=float)
        )

        record = {
            tissue_col: tissue,
            "radius": radius,
            "neighbor_type": neighbor_type,
            "exposure_metric": exposure_metric,
            "n_donors": len(donor_values),
            "median_rho": (
                np.tanh(
                    np.median(donor_values)
                )
                if len(donor_values)
                else np.nan
            ),
            "mean_rho": (
                np.tanh(
                    np.mean(donor_values)
                )
                if len(donor_values)
                else np.nan
            ),
            "wilcoxon_statistic": np.nan,
            "p_value": np.nan,
        }

        if (
            len(donor_values) >= 3 and
            not np.allclose(donor_values, 0)
        ):
            try:
                test = wilcoxon(
                    donor_values,
                    alternative=alternative,
                    zero_method="wilcox",
                )

                record["wilcoxon_statistic"] = (
                    test.statistic
                )

                record["p_value"] = test.pvalue

            except ValueError:
                pass

        tissue_records.append(record)

    tissue_tests = pd.DataFrame(
        tissue_records
    )

    tissue_tests["FDR"] = np.nan

    # Correct across cell types separately for every tissue/radius.
    correction_groups = tissue_tests.groupby(
        [tissue_col, "radius"],
        observed=True,
        dropna=False,
    ).groups

    for _, indices in correction_groups.items():

        indices = pd.Index(indices)

        valid_indices = indices[
            tissue_tests.loc[
                indices,
                "p_value",
            ].notna()
        ]

        if len(valid_indices):
            tissue_tests.loc[
                valid_indices,
                "FDR",
            ] = multipletests(
                tissue_tests.loc[
                    valid_indices,
                    "p_value",
                ],
                method="fdr_bh",
            )[1]

    return {
        "neighborhood_comp": neighborhood_comp,
        "core_correlations": core_correlations,
        "donor_correlations": donor_correlations,
        "tissue_tests": tissue_tests,
    }


def plot_radius_core_correlations(
    core_correlations,
    neighbor_types=None,
    top_n=15,
    radii=None,
    core_col="core_id",
    tissue_col="tissue_annotation",
    neighbor_col="neighbor_type",
    value_col="rho",
    tissue_order=("A", "B", "V", "None"),
    tissue_palette=None,
    width_per_panel=4.3,
    row_height=0.38,
    save=None,
    dpi=300,
    random_state=123,
):
    """
    Plot core-level score-versus-neighborhood correlations.

    Each point represents one core.
    Colors represent tissue annotations.
    Black diamonds represent the median across all displayed cores.

    Parameters
    ----------
    neighbor_types
        Specific neighboring cell types to plot.

    top_n
        If neighbor_types is None, select cell types with the largest
        mean absolute core-level correlation.

    radii
        Radii to display. If None, display all available radii.

    core_correlations : pandas.DataFrame
        Core-level correlation table from calculate_radius_niche_continuum.
    core_col, tissue_col, neighbor_col, value_col : str
        Columns defining core, tissue, neighbor type, and correlation value.
    tissue_order : sequence of str
        Tissue display order.
    tissue_palette : mapping or None
        Optional tissue-to-color mapping.
    width_per_panel, row_height : float
        Figure scaling controls.
    save : path-like or None
        Optional output path.
    dpi : int
        Resolution used when saving.
    random_state : int
        Seed for deterministic point jitter.

    Returns
    -------
    tuple
        Matplotlib Figure, Axes array, and filtered plotting table.
    """

    required = {
        core_col,
        tissue_col,
        neighbor_col,
        "radius",
        value_col,
    }

    missing = required.difference(
        core_correlations.columns
    )

    if missing:
        raise KeyError(
            f"Missing columns: {sorted(missing)}"
        )

    plot_df = core_correlations.copy()

    plot_df[value_col] = pd.to_numeric(
        plot_df[value_col],
        errors="coerce",
    )

    plot_df["radius"] = pd.to_numeric(
        plot_df["radius"],
        errors="coerce",
    )

    plot_df = plot_df.dropna(
        subset=[
            value_col,
            "radius",
            neighbor_col,
        ]
    )

    plot_df[tissue_col] = (
        plot_df[tissue_col]
        .astype("string")
        .fillna("None")
    )

    if radii is None:
        radii = sorted(
            plot_df["radius"].unique()
        )
    else:
        radii = [
            float(radius)
            for radius in radii
            if float(radius) in set(
                plot_df["radius"]
            )
        ]

    plot_df = plot_df.loc[
        plot_df["radius"].isin(radii)
    ].copy()

    if plot_df.empty:
        raise ValueError(
            "No valid core correlations remain after filtering."
        )

    # Select cell types
    if neighbor_types is not None:
        neighbor_types = [
            celltype
            for celltype in neighbor_types
            if celltype in plot_df[
                neighbor_col
            ].unique()
        ]

    elif top_n is not None:
        neighbor_types = (
            plot_df.assign(
                absolute_rho=plot_df[value_col].abs()
            )
            .groupby(
                neighbor_col,
                observed=True,
            )["absolute_rho"]
            .mean()
            .nlargest(top_n)
            .index
            .tolist()
        )

    else:
        neighbor_types = list(
            plot_df[neighbor_col].unique()
        )

    if not neighbor_types:
        raise ValueError(
            "None of the requested neighboring cell types "
            "were found."
        )

    plot_df = plot_df.loc[
        plot_df[neighbor_col].isin(
            neighbor_types
        )
    ].copy()

    # Use a consistent cell-type order across all radius panels
    neighbor_order = (
        plot_df.groupby(
            neighbor_col,
            observed=True,
        )[value_col]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    available_tissues = [
        tissue
        for tissue in tissue_order
        if tissue in plot_df[tissue_col].unique()
    ]

    additional_tissues = [
        tissue
        for tissue in plot_df[tissue_col].unique()
        if tissue not in available_tissues
    ]

    available_tissues.extend(
        additional_tissues
    )

    if tissue_palette is None:
        tissue_palette = {
            "A": "#E99A8C",
            "B": "#91B9A5",
            "V": "#5276B5",
            "None": "#B7B7B7",
        }

    fallback_colors = [
        "#D9A5B3",
        "#D5B56E",
        "#8A77B5",
        "#6FB1B8",
    ]

    for index, tissue in enumerate(
        available_tissues
    ):
        if tissue not in tissue_palette:
            tissue_palette[tissue] = (
                fallback_colors[
                    index % len(fallback_colors)
                ]
            )

    n_panels = len(radii)

    figure_height = max(
        4.0,
        row_height * len(neighbor_order) + 1.8,
    )

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            width_per_panel * n_panels,
            figure_height,
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    axes = axes.ravel()

    rng = np.random.default_rng(
        random_state
    )

    y_positions = {
        celltype: index
        for index, celltype in enumerate(
            neighbor_order
        )
    }

    if len(available_tissues) == 1:
        tissue_offsets = {
            available_tissues[0]: 0
        }
    else:
        offsets = np.linspace(
            -0.23,
            0.23,
            len(available_tissues),
        )

        tissue_offsets = dict(
            zip(
                available_tissues,
                offsets,
            )
        )

    maximum_absolute_rho = (
        plot_df[value_col]
        .abs()
        .max()
    )

    x_limit = max(
        0.2,
        np.ceil(
            maximum_absolute_rho * 10
        ) / 10,
    )

    x_limit = min(x_limit, 1.0)

    for ax, radius in zip(
        axes,
        radii,
    ):
        radius_df = plot_df.loc[
            plot_df["radius"] == radius
        ]

        ax.axvline(
            0,
            color="#8F8F8F",
            linewidth=0.9,
            linestyle="--",
            zorder=0,
        )

        for tissue in available_tissues:

            tissue_df = radius_df.loc[
                radius_df[tissue_col] == tissue
            ]

            if tissue_df.empty:
                continue

            x_values = tissue_df[
                value_col
            ].to_numpy()

            base_y = tissue_df[
                neighbor_col
            ].map(y_positions).to_numpy(
                dtype=float
            )

            jitter = rng.normal(
                loc=0,
                scale=0.025,
                size=len(tissue_df),
            )

            y_values = (
                base_y +
                tissue_offsets[tissue] +
                jitter
            )

            ax.scatter(
                x_values,
                y_values,
                s=28,
                color=tissue_palette[tissue],
                edgecolor="black",
                linewidth=0.45,
                alpha=0.82,
                zorder=2,
            )

        # Median across cores
        medians = (
            radius_df.groupby(
                neighbor_col,
                observed=True,
            )[value_col]
            .median()
        )

        for celltype, median_value in medians.items():

            ax.scatter(
                median_value,
                y_positions[celltype],
                marker="D",
                s=38,
                color="#252525",
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )

        ax.set_xlim(
            -x_limit,
            x_limit,
        )

        ax.set_title(
            f"{radius:g} µm",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xlabel(
            "Core-level Spearman ρ"
        )

        ax.set_yticks(
            np.arange(
                len(neighbor_order)
            )
        )

        ax.set_yticklabels(
            neighbor_order,
            fontsize=9,
        )

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(
            axis="y",
            length=0,
        )

    axes[0].invert_yaxis()

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=tissue_palette[tissue],
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=tissue,
        )
        for tissue in available_tissues
    ]

    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markersize=6,
            markerfacecolor="#252525",
            markeredgecolor="white",
            label="Median across cores",
        )
    )

    fig.legend(
        handles=legend_handles,
        title="Tissue annotation",
        frameon=False,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
    )

    fig.suptitle(
        "AM MHCIIhi-score association with local cellular niches",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(
        rect=(0, 0, 0.88, 0.97)
    )

    if save is not None:
        fig.savefig(
            save,
            dpi=dpi,
            bbox_inches="tight",
        )

    return fig, axes, plot_df


def summarize_nhood_by_donor(
    core_results,
    donor_col="donor_id",
    tissue_col="tissue_annotation",
):
    """Summarize core-level neighborhood enrichment within each donor.

    Parameters
    ----------
    core_results : pandas.DataFrame
        Core-level neighborhood-enrichment output.
    donor_col : str, default="donor_id"
        Column identifying the biological replicate.
    tissue_col : str, default="tissue_annotation"
        Column identifying the tissue compartment.

    Returns
    -------
    pandas.DataFrame
        One row per donor, tissue, neighboring cell type, and radius.

    Notes
    -----
    Mean and median z-scores are descriptive across-core summaries rather
    than a formal donor-level meta-analysis.
    """
    donor_summary = (
        core_results
        .groupby(
            [
                donor_col,
                tissue_col,
                "neighbor_celltype",
                "radius",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            n_cores=("core_id", "nunique"),
            total_focus_cells=("n_focus", "sum"),
            total_neighbor_cells=("n_neighbor", "sum"),
            mean_enrichment_zscore=(
                "enrichment_zscore",
                "mean",
            ),
            median_enrichment_zscore=(
                "enrichment_zscore",
                "median",
            ),
        )
        .reset_index()
    )
    return donor_summary


def plot_knn_niche_continuum(
    tissue_tests,
    tissue_order=("A", "B", "V"),
    top_n=None,
    save=None,
    dpi=300,
):
    """Plot donor-level k-nearest-neighbor niche correlations by tissue.

    Parameters
    ----------
    tissue_tests : pandas.DataFrame
        Table containing tissue_annotation, neighbor_type, median_rho, and FDR.
    tissue_order : sequence of str, default=("A", "B", "V")
        Tissue panels and their display order.
    top_n : int or None, default=None
        Optional number of neighbor types ranked by mean absolute correlation.
    save : path-like or None, default=None
        Optional figure output path.
    dpi : int, default=300
        Resolution used when saving.

    Returns
    -------
    tuple
        Matplotlib Figure and flattened array of Axes.
    """
    plot_df = tissue_tests.copy()
    plot_df = plot_df.loc[
        plot_df["tissue_annotation"].isin(tissue_order)
    ].copy()

    if top_n is not None:
        selected = (
            plot_df.assign(abs_rho=plot_df["median_rho"].abs())
            .groupby("neighbor_type", observed=True)["abs_rho"]
            .mean()
            .nlargest(top_n)
            .index
        )
        plot_df = plot_df.loc[plot_df["neighbor_type"].isin(selected)]

    available_tissues = [
        tissue
        for tissue in tissue_order
        if tissue in plot_df["tissue_annotation"].unique()
    ]
    cmap = LinearSegmentedColormap.from_list(
        "macaron_diverging",
        ["#5276B5", "#F6D6A8", "#D95C69"],
    )
    max_abs = max(plot_df["median_rho"].abs().max(), 0.1)
    norm = mpl.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    fig, axes = plt.subplots(
        1,
        len(available_tissues),
        figsize=(
            4.6 * len(available_tissues),
            max(4.5, 0.32 * plot_df["neighbor_type"].nunique()),
        ),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, tissue in zip(axes, available_tissues):
        tissue_df = (
            plot_df.loc[plot_df["tissue_annotation"] == tissue]
            .sort_values("median_rho")
            .reset_index(drop=True)
        )
        y = np.arange(len(tissue_df))
        ax.axvline(0, linestyle="--", linewidth=0.8, color="#999999")
        ax.hlines(
            y=y,
            xmin=0,
            xmax=tissue_df["median_rho"],
            color="#D8D8D8",
            linewidth=1.0,
        )
        ax.scatter(
            tissue_df["median_rho"],
            y,
            c=tissue_df["median_rho"],
            cmap=cmap,
            norm=norm,
            s=48,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        for position, row in tissue_df.iterrows():
            if pd.notna(row["FDR"]):
                if row["FDR"] < 0.001:
                    label = "***"
                elif row["FDR"] < 0.01:
                    label = "**"
                elif row["FDR"] < 0.05:
                    label = "*"
                else:
                    label = ""
                if label:
                    offset = 0.015 * max_abs
                    ax.text(
                        row["median_rho"] + (
                            offset if row["median_rho"] >= 0 else -offset
                        ),
                        position,
                        label,
                        ha=("left" if row["median_rho"] >= 0 else "right"),
                        va="center",
                        fontsize=9,
                    )
        ax.set_yticks(y)
        ax.set_yticklabels(tissue_df["neighbor_type"])
        ax.set_title(f"Tissue {tissue}", fontweight="bold")
        ax.set_xlabel("Median donor Spearman ρ")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=axes.tolist(),
        label="Spearman ρ",
        fraction=0.025,
        pad=0.03,
    )
    fig.suptitle(
        "Cellular niche across the AM MHCIIhi-score continuum",
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(
        left=0.22,
        right=0.91,
        bottom=0.12,
        top=0.88,
        wspace=0.45,
    )
    if save is not None:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig, axes


_STAGE1_BASE_COLUMNS = (
    "method", "direction", "focal_type", "target_type", "scale_type",
    "scale", "core_id", "donor_id", "tissue_annotation", "n_cells",
    "n_focal", "n_target", "observed", "metric", "null_mean", "null_sd",
    "zscore", "effect", "p_association", "p_two_sided",
)
_STAGE1_CONTACT_COLUMNS = _STAGE1_BASE_COLUMNS
_STAGE1_NICHE_COLUMNS = _STAGE1_BASE_COLUMNS + (
    "pct_focal_with_target", "mean_target_neighbors", "mean_total_neighbors",
    "pct_focal_without_neighbors",
)
_STAGE1_DISTANCE_COLUMNS = _STAGE1_BASE_COLUMNS + (
    "mean_distance", "median_distance", "q25_distance", "q75_distance",
)


def _bh_adjust(p_values):
    """Benjamini-Hochberg FDR correction."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)

    valid = np.isfinite(p_values)

    if not valid.any():
        return adjusted

    p = p_values[valid]
    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(
        adjusted_ranked[::-1]
    )[::-1]
    adjusted_ranked = np.minimum(adjusted_ranked, 1)

    adjusted_valid = np.empty(n)
    adjusted_valid[order] = adjusted_ranked
    adjusted[valid] = adjusted_valid

    return adjusted


def _permutation_summary(
    observed,
    null_values,
    smaller_is_association=False,
    pseudocount=1e-6,
):
    """
    Summarize an observed statistic against a permutation null.

    The returned effect is oriented so:
        effect > 0 = stronger association than expected.
    """
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]

    if len(null_values) == 0:
        return {
            "null_mean": np.nan,
            "null_sd": np.nan,
            "zscore": np.nan,
            "effect": np.nan,
            "p_association": np.nan,
            "p_two_sided": np.nan,
        }

    null_mean = float(np.mean(null_values))
    null_sd = (
        float(np.std(null_values, ddof=1))
        if len(null_values) > 1
        else np.nan
    )

    if smaller_is_association:
        # Smaller observed distance means stronger association.
        difference = null_mean - observed

        p_association = (
            1 + np.sum(null_values <= observed)
        ) / (
            len(null_values) + 1
        )

        effect = np.log2(
            (null_mean + pseudocount) /
            (observed + pseudocount)
        )

    else:
        # Larger contact/fraction means stronger association.
        difference = observed - null_mean

        p_association = (
            1 + np.sum(null_values >= observed)
        ) / (
            len(null_values) + 1
        )

        effect = np.log2(
            (observed + pseudocount) /
            (null_mean + pseudocount)
        )

    if np.isfinite(null_sd) and null_sd > 0:
        zscore = difference / null_sd
    else:
        zscore = np.nan

    p_two_sided = (
        1 +
        np.sum(
            np.abs(null_values - null_mean) >=
            abs(observed - null_mean)
        )
    ) / (
        len(null_values) + 1
    )

    return {
        "null_mean": null_mean,
        "null_sd": null_sd,
        "zscore": zscore,
        "effect": float(effect),
        "p_association": float(p_association),
        "p_two_sided": float(p_two_sided),
    }


def _prepare_spatial_celltypes(
    adata,
    celltype_col="CellType_refined",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    coordinate_scale=1.0,
    rename_celltypes=None,
    exclude_celltypes=(
        "Mixed",
        "Ambiguous",
        "Unknown",
        "Doublet",
        "Unassigned",
    ),
):
    """
    Extract spatial coordinates and cell annotations.

    coordinate_scale:
        Multiplier converting coordinates into micrometres.
        Use 1.0 if Xenium coordinates are already in µm.
    """
    required = [celltype_col, core_col, donor_col, tissue_col]

    missing = [
        column for column in required
        if column not in adata.obs.columns
    ]

    if missing:
        raise KeyError(
            f"Missing adata.obs columns: {missing}"
        )

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"{spatial_key!r} is absent from adata.obsm."
        )

    coordinates = np.asarray(
        adata.obsm[spatial_key],
        dtype=float,
    )

    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must contain at least "
            "two coordinate columns."
        )

    if coordinates.shape[0] != adata.n_obs:
        raise ValueError(
            "Spatial coordinates do not match adata.n_obs."
        )

    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be finite and greater than zero.")

    obs = adata.obs

    data = pd.DataFrame(
        {
            "cell_id": obs.index.astype(str),
            "celltype": (
                obs[celltype_col]
                .astype("object")
                .to_numpy()
            ),
            "core_id": (
                obs[core_col]
                .astype("object")
                .to_numpy()
            ),
            "x": coordinates[:, 0] * coordinate_scale,
            "y": coordinates[:, 1] * coordinate_scale,
        }
    )

    data["donor_id"] = (
        obs[donor_col]
        .astype("object")
        .to_numpy()
    )
    data["tissue_annotation"] = (
        obs[tissue_col]
        .astype("object")
        .to_numpy()
    )

    if rename_celltypes is not None:
        data["celltype"] = data["celltype"].replace(
            rename_celltypes
        )

    valid = (
        data["celltype"].notna() &
        data["core_id"].notna() &
        np.isfinite(data["x"]) &
        np.isfinite(data["y"])
    )

    data = data.loc[valid].copy()

    data["celltype"] = data["celltype"].astype(str)
    if exclude_celltypes is None:
        data["_stage1_eligible"] = True
    else:
        data["_stage1_eligible"] = ~data["celltype"].isin(
            set(exclude_celltypes)
        )

    return data.reset_index(drop=True)


def _resolve_target_map(
    observed_celltypes,
    focal_types,
    target_types=None,
):
    """
    Resolve which targets should be evaluated for each focal type.

    target_types can be:
      - None: every other observed cell type
      - list/tuple: same target list for every focal type
      - dict: separate target list for each focal type
    """
    observed_celltypes = list(observed_celltypes)
    observed_set = set(observed_celltypes)
    focal_types = list(focal_types)

    if target_types is None:
        return {
            focal: [
                target
                for target in observed_celltypes
                if target != focal and target in observed_set
            ]
            for focal in focal_types
        }

    if isinstance(target_types, dict):
        return {
            focal: [
                target
                for target in target_types.get(focal, [])
                if target != focal and target in observed_set
            ]
            for focal in focal_types
        }

    targets = list(target_types)

    return {
        focal: [
            target
            for target in targets
            if target != focal and target in observed_set
        ]
        for focal in focal_types
    }


def _eligible_focal_types(focal_types, observed_celltypes):
    """Return unique requested focal types eligible for hypothesis testing."""
    if isinstance(focal_types, str):
        focal_types = [focal_types]
    observed = set(observed_celltypes)
    return [
        focal
        for focal in dict.fromkeys(focal_types)
        if focal in observed
    ]


def _get_core_metadata(core_data):
    """Return donor and tissue annotation for one core."""
    core_id = core_data["core_id"].iloc[0]
    if core_data["donor_id"].isna().any():
        raise ValueError(
            f"Core {core_id!r} has missing donor annotations."
        )
    if core_data["tissue_annotation"].isna().any():
        raise ValueError(
            f"Core {core_id!r} has missing tissue annotations."
        )
    donor_values = (
        core_data["donor_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    tissue_values = (
        core_data["tissue_annotation"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(donor_values) > 1:
        raise ValueError(
            f"Core {core_id!r} contains multiple donor annotations: "
            f"{sorted(donor_values)}"
        )
    if len(tissue_values) > 1:
        raise ValueError(
            f"Core {core_id!r} contains multiple tissue annotations: "
            f"{sorted(tissue_values)}"
        )

    donor_id = donor_values[0] if len(donor_values) == 1 else pd.NA
    tissue_annotation = (
        tissue_values[0] if len(tissue_values) == 1 else pd.NA
    )

    return donor_id, tissue_annotation


def _add_within_core_fdr(results):
    """
    Correct target-cell P values within each core, focal cell type,
    method and spatial scale.

    This is primarily for Stage 1A discovery.
    """
    results = results.copy()

    if results.empty:
        results["FDR_within_core"] = pd.Series(dtype=float)
        return results

    results["FDR_within_core"] = np.nan
    results["_fdr_focal_family"] = results["focal_type"].astype("object")
    results.loc[
        results["method"].eq("Contact enrichment"),
        "_fdr_focal_family",
    ] = "__all_undirected_contact_pairs__"

    group_columns = [
        "method",
        "core_id",
        "_fdr_focal_family",
        "scale_type",
        "scale",
    ]

    for _, indices in results.groupby(
        group_columns,
        dropna=False,
        observed=True,
    ).groups.items():

        indices = list(indices)

        results.loc[indices, "FDR_within_core"] = (
            _bh_adjust(
                results.loc[
                    indices,
                    "p_association",
                ].to_numpy()
            )
        )

    return results.drop(columns="_fdr_focal_family")


def _make_query_adjacency(
    coordinates,
    query_indices,
    mode,
    scale,
):
    """
    Construct a query-cell × all-cell sparse adjacency matrix.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    query_indices = np.asarray(query_indices, dtype=int)

    n_cells = len(coordinates)
    n_query = len(query_indices)

    if n_query == 0 or n_cells < 2:
        return csr_matrix(
            (n_query, n_cells),
            dtype=np.int64,
        )

    tree = cKDTree(coordinates)

    rows = []
    columns = []

    if mode == "knn":
        k = min(int(scale), n_cells - 1)

        _, neighbors = tree.query(
            coordinates[query_indices],
            k=k + 1,
        )

        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]

        for row, cell_index in enumerate(query_indices):
            selected = neighbors[row]
            selected = selected[selected != cell_index]
            selected = selected[:k]

            rows.extend([row] * len(selected))
            columns.extend(selected.tolist())

    elif mode == "radius":
        neighborhoods = tree.query_ball_point(
            coordinates[query_indices],
            r=float(scale),
        )

        for row, cell_index in enumerate(query_indices):
            selected = [
                index
                for index in neighborhoods[row]
                if index != cell_index
            ]

            rows.extend([row] * len(selected))
            columns.extend(selected)

    else:
        raise ValueError(
            "mode must be 'knn' or 'radius'."
        )

    return csr_matrix(
        (
            np.ones(len(rows), dtype=np.int64),
            (
                np.asarray(rows, dtype=int),
                np.asarray(columns, dtype=int),
            ),
        ),
        shape=(n_query, n_cells),
    )


def _validate_stage1_parameters(
    scales,
    scale_name,
    mode,
    n_permutations,
    min_focal_cells,
    min_target_cells,
):
    """Validate common multitype Stage 1 analysis parameters."""
    if (
        not isinstance(n_permutations, (int, np.integer))
        or isinstance(n_permutations, bool)
        or n_permutations < 1
    ):
        raise ValueError("n_permutations must be a positive integer.")
    for name, value in (
        ("min_focal_cells", min_focal_cells),
        ("min_target_cells", min_target_cells),
    ):
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")

    values = list(np.atleast_1d(scales))
    if not values:
        raise ValueError(f"{scale_name} must contain at least one value.")
    if mode == "knn":
        valid = all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, bool)
            and value > 0
            for value in values
        )
    else:
        valid = all(
            np.isscalar(value)
            and np.isfinite(value)
            and float(value) > 0
            for value in values
        )
    if not valid:
        requirement = (
            "positive integers" if mode == "knn" else "finite positive values"
        )
        raise ValueError(f"{scale_name} must contain only {requirement}.")
    return values


def calculate_multitype_nhood_enrichment_by_core(
    adata,
    focal_types=("AM", "AT2"),
    target_types=None,
    radii=(25, 50, 100),
    celltype_col="CellType_refined",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    coordinate_scale=1.0,
    rename_celltypes=None,
    exclude_celltypes=(
        "Mixed",
        "Ambiguous",
        "Unknown",
        "Doublet",
        "Unassigned",
    ),
    min_focal_cells=10,
    min_target_cells=10,
    n_permutations=199,
    random_state=123,
):
    """Test whether focal-target contact pairs exceed a within-core null.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-level object containing annotations and spatial coordinates.
    focal_types : sequence of str, default=("AM", "AT2")
        Cell types whose pairwise contacts are tested.
    target_types : sequence, mapping, or None, default=None
        Shared targets, focal-specific targets, or every other eligible type.
    radii : sequence of float, default=(25, 50, 100)
        Positive contact radii in the scaled coordinate units.
    celltype_col, core_col, donor_col, tissue_col : str
        Observation columns defining cell type, core, donor, and tissue.
    spatial_key : str, default="spatial"
        ``adata.obsm`` key containing at least two coordinate columns.
    coordinate_scale : float, default=1.0
        Positive multiplier converting coordinates to the desired units.
    rename_celltypes : mapping or None, default=None
        Optional cell-type relabeling applied before analysis.
    exclude_celltypes : sequence or None
        Types excluded as focal/target hypotheses but retained as spatial
        background cells when constructing neighborhoods and nulls.
    min_focal_cells, min_target_cells : int, default=10
        Minimum per-core counts required for a focal-target test.
    n_permutations : int, default=199
        Number of within-core label permutations.
    random_state : int, default=123
        Seed for reproducible permutations.

    Returns
    -------
    pandas.DataFrame
        Core-level contact counts, null summaries, oriented log2 effects,
        one- and two-sided permutation P values, and within-core FDR.

    Notes
    -----
    Contact pairs are undirected, so reversed focal-target duplicates are
    removed. The null permutes all cell-type labels across positions within
    each core, preserving abundance but not native spatial autocorrelation.
    """
    radii = _validate_stage1_parameters(
        radii,
        "radii",
        "radius",
        n_permutations,
        min_focal_cells,
        min_target_cells,
    )
    data = _prepare_spatial_celltypes(
        adata=adata,
        celltype_col=celltype_col,
        core_col=core_col,
        donor_col=donor_col,
        tissue_col=tissue_col,
        spatial_key=spatial_key,
        coordinate_scale=coordinate_scale,
        rename_celltypes=rename_celltypes,
        exclude_celltypes=exclude_celltypes,
    )

    observed_types = sorted(
        data.loc[data["_stage1_eligible"], "celltype"].unique()
    )
    focal_types = _eligible_focal_types(focal_types, observed_types)

    target_map = _resolve_target_map(
        observed_celltypes=observed_types,
        focal_types=focal_types,
        target_types=target_types,
    )

    rng = np.random.default_rng(random_state)
    records = []

    for core_id, core_data in data.groupby(
        "core_id",
        sort=False,
        observed=True,
    ):
        core_data = core_data.reset_index(drop=True)

        labels = core_data["celltype"].to_numpy()
        coordinates = core_data[["x", "y"]].to_numpy()

        donor_id, tissue_annotation = _get_core_metadata(
            core_data
        )

        tree = cKDTree(coordinates)

        eligible_pairs = []
        seen_unordered_pairs = set()

        for focal_type in focal_types:
            n_focal = int(np.sum(labels == focal_type))

            if n_focal < min_focal_cells:
                continue

            for target_type in target_map[focal_type]:
                n_target = int(np.sum(labels == target_type))

                if n_target < min_target_cells:
                    continue

                unordered_pair = tuple(sorted((focal_type, target_type)))
                if unordered_pair in seen_unordered_pairs:
                    continue
                seen_unordered_pairs.add(unordered_pair)
                canonical_focal, canonical_target = unordered_pair
                eligible_pairs.append(
                    (
                        canonical_focal,
                        canonical_target,
                        int(np.sum(labels == canonical_focal)),
                        int(np.sum(labels == canonical_target)),
                    )
                )

        if not eligible_pairs:
            continue

        eligible_pairs.sort(key=lambda values: (values[0], values[1]))

        for radius in radii:
            pairs = tree.query_pairs(
                r=float(radius),
                output_type="ndarray",
            )

            if pairs.size == 0:
                pairs = np.empty((0, 2), dtype=int)

            left = pairs[:, 0]
            right = pairs[:, 1]
            permutation_indices = np.vstack(
                [
                    rng.permutation(len(labels))
                    for _ in range(n_permutations)
                ]
            )

            for (
                focal_type,
                target_type,
                n_focal,
                n_target,
            ) in eligible_pairs:

                observed_contacts = np.sum(
                    (
                        (labels[left] == focal_type) &
                        (labels[right] == target_type)
                    ) |
                    (
                        (labels[left] == target_type) &
                        (labels[right] == focal_type)
                    )
                )

                null_values = np.empty(
                    n_permutations,
                    dtype=float,
                )

                for permutation in range(n_permutations):
                    permuted = labels[permutation_indices[permutation]]

                    null_values[permutation] = np.sum(
                        (
                            (permuted[left] == focal_type) &
                            (permuted[right] == target_type)
                        ) |
                        (
                            (permuted[left] == target_type) &
                            (permuted[right] == focal_type)
                        )
                    )

                summary = _permutation_summary(
                    observed=float(observed_contacts),
                    null_values=null_values,
                    smaller_is_association=False,
                    pseudocount=0.5,
                )

                records.append(
                    {
                        "method": "Contact enrichment",
                        "direction": (
                            f"{focal_type} ↔ {target_type}"
                        ),
                        "focal_type": focal_type,
                        "target_type": target_type,
                        "scale_type": "radius_um",
                        "scale": float(radius),
                        "core_id": core_id,
                        "donor_id": donor_id,
                        "tissue_annotation": tissue_annotation,
                        "n_cells": len(core_data),
                        "n_focal": n_focal,
                        "n_target": n_target,
                        "observed": float(
                            observed_contacts
                        ),
                        "metric": "Focal–target contact pairs",
                        **summary,
                    }
                )

    return _add_within_core_fdr(
        pd.DataFrame.from_records(records, columns=_STAGE1_CONTACT_COLUMNS)
    )



def _calculate_multitype_local_niche_by_core(
    adata,
    mode,
    scales,
    focal_types=("AM", "AT2"),
    target_types=None,
    celltype_col="CellType_refined",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    coordinate_scale=1.0,
    rename_celltypes=None,
    exclude_celltypes=(
        "Mixed",
        "Ambiguous",
        "Unknown",
        "Doublet",
        "Unassigned",
    ),
    min_focal_cells=10,
    min_target_cells=10,
    n_permutations=199,
    random_state=123,
):
    """
    Directional focal-cell neighborhood analysis.

    For every focal cell, calculate the fraction of its neighborhood
    occupied by a selected target cell type.

    Null:
        Focal positions remain fixed. Target labels are randomized
        among non-focal cells in the same core while preserving the
        number of target cells.
    """
    scale_name = "k_values" if mode == "knn" else "radii"
    scales = _validate_stage1_parameters(
        scales,
        scale_name,
        mode,
        n_permutations,
        min_focal_cells,
        min_target_cells,
    )
    data = _prepare_spatial_celltypes(
        adata=adata,
        celltype_col=celltype_col,
        core_col=core_col,
        donor_col=donor_col,
        tissue_col=tissue_col,
        spatial_key=spatial_key,
        coordinate_scale=coordinate_scale,
        rename_celltypes=rename_celltypes,
        exclude_celltypes=exclude_celltypes,
    )

    observed_types = sorted(
        data.loc[data["_stage1_eligible"], "celltype"].unique()
    )
    focal_types = _eligible_focal_types(focal_types, observed_types)

    target_map = _resolve_target_map(
        observed_celltypes=observed_types,
        focal_types=focal_types,
        target_types=target_types,
    )

    method_name = (
        "kNN niche enrichment"
        if mode == "knn"
        else "Fixed-radius niche enrichment"
    )

    scale_type = (
        "k_neighbors"
        if mode == "knn"
        else "radius_um"
    )

    rng = np.random.default_rng(random_state)
    records = []

    for core_id, core_data in data.groupby(
        "core_id",
        sort=False,
        observed=True,
    ):
        core_data = core_data.reset_index(drop=True)

        labels = core_data["celltype"].to_numpy()
        coordinates = core_data[["x", "y"]].to_numpy()

        donor_id, tissue_annotation = _get_core_metadata(
            core_data
        )

        for focal_type in focal_types:
            query_indices = np.flatnonzero(
                labels == focal_type
            )

            n_focal = len(query_indices)

            if n_focal < min_focal_cells:
                continue

            candidate_indices = np.flatnonzero(
                labels != focal_type
            )

            for scale in scales:
                adjacency = _make_query_adjacency(
                    coordinates=coordinates,
                    query_indices=query_indices,
                    mode=mode,
                    scale=scale,
                )

                total_neighbors = np.asarray(
                    adjacency.sum(axis=1)
                ).ravel()

                for target_type in target_map[focal_type]:
                    target_mask = (
                        labels == target_type
                    ).astype(np.int64)

                    n_target = int(target_mask.sum())

                    if n_target < min_target_cells:
                        continue

                    observed_counts = np.asarray(
                        adjacency @ target_mask
                    ).ravel()

                    observed_fractions = np.divide(
                        observed_counts,
                        total_neighbors,
                        out=np.zeros(
                            n_focal,
                            dtype=float,
                        ),
                        where=total_neighbors > 0,
                    )

                    observed = float(
                        np.mean(observed_fractions)
                    )

                    null_values = np.empty(
                        n_permutations,
                        dtype=float,
                    )

                    for permutation in range(n_permutations):
                        selected = rng.choice(
                            candidate_indices,
                            size=n_target,
                            replace=False,
                        )

                        permuted_target = np.zeros(
                            len(core_data),
                            dtype=np.int64,
                        )
                        permuted_target[selected] = 1

                        permuted_counts = np.asarray(
                            adjacency @ permuted_target
                        ).ravel()

                        permuted_fractions = np.divide(
                            permuted_counts,
                            total_neighbors,
                            out=np.zeros(
                                n_focal,
                                dtype=float,
                            ),
                            where=total_neighbors > 0,
                        )

                        null_values[permutation] = np.mean(
                            permuted_fractions
                        )

                    summary = _permutation_summary(
                        observed=observed,
                        null_values=null_values,
                        smaller_is_association=False,
                    )

                    records.append(
                        {
                            "method": method_name,
                            "direction": (
                                f"{focal_type} → "
                                f"{target_type}"
                            ),
                            "focal_type": focal_type,
                            "target_type": target_type,
                            "scale_type": scale_type,
                            "scale": float(scale),
                            "core_id": core_id,
                            "donor_id": donor_id,
                            "tissue_annotation": (
                                tissue_annotation
                            ),
                            "n_cells": len(core_data),
                            "n_focal": n_focal,
                            "n_target": n_target,
                            "observed": observed,
                            "metric": (
                                "Mean target fraction "
                                "per focal neighborhood"
                            ),
                            "pct_focal_with_target": (
                                100 *
                                np.mean(observed_counts > 0)
                            ),
                            "mean_target_neighbors": float(
                                np.mean(observed_counts)
                            ),
                            "mean_total_neighbors": float(
                                np.mean(total_neighbors)
                            ),
                            "pct_focal_without_neighbors": (
                                100 *
                                np.mean(total_neighbors == 0)
                            ),
                            **summary,
                        }
                    )

    return _add_within_core_fdr(
        pd.DataFrame.from_records(records, columns=_STAGE1_NICHE_COLUMNS)
    )



def calculate_multitype_nearest_distance_by_core(
    adata,
    focal_types=("AM", "AT2"),
    target_types=None,
    celltype_col="CellType_refined",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    coordinate_scale=1.0,
    rename_celltypes=None,
    exclude_celltypes=(
        "Mixed",
        "Ambiguous",
        "Unknown",
        "Doublet",
        "Unassigned",
    ),
    min_focal_cells=10,
    min_target_cells=10,
    n_permutations=199,
    random_state=123,
):
    """Test focal-to-target nearest distances against a within-core null.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-level object containing annotations and spatial coordinates.
    focal_types : sequence of str, default=("AM", "AT2")
        Cell types used as directional focal populations.
    target_types : sequence, mapping, or None, default=None
        Shared targets, focal-specific targets, or every other eligible type.
    celltype_col, core_col, donor_col, tissue_col : str
        Observation columns defining cell type, core, donor, and tissue.
    spatial_key : str, default="spatial"
        ``adata.obsm`` key containing at least two coordinate columns.
    coordinate_scale : float, default=1.0
        Positive multiplier converting coordinates to the desired units.
    rename_celltypes : mapping or None, default=None
        Optional cell-type relabeling applied before analysis.
    exclude_celltypes : sequence or None
        Types excluded from focal/target hypotheses but retained as spatial
        background cells.
    min_focal_cells, min_target_cells : int, default=10
        Minimum per-core counts required for a directional test.
    n_permutations : int, default=199
        Number of target-label randomizations within each core.
    random_state : int, default=123
        Seed for reproducible permutations.

    Returns
    -------
    pandas.DataFrame
        Directional core-level distance summaries, null statistics, oriented
        log2 effects, permutation P values, and within-core FDR.

    Notes
    -----
    Positive effect means shorter distance than expected. Focal positions stay
    fixed while target labels are sampled from non-focal cells in the core.
    """
    _validate_stage1_parameters(
        (1.0,),
        "nearest_target_scale",
        "radius",
        n_permutations,
        min_focal_cells,
        min_target_cells,
    )
    data = _prepare_spatial_celltypes(
        adata=adata,
        celltype_col=celltype_col,
        core_col=core_col,
        donor_col=donor_col,
        tissue_col=tissue_col,
        spatial_key=spatial_key,
        coordinate_scale=coordinate_scale,
        rename_celltypes=rename_celltypes,
        exclude_celltypes=exclude_celltypes,
    )

    observed_types = sorted(
        data.loc[data["_stage1_eligible"], "celltype"].unique()
    )
    focal_types = _eligible_focal_types(focal_types, observed_types)

    target_map = _resolve_target_map(
        observed_celltypes=observed_types,
        focal_types=focal_types,
        target_types=target_types,
    )

    rng = np.random.default_rng(random_state)
    records = []

    for core_id, core_data in data.groupby(
        "core_id",
        sort=False,
        observed=True,
    ):
        core_data = core_data.reset_index(drop=True)

        labels = core_data["celltype"].to_numpy()
        coordinates = core_data[["x", "y"]].to_numpy()

        donor_id, tissue_annotation = _get_core_metadata(
            core_data
        )

        for focal_type in focal_types:
            focal_indices = np.flatnonzero(
                labels == focal_type
            )

            n_focal = len(focal_indices)

            if n_focal < min_focal_cells:
                continue

            focal_coordinates = coordinates[focal_indices]

            candidate_indices = np.flatnonzero(
                labels != focal_type
            )

            for target_type in target_map[focal_type]:
                target_indices = np.flatnonzero(
                    labels == target_type
                )

                n_target = len(target_indices)

                if n_target < min_target_cells:
                    continue

                target_tree = cKDTree(
                    coordinates[target_indices]
                )

                observed_distances, _ = target_tree.query(
                    focal_coordinates,
                    k=1,
                )

                observed = float(
                    np.median(observed_distances)
                )

                null_values = np.empty(
                    n_permutations,
                    dtype=float,
                )

                for permutation in range(n_permutations):
                    selected = rng.choice(
                        candidate_indices,
                        size=n_target,
                        replace=False,
                    )

                    permuted_tree = cKDTree(
                        coordinates[selected]
                    )

                    permuted_distances, _ = (
                        permuted_tree.query(
                            focal_coordinates,
                            k=1,
                        )
                    )

                    null_values[permutation] = np.median(
                        permuted_distances
                    )

                summary = _permutation_summary(
                    observed=observed,
                    null_values=null_values,
                    smaller_is_association=True,
                )

                records.append(
                    {
                        "method": "Nearest-target distance",
                        "direction": (
                            f"{focal_type} → {target_type}"
                        ),
                        "focal_type": focal_type,
                        "target_type": target_type,
                        "scale_type": "nearest_target",
                        "scale": 1.0,
                        "core_id": core_id,
                        "donor_id": donor_id,
                        "tissue_annotation": (
                            tissue_annotation
                        ),
                        "n_cells": len(core_data),
                        "n_focal": n_focal,
                        "n_target": n_target,
                        "observed": observed,
                        "metric": (
                            "Median nearest-target distance"
                        ),
                        "mean_distance": float(
                            np.mean(observed_distances)
                        ),
                        "median_distance": float(
                            np.median(observed_distances)
                        ),
                        "q25_distance": float(
                            np.quantile(
                                observed_distances,
                                0.25,
                            )
                        ),
                        "q75_distance": float(
                            np.quantile(
                                observed_distances,
                                0.75,
                            )
                        ),
                        **summary,
                    }
                )

    return _add_within_core_fdr(
        pd.DataFrame.from_records(records, columns=_STAGE1_DISTANCE_COLUMNS)
    )



def calculate_multitype_knn_niche_by_core(
    adata,
    k_values=(5, 15, 30),
    **kwargs,
):
    """Calculate directional multitype enrichment in k-nearest niches.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-level object with cell-type, core, and spatial annotations.
    k_values : sequence of int, default=(5, 15, 30)
        Positive numbers of non-self nearest neighbors.
    **kwargs
        Additional arguments forwarded to the shared multitype niche engine,
        including focal/target types, metadata keys, minimum cell counts,
        permutation count, and random seed.

    Returns
    -------
    pandas.DataFrame
        One row per core, focal type, target type, and k value, with observed
        target fractions, permutation effects, P values, and within-core FDR.

    Notes
    -----
    Query positions remain fixed while target labels are randomized among
    non-focal cells within the same core.
    """
    return _calculate_multitype_local_niche_by_core(
        adata=adata,
        mode="knn",
        scales=k_values,
        **kwargs,
    )


def calculate_multitype_radius_niche_by_core(
    adata,
    radii=(25, 50, 100),
    **kwargs,
):
    """Calculate directional multitype enrichment in fixed-radius niches.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-level object with cell-type, core, and spatial annotations.
    radii : sequence of float, default=(25, 50, 100)
        Positive physical radii in the scaled coordinate units.
    **kwargs
        Additional arguments forwarded to the shared multitype niche engine,
        including focal/target types, metadata keys, minimum cell counts,
        permutation count, and random seed.

    Returns
    -------
    pandas.DataFrame
        One row per core, focal type, target type, and radius, with observed
        target fractions, permutation effects, P values, and within-core FDR.

    Notes
    -----
    Focal cells with no neighbors at a radius contribute a zero target
    fraction and are reported through pct_focal_without_neighbors.
    """
    return _calculate_multitype_local_niche_by_core(
        adata=adata,
        mode="radius",
        scales=radii,
        **kwargs,
    )


def summarize_stage1_by_donor_and_tissue(
    core_results,
    min_donors=3,
):
    """Aggregate Stage 1 core effects and test donor effects within tissue.

    Parameters
    ----------
    core_results : pandas.DataFrame
        Core-level output from a multitype Stage 1 spatial function. Required
        columns identify method, direction, focal/target types, scale, core,
        donor, tissue, effect, z-score, and focal/target cell counts.
    min_donors : int, default=3
        Minimum number of non-missing donor effects required to run the
        one-sided Wilcoxon signed-rank test.

    Returns
    -------
    donor_summary : pandas.DataFrame
        Mean core effect and z-score for each donor, tissue, method, direction,
        focal/target pair, and scale.
    tissue_tests : pandas.DataFrame
        One-sided donor-level Wilcoxon tests of positive association, with
        Benjamini-Hochberg correction across target types within each focal
        type, method, tissue, and spatial scale.

    Notes
    -----
    Donors are the inferential replicates. Tests with fewer than five donors
    are retained for transparency but have very limited exact-test resolution.
    """
    required_columns = {
        "method", "direction", "focal_type", "target_type", "scale_type",
        "scale", "core_id", "donor_id", "tissue_annotation", "effect",
        "zscore", "n_focal", "n_target",
    }
    missing = required_columns.difference(core_results.columns)
    if missing:
        raise KeyError(
            "core_results is missing required columns: "
            f"{sorted(missing)}"
        )
    if (
        not isinstance(min_donors, (int, np.integer))
        or isinstance(min_donors, bool)
        or min_donors < 1
    ):
        raise ValueError("min_donors must be a positive integer.")

    grouping_columns = [
        "method", "direction", "focal_type", "target_type", "scale_type",
        "scale", "donor_id", "tissue_annotation",
    ]
    donor_summary = (
        core_results
        .groupby(grouping_columns, observed=True, dropna=False)
        .agg(
            donor_effect=("effect", "mean"),
            donor_zscore=("zscore", "mean"),
            n_cores=("core_id", "nunique"),
            mean_n_focal=("n_focal", "mean"),
            mean_n_target=("n_target", "mean"),
        )
        .reset_index()
    )

    test_columns = [
        "method", "direction", "focal_type", "target_type", "scale_type",
        "scale", "tissue_annotation",
    ]
    test_records = []
    for keys, group in donor_summary.groupby(
        test_columns,
        observed=True,
        dropna=False,
    ):
        values = group["donor_effect"].dropna().to_numpy(dtype=float)
        n_donors = len(values)
        statistic = np.nan
        p_value = np.nan
        if n_donors >= min_donors and np.any(values != 0):
            result = wilcoxon(
                values,
                alternative="greater",
                zero_method="wilcox",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        record = dict(zip(test_columns, keys))
        record.update(
            {
                "n_donors": n_donors,
                "mean_effect": (
                    float(np.mean(values)) if n_donors else np.nan
                ),
                "median_effect": (
                    float(np.median(values)) if n_donors else np.nan
                ),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
        test_records.append(record)

    tissue_tests = pd.DataFrame(test_records)
    if not tissue_tests.empty:
        tissue_tests["FDR"] = np.nan
        tissue_tests["_fdr_focal_family"] = (
            tissue_tests["focal_type"].astype("object")
        )
        tissue_tests.loc[
            tissue_tests["method"].eq("Contact enrichment"),
            "_fdr_focal_family",
        ] = "__all_undirected_contact_pairs__"
        fdr_groups = [
            "method", "_fdr_focal_family", "scale_type", "scale",
            "tissue_annotation",
        ]
        for _, indices in tissue_tests.groupby(
            fdr_groups,
            observed=True,
            dropna=False,
        ).groups.items():
            indices = list(indices)
            tissue_tests.loc[indices, "FDR"] = _bh_adjust(
                tissue_tests.loc[indices, "p_value"].to_numpy()
            )
        tissue_tests = tissue_tests.drop(columns="_fdr_focal_family")
    else:
        tissue_tests = pd.DataFrame(
            columns=test_columns + [
                "n_donors", "mean_effect", "median_effect",
                "wilcoxon_statistic", "p_value", "FDR",
            ]
        )

    return donor_summary, tissue_tests


_STAGE2_COMMON_COLUMNS = (
    "method", "direction", "scale_type", "scale", "effect_type",
    "core_id", "donor_id", "tissue_annotation", "n_cells", "n_am",
    "n_at2", "n_am_analyzed", "n_am_without_neighbors", "observed",
    "null_mean", "null_sd", "zscore", "effect", "p_association",
    "p_two_sided",
)
_STAGE2_EXPOSURE_COLUMNS = _STAGE2_COMMON_COLUMNS + (
    "rho_score_at2_fraction", "mean_at2_fraction", "pct_am_with_at2",
    "mean_at2_neighbors", "mean_total_neighbors", "pct_am_without_neighbors",
)
_STAGE2_NEAREST_COLUMNS = _STAGE2_COMMON_COLUMNS + (
    "rho_score_distance_raw", "mean_nearest_distance",
    "median_nearest_distance", "q25_nearest_distance", "q75_nearest_distance",
)
_STAGE2_EXTREME_COLUMNS = _STAGE2_COMMON_COLUMNS + (
    "n_high", "n_low", "mean_high_at2_fraction", "mean_low_at2_fraction",
    "mean_high_score", "mean_low_score", "pct_am_without_neighbors",
)


def _empty_frame(columns):
    """Return an empty result table with a stable ordered schema."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _validate_positive_integer(value, name):
    """Reject booleans and nonpositive or nonintegral count parameters."""
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")


def _safe_spearman(x, y, min_cells=5):
    """Calculate Spearman correlation after finite-value filtering."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < min_cells or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def _signed_permutation_summary(observed, null_values):
    """Summarize a signed statistic without applying a ratio transformation."""
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0:
        return {
            "null_mean": np.nan,
            "null_sd": np.nan,
            "zscore": np.nan,
            "effect": float(observed),
            "p_association": np.nan,
            "p_two_sided": np.nan,
        }
    null_mean = float(np.mean(null_values))
    null_sd = (
        float(np.std(null_values, ddof=1)) if len(null_values) > 1 else np.nan
    )
    difference = float(observed) - null_mean
    zscore = (
        difference / null_sd
        if np.isfinite(null_sd) and null_sd > 0
        else np.nan
    )
    p_association = (
        1 + np.sum(null_values >= observed)
    ) / (len(null_values) + 1)
    p_two_sided = (
        1
        + np.sum(
            np.abs(null_values - null_mean)
            >= abs(float(observed) - null_mean)
        )
    ) / (len(null_values) + 1)
    return {
        "null_mean": null_mean,
        "null_sd": null_sd,
        "zscore": float(zscore),
        "effect": float(observed),
        "p_association": float(p_association),
        "p_two_sided": float(p_two_sided),
    }


def _prepare_stage2_data(
    adata,
    score_col="MHCIIhi_score",
    celltype_col="CellType_refined",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    am_labels=("AM", "Alveolar Macrophage"),
    at2_labels=("AT2",),
    coordinate_scale=1.0,
    rename_celltypes=None,
):
    """Prepare strictly validated spatial and MHCII-score metadata."""
    required = [score_col, celltype_col, core_col, donor_col, tissue_col]
    missing = [column for column in required if column not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing adata.obs columns: {missing}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"{spatial_key!r} is absent from adata.obsm.")
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be finite and greater than zero.")
    coordinates = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must contain at least two coordinate columns."
        )
    if coordinates.shape[0] != adata.n_obs:
        raise ValueError("Spatial coordinates do not match adata.n_obs.")
    if not np.isfinite(coordinates[:, :2]).all():
        raise ValueError("Spatial coordinates must be finite.")

    obs = adata.obs
    celltypes = obs[celltype_col].astype("object").copy()
    if rename_celltypes is not None:
        celltypes = celltypes.replace(rename_celltypes)
    data = pd.DataFrame(
        {
            "cell_id": obs.index.astype(str),
            "celltype": celltypes.to_numpy(),
            "score": pd.to_numeric(obs[score_col], errors="coerce").to_numpy(),
            "core_id": obs[core_col].astype("object").to_numpy(),
            "donor_id": obs[donor_col].astype("object").to_numpy(),
            "tissue_annotation": obs[tissue_col].astype("object").to_numpy(),
            "x": coordinates[:, 0] * coordinate_scale,
            "y": coordinates[:, 1] * coordinate_scale,
        }
    )
    metadata = ["celltype", "core_id", "donor_id", "tissue_annotation"]
    if data[metadata].isna().any().any():
        raise ValueError(
            "Stage 2 cell type, core, donor, and tissue metadata cannot be missing."
        )
    data["celltype"] = data["celltype"].astype(str)
    data["is_am"] = data["celltype"].isin(set(am_labels))
    data["is_at2"] = data["celltype"].isin(set(at2_labels))
    for _, core_data in data.groupby("core_id", sort=False, observed=True):
        _get_core_metadata(core_data)
    return data.reset_index(drop=True)


def _validate_stage2_common(
    *, n_permutations, min_am_cells, min_at2_cells, min_analyzed_am_cells,
):
    """Validate Stage 2 count and permutation parameters."""
    for name, value in (
        ("n_permutations", n_permutations),
        ("min_am_cells", min_am_cells),
        ("min_at2_cells", min_at2_cells),
        ("min_analyzed_am_cells", min_analyzed_am_cells),
    ):
        _validate_positive_integer(value, name)


def _significance_label(p):
    """Return conventional asterisk notation for a finite P or Q value."""
    if not np.isfinite(p):
        return "NA"
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _calculate_stage2_continuous_exposure_by_core(
    adata,
    mode,
    scales,
    min_am_cells=20,
    min_at2_cells=10,
    min_analyzed_am_cells=5,
    n_permutations=999,
    random_state=123,
    score_col="MHCIIhi_score",
    celltype_col="CellType_refined",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    am_labels=("AM", "Alveolar Macrophage"),
    at2_labels=("AT2",),
    coordinate_scale=1.0,
    rename_celltypes=None,
):
    """Calculate a core-level score correlation with local AT2 fraction."""
    if mode not in {"knn", "radius"}:
        raise ValueError("mode must be 'knn' or 'radius'.")
    scale_name = "k_values" if mode == "knn" else "radii"
    scales = _validate_stage1_parameters(
        scales,
        scale_name,
        mode,
        n_permutations,
        min_am_cells,
        min_at2_cells,
    )
    _validate_positive_integer(min_analyzed_am_cells, "min_analyzed_am_cells")
    data = _prepare_stage2_data(
        adata=adata,
        score_col=score_col,
        celltype_col=celltype_col,
        core_col=core_col,
        donor_col=donor_col,
        tissue_col=tissue_col,
        spatial_key=spatial_key,
        am_labels=am_labels,
        at2_labels=at2_labels,
        coordinate_scale=coordinate_scale,
        rename_celltypes=rename_celltypes,
    )
    method_name = (
        "Continuous MHCII kNN" if mode == "knn" else "Continuous MHCII radius"
    )
    scale_type = "k_neighbors" if mode == "knn" else "radius_um"
    rng = np.random.default_rng(random_state)
    records = []

    for core_id, core_data in data.groupby("core_id", sort=False, observed=True):
        core_data = core_data.reset_index(drop=True)
        scores_all = core_data["score"].to_numpy(dtype=float)
        am_indices = np.flatnonzero(
            core_data["is_am"].to_numpy() & np.isfinite(scores_all)
        )
        at2_mask = core_data["is_at2"].to_numpy().astype(np.int64)
        n_am = len(am_indices)
        n_at2 = int(at2_mask.sum())
        if n_am < min_am_cells or n_at2 < min_at2_cells:
            continue
        coordinates = core_data[["x", "y"]].to_numpy(dtype=float)
        donor_id, tissue_annotation = _get_core_metadata(core_data)

        for scale in scales:
            adjacency = _make_query_adjacency(coordinates, am_indices, mode, scale)
            total_neighbors = np.asarray(adjacency.sum(axis=1)).ravel()
            at2_counts = np.asarray(adjacency @ at2_mask).ravel()
            analyzable = total_neighbors > 0
            n_analyzed = int(analyzable.sum())
            if n_analyzed < min_analyzed_am_cells:
                continue
            analyzed_scores = scores_all[am_indices][analyzable]
            fractions = at2_counts[analyzable] / total_neighbors[analyzable]
            observed = _safe_spearman(
                analyzed_scores,
                fractions,
                min_cells=min_analyzed_am_cells,
            )
            if not np.isfinite(observed):
                continue
            null_values = np.asarray(
                [
                    _safe_spearman(
                        rng.permutation(analyzed_scores),
                        fractions,
                        min_cells=min_analyzed_am_cells,
                    )
                    for _ in range(n_permutations)
                ],
                dtype=float,
            )
            summary = _signed_permutation_summary(observed, null_values)
            records.append(
                {
                    "method": method_name,
                    "direction": "MHCII score → AT2 exposure",
                    "scale_type": scale_type,
                    "scale": float(scale),
                    "effect_type": "correlation",
                    "core_id": core_id,
                    "donor_id": donor_id,
                    "tissue_annotation": tissue_annotation,
                    "n_cells": len(core_data),
                    "n_am": n_am,
                    "n_at2": n_at2,
                    "n_am_analyzed": n_analyzed,
                    "n_am_without_neighbors": int((~analyzable).sum()),
                    "observed": observed,
                    "rho_score_at2_fraction": observed,
                    "mean_at2_fraction": float(np.mean(fractions)),
                    "pct_am_with_at2": float(
                        100 * np.mean(at2_counts[analyzable] > 0)
                    ),
                    "mean_at2_neighbors": float(np.mean(at2_counts[analyzable])),
                    "mean_total_neighbors": float(
                        np.mean(total_neighbors[analyzable])
                    ),
                    "pct_am_without_neighbors": float(100 * np.mean(~analyzable)),
                    **summary,
                }
            )
    if not records:
        return _empty_frame(_STAGE2_EXPOSURE_COLUMNS)
    return pd.DataFrame.from_records(records, columns=_STAGE2_EXPOSURE_COLUMNS)


def calculate_stage2_knn_continuum_by_core(adata, k_values=(5, 15, 30), **kwargs):
    """Correlate AM MHCII score with AT2 fraction among k nearest cells.

    Parameters
    ----------
    adata
        AnnData-like object containing observation metadata and spatial coordinates.
    k_values
        Positive numbers of nearest neighboring cells.
    **kwargs
        Additional Stage 2 thresholds, labels, column names, and seed settings.

    Returns
    -------
    pandas.DataFrame
        One core-level result per requested k when analysis criteria are met.
    """
    return _calculate_stage2_continuous_exposure_by_core(
        adata=adata, mode="knn", scales=k_values, **kwargs
    )


def calculate_stage2_radius_continuum_by_core(adata, radii=(25, 50, 100), **kwargs):
    """Correlate AM MHCII score with local AT2 fraction at fixed radii.

    Parameters
    ----------
    adata
        AnnData-like object containing observation metadata and spatial coordinates.
    radii
        Positive physical radii after applying `coordinate_scale`.
    **kwargs
        Additional Stage 2 thresholds, labels, column names, and seed settings.

    Returns
    -------
    pandas.DataFrame
        Core-level correlations; AMs without any neighbor are excluded by radius.
    """
    return _calculate_stage2_continuous_exposure_by_core(
        adata=adata, mode="radius", scales=radii, **kwargs
    )


def calculate_stage2_nearest_at2_by_core(
    adata,
    min_am_cells=20,
    min_at2_cells=10,
    min_analyzed_am_cells=5,
    n_permutations=999,
    random_state=123,
    score_col="MHCIIhi_score",
    celltype_col="CellType_refined",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    am_labels=("AM", "Alveolar Macrophage"),
    at2_labels=("AT2",),
    coordinate_scale=1.0,
    rename_celltypes=None,
):
    """Correlate AM MHCII score with nearest-AT2 proximity by spatial core.

    Parameters
    ----------
    adata
        AnnData-like object containing score, cell type, core, donor, and tissue data.
    min_am_cells, min_at2_cells, min_analyzed_am_cells
        Minimum eligible AM, AT2, and analyzable AM counts per core.
    n_permutations
        Number of within-core score permutations.
    random_state
        Reproducible random seed.
    score_col, celltype_col, core_col, donor_col, tissue_col, spatial_key
        Observation-column and spatial-coordinate keys.
    am_labels, at2_labels
        Labels defining alveolar macrophages and AT2 cells.
    coordinate_scale
        Positive multiplier converting spatial coordinates to micrometres.
    rename_celltypes
        Optional mapping applied before matching cell-type labels.

    Returns
    -------
    pandas.DataFrame
        Core-level correlations with `effect = -rho`; positive means closer to AT2.
    """
    _validate_stage2_common(
        n_permutations=n_permutations,
        min_am_cells=min_am_cells,
        min_at2_cells=min_at2_cells,
        min_analyzed_am_cells=min_analyzed_am_cells,
    )
    data = _prepare_stage2_data(
        adata=adata,
        score_col=score_col,
        celltype_col=celltype_col,
        core_col=core_col,
        donor_col=donor_col,
        tissue_col=tissue_col,
        spatial_key=spatial_key,
        am_labels=am_labels,
        at2_labels=at2_labels,
        coordinate_scale=coordinate_scale,
        rename_celltypes=rename_celltypes,
    )
    rng = np.random.default_rng(random_state)
    records = []
    for core_id, core_data in data.groupby("core_id", sort=False, observed=True):
        core_data = core_data.reset_index(drop=True)
        scores_all = core_data["score"].to_numpy(dtype=float)
        am_indices = np.flatnonzero(
            core_data["is_am"].to_numpy() & np.isfinite(scores_all)
        )
        at2_indices = np.flatnonzero(core_data["is_at2"].to_numpy())
        n_am, n_at2 = len(am_indices), len(at2_indices)
        if (
            n_am < min_am_cells
            or n_at2 < min_at2_cells
            or n_am < min_analyzed_am_cells
        ):
            continue
        am_scores = scores_all[am_indices]
        coordinates = core_data[["x", "y"]].to_numpy(dtype=float)
        nearest, _ = cKDTree(coordinates[at2_indices]).query(
            coordinates[am_indices], k=1
        )
        raw_rho = _safe_spearman(
            am_scores, nearest, min_cells=min_analyzed_am_cells
        )
        if not np.isfinite(raw_rho):
            continue
        observed = -raw_rho
        null_values = np.asarray(
            [
                -_safe_spearman(
                    rng.permutation(am_scores),
                    nearest,
                    min_cells=min_analyzed_am_cells,
                )
                for _ in range(n_permutations)
            ],
            dtype=float,
        )
        summary = _signed_permutation_summary(observed, null_values)
        donor_id, tissue_annotation = _get_core_metadata(core_data)
        records.append(
            {
                "method": "Continuous MHCII nearest AT2",
                "direction": "MHCII score → AT2 proximity",
                "scale_type": "nearest_target",
                "scale": 1.0,
                "effect_type": "correlation",
                "core_id": core_id,
                "donor_id": donor_id,
                "tissue_annotation": tissue_annotation,
                "n_cells": len(core_data),
                "n_am": n_am,
                "n_at2": n_at2,
                "n_am_analyzed": n_am,
                "n_am_without_neighbors": 0,
                "observed": observed,
                "rho_score_distance_raw": raw_rho,
                "mean_nearest_distance": float(np.mean(nearest)),
                "median_nearest_distance": float(np.median(nearest)),
                "q25_nearest_distance": float(np.quantile(nearest, 0.25)),
                "q75_nearest_distance": float(np.quantile(nearest, 0.75)),
                **summary,
            }
        )
    if not records:
        return _empty_frame(_STAGE2_NEAREST_COLUMNS)
    return pd.DataFrame.from_records(records, columns=_STAGE2_NEAREST_COLUMNS)


def calculate_stage2_balanced_extremes_by_core(
    adata,
    radii=(25, 50, 100),
    extreme_fraction=0.25,
    min_am_cells=20,
    min_at2_cells=10,
    min_extreme_cells=5,
    min_analyzed_am_cells=5,
    n_permutations=999,
    random_state=123,
    score_col="MHCIIhi_score",
    celltype_col="CellType_refined",
    core_col="core_id",
    donor_col="donor_id",
    tissue_col="tissue_annotation",
    spatial_key="spatial",
    am_labels=("AM", "Alveolar Macrophage"),
    at2_labels=("AT2",),
    coordinate_scale=1.0,
    rename_celltypes=None,
):
    """Compare local AT2 exposure between balanced MHCII-score extremes.

    Parameters
    ----------
    adata
        AnnData-like object containing score, cell type, core, donor, and tissue data.
    radii
        Positive physical neighborhood radii.
    extreme_fraction
        Fraction selected from each score tail; greater than zero and at most 0.5.
    min_am_cells, min_at2_cells, min_extreme_cells, min_analyzed_am_cells
        Minimum eligible cell counts required for a core/radius result.
    n_permutations
        Number of within-core balanced-label permutations.
    random_state
        Reproducible random seed.
    score_col, celltype_col, core_col, donor_col, tissue_col, spatial_key
        Observation-column and spatial-coordinate keys.
    am_labels, at2_labels
        Labels defining alveolar macrophages and AT2 cells.
    coordinate_scale
        Positive multiplier converting coordinates to micrometres.
    rename_celltypes
        Optional mapping applied before matching cell-type labels.

    Returns
    -------
    pandas.DataFrame
        Core-level high-minus-low AT2-exposure differences. Ambiguous tied score
        boundaries are omitted rather than split arbitrarily.
    """
    radii = _validate_stage1_parameters(
        radii, "radii", "radius", n_permutations, min_am_cells, min_at2_cells
    )
    _validate_positive_integer(min_extreme_cells, "min_extreme_cells")
    _validate_positive_integer(min_analyzed_am_cells, "min_analyzed_am_cells")
    if (
        not np.isscalar(extreme_fraction)
        or not np.isfinite(extreme_fraction)
        or not 0 < float(extreme_fraction) <= 0.5
    ):
        raise ValueError(
            "extreme_fraction must be greater than zero and at most 0.5."
        )
    data = _prepare_stage2_data(
        adata=adata,
        score_col=score_col,
        celltype_col=celltype_col,
        core_col=core_col,
        donor_col=donor_col,
        tissue_col=tissue_col,
        spatial_key=spatial_key,
        am_labels=am_labels,
        at2_labels=at2_labels,
        coordinate_scale=coordinate_scale,
        rename_celltypes=rename_celltypes,
    )
    rng = np.random.default_rng(random_state)
    records = []
    for core_id, core_data in data.groupby("core_id", sort=False, observed=True):
        core_data = core_data.reset_index(drop=True)
        scores_all = core_data["score"].to_numpy(dtype=float)
        am_indices = np.flatnonzero(
            core_data["is_am"].to_numpy() & np.isfinite(scores_all)
        )
        at2_mask = core_data["is_at2"].to_numpy().astype(np.int64)
        n_am, n_at2 = len(am_indices), int(at2_mask.sum())
        if n_am < min_am_cells or n_at2 < min_at2_cells:
            continue
        coordinates = core_data[["x", "y"]].to_numpy(dtype=float)
        donor_id, tissue_annotation = _get_core_metadata(core_data)
        for radius in radii:
            adjacency = _make_query_adjacency(
                coordinates, am_indices, "radius", radius
            )
            total_neighbors = np.asarray(adjacency.sum(axis=1)).ravel()
            at2_counts = np.asarray(adjacency @ at2_mask).ravel()
            analyzable = total_neighbors > 0
            n_analyzed = int(analyzable.sum())
            if n_analyzed < min_analyzed_am_cells:
                continue
            analyzed_scores = scores_all[am_indices][analyzable]
            fractions = at2_counts[analyzable] / total_neighbors[analyzable]
            n_extreme = min(
                int(np.floor(n_analyzed * extreme_fraction)), n_analyzed // 2
            )
            if n_extreme < min_extreme_cells:
                continue
            order = np.argsort(analyzed_scores, kind="stable")
            sorted_scores = analyzed_scores[order]
            low_tied = (
                n_extreme < n_analyzed
                and sorted_scores[n_extreme - 1] == sorted_scores[n_extreme]
            )
            high_tied = (
                n_extreme < n_analyzed
                and sorted_scores[-n_extreme] == sorted_scores[-n_extreme - 1]
            )
            if low_tied or high_tied:
                continue
            low_positions = order[:n_extreme]
            high_positions = order[-n_extreme:]
            low_values = fractions[low_positions]
            high_values = fractions[high_positions]
            observed = float(np.mean(high_values) - np.mean(low_values))
            selected = np.concatenate([low_values, high_values])
            null_values = np.empty(n_permutations, dtype=float)
            for permutation in range(n_permutations):
                shuffled = selected[rng.permutation(len(selected))]
                null_values[permutation] = float(
                    np.mean(shuffled[n_extreme:]) - np.mean(shuffled[:n_extreme])
                )
            summary = _signed_permutation_summary(observed, null_values)
            records.append(
                {
                    "method": "Balanced MHCII extremes",
                    "direction": "MHCII-high minus MHCII-low",
                    "scale_type": "radius_um",
                    "scale": float(radius),
                    "effect_type": "difference",
                    "core_id": core_id,
                    "donor_id": donor_id,
                    "tissue_annotation": tissue_annotation,
                    "n_cells": len(core_data),
                    "n_am": n_am,
                    "n_at2": n_at2,
                    "n_am_analyzed": n_analyzed,
                    "n_am_without_neighbors": int((~analyzable).sum()),
                    "observed": observed,
                    "n_high": n_extreme,
                    "n_low": n_extreme,
                    "mean_high_at2_fraction": float(np.mean(high_values)),
                    "mean_low_at2_fraction": float(np.mean(low_values)),
                    "mean_high_score": float(
                        np.mean(analyzed_scores[high_positions])
                    ),
                    "mean_low_score": float(np.mean(analyzed_scores[low_positions])),
                    "pct_am_without_neighbors": float(100 * np.mean(~analyzable)),
                    **summary,
                }
            )
    if not records:
        return _empty_frame(_STAGE2_EXTREME_COLUMNS)
    return pd.DataFrame.from_records(records, columns=_STAGE2_EXTREME_COLUMNS)


def _run_spatial_function_one_core(
    analysis_function,
    core_obs,
    core_spatial,
    function_kwargs,
    random_seed,
):
    """Run one spatial function using a lightweight AnnData-like object."""
    from types import SimpleNamespace

    spatial_key = function_kwargs.get("spatial_key", "spatial")
    core_adata = SimpleNamespace(
        obs=core_obs,
        obsm={spatial_key: core_spatial},
        n_obs=len(core_obs),
    )
    worker_kwargs = function_kwargs.copy()
    worker_kwargs["random_state"] = int(random_seed)
    return analysis_function(adata=core_adata, **worker_kwargs)


def _stable_core_seed(random_state, core_id):
    """Derive a reproducible seed from the master seed and core identity."""
    if (
        not isinstance(random_state, (int, np.integer))
        or isinstance(random_state, bool)
        or random_state < 0
    ):
        raise ValueError("random_state must be a nonnegative integer.")
    identity = (
        f"{int(random_state)}\0"
        f"{type(core_id).__module__}.{type(core_id).__qualname__}\0"
        f"{core_id!r}"
    ).encode("utf-8")
    digest = hashlib.blake2b(identity, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def run_spatial_function_multicore(
    analysis_function,
    adata,
    n_jobs=16,
    random_state=123,
    verbose=10,
    **function_kwargs,
):
    """Run a metadata-only spatial analysis independently for every core.

    Parameters
    ----------
    analysis_function
        Callable accepting `adata`, `random_state`, and `function_kwargs`.
    adata
        Main AnnData object. Its expression matrix is not copied to workers.
    n_jobs
        Positive number of worker processes, capped at the number of cores.
    random_state
        Master seed used to generate reproducible core-specific seeds.
    verbose
        Joblib progress level when more than one worker is requested.
    **function_kwargs
        Arguments forwarded to the spatial analysis function.

    Returns
    -------
    pandas.DataFrame
        Concatenated core-level outputs, with diagnostic Stage 2 core FDR.
    """
    if not callable(analysis_function):
        raise TypeError("analysis_function must be callable.")
    _validate_positive_integer(n_jobs, "n_jobs")
    core_col = function_kwargs.get("core_col", "core_id")
    celltype_col = function_kwargs.get("celltype_col", "CellType_refined")
    donor_col = function_kwargs.get("donor_col", "donor_id")
    tissue_col = function_kwargs.get("tissue_col", "tissue_annotation")
    score_col = function_kwargs.get("score_col", "MHCIIhi_score")
    spatial_key = function_kwargs.get("spatial_key", "spatial")
    required = [core_col, celltype_col, donor_col, tissue_col]
    missing = [column for column in required if column not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing adata.obs columns: {missing}")
    if spatial_key not in adata.obsm:
        raise KeyError(f"{spatial_key!r} is absent from adata.obsm.")
    coordinates = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must contain at least two coordinate columns."
        )
    if len(coordinates) != adata.n_obs or not np.isfinite(
        coordinates[:, :2]
    ).all():
        raise ValueError("Spatial coordinates must be finite and match adata.n_obs.")
    metadata_columns = list(
        dict.fromkeys(
            required + ([score_col] if score_col in adata.obs.columns else [])
        )
    )
    core_indices = adata.obs.groupby(
        core_col, observed=True, sort=False
    ).indices
    core_ids = list(core_indices)
    if not core_ids:
        raise ValueError(f"No cores were found in {core_col!r}.")
    worker_count = min(int(n_jobs), len(core_ids))
    seeds = [_stable_core_seed(random_state, core_id) for core_id in core_ids]
    tasks = []
    for core_id, seed in zip(core_ids, seeds):
        positions = np.asarray(core_indices[core_id], dtype=int)
        tasks.append(
            (
                adata.obs.iloc[positions][metadata_columns].copy(),
                coordinates[positions, :2].copy(),
                seed,
            )
        )
    if worker_count == 1:
        results = [
            _run_spatial_function_one_core(
                analysis_function,
                core_obs,
                core_spatial,
                function_kwargs,
                seed,
            )
            for core_obs, core_spatial, seed in tasks
        ]
    else:
        try:
            from joblib import Parallel, delayed, parallel_backend
        except ImportError as exc:
            raise ImportError(
                "Parallel spatial execution requires joblib; install it on the HPC."
            ) from exc
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(
                n_jobs=worker_count, verbose=verbose, batch_size=1
            )(
                delayed(_run_spatial_function_one_core)(
                    analysis_function,
                    core_obs,
                    core_spatial,
                    function_kwargs,
                    seed,
                )
                for core_obs, core_spatial, seed in tasks
            )
    frames = [result for result in results if result is not None]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "effect_type" in combined.columns and "p_association" in combined.columns:
        combined["FDR_core"] = np.nan
        if not combined.empty:
            groups = combined.groupby(
                ["method", "scale_type", "scale"],
                observed=True,
                dropna=False,
            ).groups
            for indices in groups.values():
                indices = list(indices)
                combined.loc[indices, "FDR_core"] = _bh_adjust(
                    combined.loc[indices, "p_association"].to_numpy()
                )
    return combined


def run_stage2_multicore(
    analysis_function,
    adata,
    n_jobs=16,
    random_state=123,
    verbose=10,
    **function_kwargs,
):
    """Run a Stage 2 spatial function in parallel by core.

    Parameters
    ----------
    analysis_function
        One of the public `calculate_stage2_*_by_core` functions.
    adata
        Main AnnData object.
    n_jobs, random_state, verbose
        Worker count, reproducible master seed, and joblib progress level.
    **function_kwargs
        Arguments forwarded to the Stage 2 function.

    Returns
    -------
    pandas.DataFrame
        Combined core-level Stage 2 results.
    """
    return run_spatial_function_multicore(
        analysis_function=analysis_function,
        adata=adata,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        **function_kwargs,
    )


def summarize_stage2_by_donor_and_tissue(core_results, min_donors=3):
    """Aggregate Stage 2 core effects and perform donor-level tissue tests.

    Parameters
    ----------
    core_results
        Core-level output from one or more Stage 2 calculation functions.
    min_donors
        Minimum donor count required for a one-sided Wilcoxon signed-rank test.

    Returns
    -------
    donor_summary, tissue_tests : tuple[pandas.DataFrame, pandas.DataFrame]
        Donor-by-tissue effects and tissue-level tests with BH-adjusted FDR.
    """
    _validate_positive_integer(min_donors, "min_donors")
    grouping = [
        "method", "direction", "effect_type", "scale_type", "scale",
        "donor_id", "tissue_annotation",
    ]
    donor_columns = grouping + [
        "donor_effect", "donor_zscore", "n_cores", "mean_n_am",
        "mean_n_at2", "mean_n_am_analyzed",
    ]
    test_grouping = [
        "method", "direction", "effect_type", "scale_type", "scale",
        "tissue_annotation",
    ]
    test_columns = test_grouping + [
        "n_donors", "mean_effect", "median_effect", "wilcoxon_statistic",
        "p_value", "FDR",
    ]
    if core_results.empty and not set(
        grouping + ["core_id", "effect"]
    ).issubset(core_results.columns):
        return (
            pd.DataFrame(columns=donor_columns),
            pd.DataFrame(columns=test_columns),
        )
    required = set(
        grouping
        + ["core_id", "effect", "zscore", "n_am", "n_at2", "n_am_analyzed"]
    )
    missing = required.difference(core_results.columns)
    if missing:
        raise KeyError(f"Missing core_results columns: {sorted(missing)}")
    donor_summary = (
        core_results.groupby(grouping, observed=True, dropna=False)
        .agg(
            donor_effect=("effect", "mean"),
            donor_zscore=("zscore", "mean"),
            n_cores=("core_id", "nunique"),
            mean_n_am=("n_am", "mean"),
            mean_n_at2=("n_at2", "mean"),
            mean_n_am_analyzed=("n_am_analyzed", "mean"),
        )
        .reset_index()
    )
    records = []
    for keys, group in donor_summary.groupby(
        test_grouping, observed=True, dropna=False
    ):
        values = group["donor_effect"].dropna().to_numpy(dtype=float)
        n_donors = len(values)
        statistic = p_value = np.nan
        if n_donors >= min_donors and np.any(values != 0):
            result = wilcoxon(
                values, alternative="greater", zero_method="wilcox"
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        record = dict(zip(test_grouping, keys))
        record.update(
            {
                "n_donors": n_donors,
                "mean_effect": float(np.mean(values)) if n_donors else np.nan,
                "median_effect": float(np.median(values)) if n_donors else np.nan,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
        records.append(record)
    tissue_tests = pd.DataFrame(records)
    if tissue_tests.empty:
        tissue_tests = pd.DataFrame(columns=test_columns)
    else:
        tissue_tests["FDR"] = np.nan
        fdr_family = [
            "method", "effect_type", "scale_type", "scale", "tissue_annotation"
        ]
        groups = tissue_tests.groupby(
            fdr_family, observed=True, dropna=False
        ).groups
        for indices in groups.values():
            indices = list(indices)
            tissue_tests.loc[indices, "FDR"] = _bh_adjust(
                tissue_tests.loc[indices, "p_value"].to_numpy()
            )
        tissue_tests = tissue_tests[test_columns]
    return donor_summary[donor_columns], tissue_tests


def plot_stage1A_niche_dotmap(
    tissue_tests,
    focal_types=("AM", "AT2"),
    tissue_order=("A", "B", "V"),
    method_config=None,
    top_n=15,
    effect_clip=None,
    fdr_threshold=0.05,
    ranking_fdr_threshold=0.10,
    figsize_per_panel=(5.2, 7.2),
    save_prefix=None,
    dpi=300,
):
    """
    Plot Stage 1A donor-level cell-type niche discovery results.

    One figure is created for every focal cell type.

    Encoding
    --------
    Dot colour:
        Median donor spatial effect.
        Positive = enriched or closer than expected.
        Negative = depleted or farther than expected.

    Dot size:
        -log10(FDR).

    Black outline:
        FDR < fdr_threshold.

    Target ranking:
        1. Number of significant positive method/tissue combinations.
        2. Number of positive combinations.
        3. Median within-method rank.

    Parameters
    ----------
    tissue_tests
        Tissue-level output from
        summarize_stage1_by_donor_and_tissue().

    focal_types
        Focal cell types to plot.

    tissue_order
        Tissue annotation order.

    method_config
        Optional method configuration. The defaults support names from
        both versions of the Stage 1 functions.

    top_n
        Number of target cell types to show per focal type.

    effect_clip
        Symmetric colour limit. If None, determined separately for each
        focal-cell figure using the 95th percentile.

    save_prefix
        For example "Stage1A_niche". This creates:
          Stage1A_niche_AM.pdf
          Stage1A_niche_AT2.pdf

    Returns
    -------
    figures
        Dictionary of matplotlib figures.

    axes_dictionary
        Dictionary of axes arrays.

    plot_tables
        Dictionary containing the plotted summary data.
    """
    required_columns = {
        "method",
        "focal_type",
        "target_type",
        "scale",
        "tissue_annotation",
        "median_effect",
        "FDR",
    }

    missing = required_columns.difference(
        tissue_tests.columns
    )

    if missing:
        raise KeyError(
            f"Missing tissue_tests columns: {sorted(missing)}"
        )

    if method_config is None:
        method_config = [
            {
                "candidates": (
                    "Contact enrichment",
                    "Neighborhood contact enrichment",
                ),
                "scale": 50,
                "label": "Contact\n50 µm",
            },
            {
                "candidates": (
                    "kNN niche enrichment",
                    "Bidirectional kNN enrichment",
                ),
                "scale": 15,
                "label": "kNN\nk = 15",
            },
            {
                "candidates": (
                    "Fixed-radius niche enrichment",
                    "Bidirectional fixed-radius enrichment",
                ),
                "scale": 50,
                "label": "Radius\n50 µm",
            },
            {
                "candidates": (
                    "Nearest-target distance",
                    "Bidirectional nearest-target distance",
                ),
                "scale": 1,
                "label": "Nearest\ntarget",
            },
        ]

    results = tissue_tests.copy()

    available_methods = set(
        results["method"]
        .dropna()
        .astype(str)
        .unique()
    )

    resolved_methods = []

    for configuration in method_config:
        selected_method = next(
            (
                candidate
                for candidate in configuration["candidates"]
                if candidate in available_methods
            ),
            None,
        )

        if selected_method is not None:
            resolved_methods.append(
                {
                    "method": selected_method,
                    "scale": float(
                        configuration["scale"]
                    ),
                    "label": configuration["label"],
                }
            )

    if not resolved_methods:
        raise ValueError(
            "No method names in method_config were found in "
            "tissue_tests."
        )

    selected_parts = []

    for configuration in resolved_methods:
        selected = results.loc[
            (
                results["method"] ==
                configuration["method"]
            ) &
            np.isclose(
                results["scale"].astype(float),
                configuration["scale"],
            )
        ].copy()

        selected["method_label"] = (
            configuration["label"]
        )

        selected_parts.append(selected)

    selected_results = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected_results = selected_results.loc[
        selected_results["tissue_annotation"].isin(
            tissue_order
        )
    ].copy()

    # Occasionally contact results may contain duplicate representations.
    # Collapse these before plotting.
    group_columns = [
        "focal_type",
        "target_type",
        "tissue_annotation",
        "method",
        "method_label",
        "scale",
    ]

    aggregation = {
        "median_effect": "median",
        "FDR": "min",
    }

    if "n_donors" in selected_results.columns:
        aggregation["n_donors"] = "max"

    selected_results = (
        selected_results
        .groupby(
            group_columns,
            observed=True,
            dropna=False,
        )
        .agg(aggregation)
        .reset_index()
    )

    method_labels = [
        configuration["label"]
        for configuration in resolved_methods
    ]

    # Saturated but still cell-style diverging colours.
    niche_cmap = LinearSegmentedColormap.from_list(
        "stage1A_niche",
        [
            "#5276B5",  # spatial depletion
            "#F4F0E8",  # null
            "#D96873",  # spatial enrichment
        ],
    )

    figures = {}
    axes_dictionary = {}
    plot_tables = {}

    for focal_type in focal_types:
        focal_data = selected_results.loc[
            selected_results["focal_type"].astype(str) ==
            str(focal_type)
        ].copy()

        if focal_data.empty:
            print(
                f"No Stage 1A results found for "
                f"focal_type={focal_type!r}."
            )
            continue

        # ----------------------------------------------------
        # Rank targets without averaging effect magnitudes
        # across different methods.
        # ----------------------------------------------------

        focal_data["positive"] = (
            focal_data["median_effect"] > 0
        )

        focal_data["significant_positive"] = (
            (focal_data["median_effect"] > 0) &
            (focal_data["FDR"] < ranking_fdr_threshold)
        )

        # Within each method and tissue, convert effect into a
        # percentile rank. This avoids directly averaging effect
        # magnitudes from different spatial methods.
        focal_data["effect_percentile"] = (
            focal_data
            .groupby(
                [
                    "method_label",
                    "tissue_annotation",
                ],
                observed=True,
            )["median_effect"]
            .rank(
                method="average",
                pct=True,
            )
        )

        ranking = (
            focal_data
            .groupby(
                "target_type",
                observed=True,
            )
            .agg(
                significant_positive_tests=(
                    "significant_positive",
                    "sum",
                ),
                positive_tests=(
                    "positive",
                    "sum",
                ),
                median_effect_percentile=(
                    "effect_percentile",
                    "median",
                ),
                best_FDR=(
                    "FDR",
                    "min",
                ),
            )
            .reset_index()
            .sort_values(
                [
                    "significant_positive_tests",
                    "positive_tests",
                    "median_effect_percentile",
                    "best_FDR",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
            )
        )

        selected_targets = (
            ranking
            .head(top_n)["target_type"]
            .tolist()
        )

        plot_data = focal_data.loc[
            focal_data["target_type"].isin(
                selected_targets
            )
        ].copy()

        # Preserve ranked order.
        target_order = selected_targets

        plot_data["target_type"] = pd.Categorical(
            plot_data["target_type"],
            categories=target_order,
            ordered=True,
        )

        plot_data["method_label"] = pd.Categorical(
            plot_data["method_label"],
            categories=method_labels,
            ordered=True,
        )

        plot_data["tissue_annotation"] = pd.Categorical(
            plot_data["tissue_annotation"],
            categories=tissue_order,
            ordered=True,
        )

        plot_data = plot_data.sort_values(
            [
                "target_type",
                "tissue_annotation",
                "method_label",
            ]
        )

        # ----------------------------------------------------
        # Colour limits
        # ----------------------------------------------------

        finite_effects = (
            plot_data["median_effect"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .abs()
            .to_numpy()
        )

        if effect_clip is None:
            if len(finite_effects) > 0:
                focal_effect_clip = float(
                    np.quantile(
                        finite_effects,
                        0.95,
                    )
                )
            else:
                focal_effect_clip = 1.0

            focal_effect_clip = max(
                focal_effect_clip,
                0.10,
            )
        else:
            focal_effect_clip = float(effect_clip)

        norm = TwoSlopeNorm(
            vmin=-focal_effect_clip,
            vcenter=0,
            vmax=focal_effect_clip,
        )

        n_tissues = len(tissue_order)

        fig, axes = plt.subplots(
            1,
            n_tissues,
            figsize=(
                figsize_per_panel[0] * n_tissues,
                max(
                    figsize_per_panel[1],
                    0.42 * len(target_order) + 2.6,
                ),
            ),
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        axes = axes.ravel()

        x_positions = {
            method: position
            for position, method in enumerate(
                method_labels
            )
        }

        # First ranked target should appear at the top.
        y_positions = {
            target: (
                len(target_order) - 1 - position
            )
            for position, target in enumerate(
                target_order
            )
        }

        for ax, tissue in zip(
            axes,
            tissue_order,
        ):
            panel = plot_data.loc[
                plot_data["tissue_annotation"] == tissue
            ].copy()

            for _, row in panel.iterrows():
                if pd.isna(row["median_effect"]):
                    continue

                x = x_positions[
                    str(row["method_label"])
                ]

                y = y_positions[
                    str(row["target_type"])
                ]

                fdr = row["FDR"]

                if pd.notna(fdr) and fdr > 0:
                    evidence = min(
                        -np.log10(float(fdr)),
                        4.0,
                    )
                else:
                    evidence = 0

                marker_size = 35 + 38 * evidence

                significant = (
                    pd.notna(fdr) and
                    float(fdr) < fdr_threshold
                )

                ax.scatter(
                    x,
                    y,
                    s=marker_size,
                    c=[row["median_effect"]],
                    cmap=niche_cmap,
                    norm=norm,
                    edgecolor=(
                        "#171717"
                        if significant
                        else "#8D8D8D"
                    ),
                    linewidth=(
                        1.25
                        if significant
                        else 0.45
                    ),
                    alpha=(
                        0.98
                        if significant
                        else 0.72
                    ),
                    zorder=3,
                )

            ax.axhline(
                -0.5,
                color="#D5D5D5",
                linewidth=0.7,
            )

            ax.set_title(
                f"Tissue {tissue}",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )

            ax.set_xticks(
                range(len(method_labels))
            )

            ax.set_xticklabels(
                method_labels,
                rotation=0,
                ha="center",
                fontsize=8.5,
            )

            ax.set_xlim(
                -0.55,
                len(method_labels) - 0.45,
            )

            ax.set_ylim(
                -0.6,
                len(target_order) - 0.4,
            )

            ax.grid(False)

            ax.tick_params(
                axis="x",
                length=0,
            )

            ax.tick_params(
                axis="y",
                length=0,
            )

            for spine in ax.spines.values():
                spine.set_visible(False)

        axes[0].set_yticks(
            list(y_positions.values())
        )

        axes[0].set_yticklabels(
            target_order,
            fontsize=9,
        )

        for ax in axes[1:]:
            ax.tick_params(
                labelleft=False
            )

        fig.suptitle(
            f"{focal_type}-centred cellular niche",
            fontsize=14,
            fontweight="bold",
            y=1.015,
        )

        fig.text(
            0.5,
            0.015,
            (
                "Dot colour: median donor effect "
                "(positive = enriched or closer); "
                "dot size: −log10(FDR); "
                "black outline: FDR < 0.05"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#454545",
        )

        # Shared colour bar.
        scalar_mappable = plt.cm.ScalarMappable(
            norm=norm,
            cmap=niche_cmap,
        )

        scalar_mappable.set_array([])

        colorbar = fig.colorbar(
            scalar_mappable,
            ax=axes.tolist(),
            fraction=0.025,
            pad=0.025,
        )

        colorbar.set_label(
            "Median donor spatial effect",
            fontsize=9,
        )

        colorbar.outline.set_linewidth(0.6)

        # Dot-size legend.
        example_fdrs = [0.10, 0.05, 0.01, 0.001]

        size_handles = []

        for example_fdr in example_fdrs:
            evidence = min(
                -np.log10(example_fdr),
                4.0,
            )

            size = 35 + 38 * evidence

            size_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markersize=np.sqrt(size),
                    markerfacecolor="#D9D9D9",
                    markeredgecolor=(
                        "#171717"
                        if example_fdr < fdr_threshold
                        else "#8D8D8D"
                    ),
                    markeredgewidth=(
                        1.2
                        if example_fdr < fdr_threshold
                        else 0.5
                    ),
                    label=f"q = {example_fdr:g}",
                )
            )

        fig.legend(
            handles=size_handles,
            title="Statistical evidence",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=len(size_handles),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )

        fig.subplots_adjust(
            left=0.18,
            right=0.91,
            bottom=0.12,
            top=0.87,
            wspace=0.08,
        )

        if save_prefix is not None:
            safe_focal = (
                str(focal_type)
                .replace(" ", "_")
                .replace("/", "_")
            )

            output_file = (
                f"{save_prefix}_{safe_focal}.pdf"
            )

            fig.savefig(
                output_file,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )

        figures[focal_type] = fig
        axes_dictionary[focal_type] = axes
        plot_tables[focal_type] = plot_data

    return figures, axes_dictionary, plot_tables


def _plot_stage1B_primary_impl(
    donor_summary,
    tissue_tests,
    tissue_order=("A", "B", "V"),
    tissue_colors=None,
    method_config=None,
    donor_col="donor_id",
    value_col="donor_effect",
    show_donor_labels=False,
    random_state=123,
    save=None,
    dpi=300,
):
    """
    Plot donor-level Stage 1B AM–AT2 spatial effects.

    Each panel represents one spatial method at one selected scale.

    Parameters
    ----------
    donor_summary
        Output from summarize_stage1_by_donor_and_tissue()[0].

    tissue_tests
        Output from summarize_stage1_by_donor_and_tissue()[1].

    tissue_order
        Tissue annotation plotting order.

    method_config
        List of dictionaries specifying method candidates, selected
        scale and panel title.

    show_donor_labels
        Add donor IDs beside points. Usually False for the main figure.

    Returns
    -------
    fig, axes, plot_data
    """
    if tissue_colors is None:
        tissue_colors = {
            "A": "#E8928F",     # macaron coral
            "B": "#7FAED2",     # macaron blue
            "V": "#82BFA0",     # macaron green
            "None": "#B8B8B8",
        }

    # Supports names from both versions of the functions.
    if method_config is None:
        method_config = [
            {
                "candidates": (
                    "Contact enrichment",
                    "Neighborhood contact enrichment",
                ),
                "scale": 50,
                "title": "Contact enrichment",
                "subtitle": "Cells within 50 µm",
            },
            {
                "candidates": (
                    "kNN niche enrichment",
                    "Bidirectional kNN enrichment",
                ),
                "scale": 15,
                "title": "kNN enrichment",
                "subtitle": "15 nearest neighbors",
            },
            {
                "candidates": (
                    "Fixed-radius niche enrichment",
                    "Bidirectional fixed-radius enrichment",
                ),
                "scale": 50,
                "title": "Radius enrichment",
                "subtitle": "Cells within 50 µm",
            },
            {
                "candidates": (
                    "Nearest-target distance",
                    "Bidirectional nearest-target distance",
                ),
                "scale": 1,
                "title": "Nearest-cell proximity",
                "subtitle": "Nearest target cell",
            },
        ]

    required_donor_columns = {
        "method",
        "direction",
        "scale",
        "tissue_annotation",
        donor_col,
        value_col,
    }

    missing = required_donor_columns.difference(
        donor_summary.columns
    )

    if missing:
        raise KeyError(
            f"Missing donor_summary columns: {sorted(missing)}"
        )

    donor_summary = donor_summary.copy()
    tissue_tests = tissue_tests.copy()

    # Identify the available method name for each panel.
    resolved_config = []

    available_methods = set(
        donor_summary["method"].dropna()
    )

    for config in method_config:
        selected_method = next(
            (
                candidate
                for candidate in config["candidates"]
                if candidate in available_methods
            ),
            None,
        )

        if selected_method is not None:
            resolved = config.copy()
            resolved["method"] = selected_method
            resolved_config.append(resolved)

    if len(resolved_config) == 0:
        raise ValueError(
            "None of the method names in method_config were found."
        )

    n_panels = len(resolved_config)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 4.8 * nrows),
        squeeze=False,
    )

    axes_flat = axes.ravel()
    rng = np.random.default_rng(random_state)

    plotted_tables = []

    for panel_index, config in enumerate(resolved_config):
        ax = axes_flat[panel_index]

        method = config["method"]
        selected_scale = float(config["scale"])

        panel_data = donor_summary.loc[
            (donor_summary["method"] == method) &
            np.isclose(
                donor_summary["scale"].astype(float),
                selected_scale,
            ) &
            donor_summary["tissue_annotation"].isin(
                tissue_order
            )
        ].copy()

        if panel_data.empty:
            ax.set_visible(False)
            continue

        plotted_tables.append(panel_data)

        observed_directions = list(
            panel_data["direction"]
            .dropna()
            .astype(str)
            .unique()
        )

        preferred_directions = [
            "AM ↔ AT2",
            "AM → AT2",
            "AT2 → AM",
        ]

        direction_order = [
            direction
            for direction in preferred_directions
            if direction in observed_directions
        ]

        direction_order += [
            direction
            for direction in observed_directions
            if direction not in direction_order
        ]

        row_definitions = []

        for direction in direction_order:
            for tissue in tissue_order:
                available = panel_data.loc[
                    (
                        panel_data["direction"].astype(str) ==
                        direction
                    ) &
                    (
                        panel_data["tissue_annotation"].astype(str) ==
                        str(tissue)
                    ),
                    value_col,
                ].notna().any()

                if available:
                    row_definitions.append(
                        (direction, tissue)
                    )

        y_positions = np.arange(
            len(row_definitions)
        )[::-1]

        y_labels = []
        all_values = []

        for y, (direction, tissue) in zip(
            y_positions,
            row_definitions,
        ):
            row_data = panel_data.loc[
                (
                    panel_data["direction"].astype(str) ==
                    direction
                ) &
                (
                    panel_data["tissue_annotation"].astype(str) ==
                    str(tissue)
                )
            ].copy()

            values = (
                row_data[value_col]
                .dropna()
                .astype(float)
                .to_numpy()
            )

            all_values.extend(values.tolist())

            color = tissue_colors.get(
                tissue,
                "#B8B8B8",
            )

            jitter = rng.uniform(
                -0.13,
                0.13,
                size=len(values),
            )

            ax.scatter(
                values,
                y + jitter,
                s=37,
                color=color,
                edgecolor="#2F2F2F",
                linewidth=0.55,
                alpha=0.82,
                zorder=3,
            )

            if len(values) > 0:
                median = np.median(values)
                q25, q75 = np.quantile(
                    values,
                    [0.25, 0.75],
                )

                ax.plot(
                    [q25, q75],
                    [y, y],
                    color="#303030",
                    linewidth=2.0,
                    solid_capstyle="round",
                    zorder=4,
                )

                ax.scatter(
                    median,
                    y,
                    marker="D",
                    s=62,
                    color=color,
                    edgecolor="#161616",
                    linewidth=1.1,
                    zorder=5,
                )

            if show_donor_labels:
                for _, row in row_data.iterrows():
                    if pd.notna(row[value_col]):
                        ax.text(
                            row[value_col],
                            y + rng.uniform(-0.12, 0.12),
                            f" {row[donor_col]}",
                            fontsize=6.5,
                            va="center",
                            color="#444444",
                        )

            # Locate the donor-level tissue test.
            test_match = tissue_tests.loc[
                (tissue_tests["method"] == method) &
                (
                    tissue_tests["direction"].astype(str) ==
                    direction
                ) &
                (
                    tissue_tests["tissue_annotation"].astype(str) ==
                    str(tissue)
                ) &
                np.isclose(
                    tissue_tests["scale"].astype(float),
                    selected_scale,
                )
            ]

            if not test_match.empty:
                if (
                    "FDR" in test_match.columns and
                    test_match["FDR"].notna().any()
                ):
                    q_value = float(
                        test_match["FDR"]
                        .dropna()
                        .iloc[0]
                    )
                    significance = _significance_label(
                        q_value
                    )
                    statistical_text = (
                        f"{significance}  q={q_value:.3g}"
                    )
                else:
                    p_value = float(
                        test_match["p_value"]
                        .dropna()
                        .iloc[0]
                    )
                    significance = _significance_label(
                        p_value
                    )
                    statistical_text = (
                        f"{significance}  p={p_value:.3g}"
                    )
            else:
                statistical_text = ""

            if len(direction_order) == 1:
                y_label = (
                    f"{tissue}   "
                    f"(n={len(values)})"
                )
            else:
                y_label = (
                    f"{direction} | {tissue}   "
                    f"(n={len(values)})"
                )

            y_labels.append(y_label)

            if statistical_text:
                ax.text(
                    0.985,
                    y,
                    statistical_text,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=8,
                    color="#303030",
                )

        ax.axvline(
            0,
            color="#777777",
            linestyle=(0, (3, 3)),
            linewidth=1.0,
            zorder=1,
        )

        if len(all_values) > 0:
            maximum = max(
                abs(np.nanmin(all_values)),
                abs(np.nanmax(all_values)),
                0.05,
            )

            # Extra space for statistical labels.
            ax.set_xlim(
                -1.25 * maximum,
                1.55 * maximum,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            y_labels,
            fontsize=8.5,
        )

        ax.set_xlabel(
            "Donor-level spatial effect\n"
            "Positive = enriched or closer",
            fontsize=10,
        )

        ax.set_title(
            config["title"],
            fontsize=12,
            fontweight="bold",
            pad=18,
        )

        ax.text(
            0.5,
            1.02,
            config["subtitle"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
        )

        ax.grid(False)
        ax.tick_params(
            axis="both",
            length=3,
            width=0.8,
            color="#444444",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")

    for empty_index in range(
        n_panels,
        len(axes_flat),
    ):
        axes_flat[empty_index].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=tissue_colors.get(
                tissue,
                "#B8B8B8",
            ),
            markeredgecolor="#2F2F2F",
            markeredgewidth=0.6,
            markersize=7,
            label=str(tissue),
        )
        for tissue in tissue_order
    ]

    fig.legend(
        handles=legend_handles,
        title="Tissue annotation",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(tissue_order),
        frameon=False,
    )

    fig.suptitle(
        "AM–AT2 spatial association across donors",
        fontsize=14,
        fontweight="bold",
        y=1.045,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.97)
    )

    if save is not None:
        fig.savefig(
            save,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    plot_data = (
        pd.concat(
            plotted_tables,
            ignore_index=True,
        )
        if plotted_tables
        else pd.DataFrame()
    )

    return fig, axes, plot_data


def plot_stage1B_primary(
    donor_summary,
    tissue_tests,
    tissue_order=("A", "B", "V"),
    tissue_colors=None,
    method_config=None,
    donor_col="donor_id",
    value_col="donor_effect",
    show_donor_labels=False,
    random_state=123,
    save=None,
    dpi=300,
):
    """Plot donor-level Stage 1B effects for the AM–AT2 pair only.

    Parameters
    ----------
    donor_summary
        Donor-level output from `summarize_stage1_by_donor_and_tissue`. Rows
        must include `focal_type` and `target_type`; non-AM–AT2 rows are removed.
    tissue_tests
        Tissue-level output from `summarize_stage1_by_donor_and_tissue`. It is
        filtered to the same unordered AM–AT2 pair as `donor_summary`.
    tissue_order
        Tissue annotations and display order.
    tissue_colors
        Optional mapping from tissue annotation to plot color.
    method_config
        Optional list of method candidates, selected scale, title, and subtitle.
    donor_col
        Column containing biological donor identifiers.
    value_col
        Column containing donor-level spatial effects.
    show_donor_labels
        Whether to annotate individual plotted points with donor identifiers.
    random_state
        Seed controlling visual point jitter only.
    save
        Optional output filename passed to Matplotlib.
    dpi
        Resolution used when `save` is provided.

    Returns
    -------
    fig, axes, plot_data : tuple
        Matplotlib figure, axes array, and the AM–AT2 donor rows displayed.

    Notes
    -----
    The supplied plotting implementation is called unchanged after this input
    guard. Both `AM → AT2` and `AT2 → AM` directions are retained; unrelated
    multitype comparisons cannot enter a figure titled as AM–AT2.
    """
    pair_columns = {"focal_type", "target_type"}
    missing_donor = pair_columns.difference(donor_summary.columns)
    if missing_donor:
        raise KeyError(
            f"Missing donor_summary columns: {sorted(missing_donor)}"
        )
    missing_tests = pair_columns.difference(tissue_tests.columns)
    if missing_tests:
        raise KeyError(
            f"Missing tissue_tests columns: {sorted(missing_tests)}"
        )

    am_labels = {"AM", "Alveolar Macrophage"}
    at2_labels = {"AT2"}

    def pair_mask(table):
        focal = table["focal_type"].astype(str)
        target = table["target_type"].astype(str)
        return (
            focal.isin(am_labels) & target.isin(at2_labels)
        ) | (
            focal.isin(at2_labels) & target.isin(am_labels)
        )

    filtered_donors = donor_summary.loc[pair_mask(donor_summary)].copy()
    filtered_tests = tissue_tests.loc[pair_mask(tissue_tests)].copy()
    if filtered_donors.empty:
        raise ValueError("No AM–AT2 donor results were found to plot.")

    return _plot_stage1B_primary_impl(
        donor_summary=filtered_donors,
        tissue_tests=filtered_tests,
        tissue_order=tissue_order,
        tissue_colors=tissue_colors,
        method_config=method_config,
        donor_col=donor_col,
        value_col=value_col,
        show_donor_labels=show_donor_labels,
        random_state=random_state,
        save=save,
        dpi=dpi,
    )


__all__ = [
    "CANONICAL_UNMEASURED_CHECKS", "CELLTYPE_PALETTE", "CONTEXT_GREY",
    "DARK_TEXT", "EXPRESSION_CMAP", "FOCUS_PALETTE", "MARKER_MODULES",
    "MACROPHAGE_SUBTYPE_PALETTE", "PROGRAM_CMAP", "REQUIRED_OBS_COLUMNS",
    "SEX_PALETTE", "TISSUE_PALETTE", "TMA_PALETTE",
    "add_human_gene_name", "assign_balanced_mhcii_score_groups",
    "assign_mhcii_single_signature", "calculate_knn_niche_continuum",
    "calculate_multitype_knn_niche_by_core",
    "calculate_multitype_nearest_distance_by_core",
    "calculate_multitype_nhood_enrichment_by_core",
    "calculate_multitype_radius_niche_by_core",
    "calculate_stage2_balanced_extremes_by_core",
    "calculate_stage2_knn_continuum_by_core",
    "calculate_stage2_nearest_at2_by_core",
    "calculate_stage2_radius_continuum_by_core",
    "calculate_nhood_enrichment_by_core", "calculate_radius_niche_continuum",
    "cluster_expression_summary", "compute_program_scores",
    "configure_plot_style", "extract_marker_matrices",
    "gene_detection_by_group", "marker_availability_table",
    "merge_obs_to_main", "plot_am_at2_pct_by_donor",
    "plot_am_at2_spatial", "plot_focus_umap", "plot_full_umap",
    "plot_knn_niche_continuum",
    "plot_stage1A_niche_dotmap", "plot_stage1B_primary",
    "plot_macrophage_pct_by_tissue", "plot_metadata_summary",
    "plot_marker_dotplot", "plot_nhood_enrichment_donor_tissue",
    "plot_program_umap", "plot_radius_core_correlations",
    "plot_spatial_celltypes",
    "plot_spatial_focus", "plot_spatial_programs", "save_figure",
    "select_representative_cores", "summarize_markers",
    "summarize_nhood_by_donor", "summarize_stage1_by_donor_and_tissue",
    "summarize_stage2_by_donor_and_tissue",
    "test_continuous_mhcii_at2_proximity",
    "run_spatial_function_multicore", "run_stage2_multicore",
    "validate_xenium_metadata",
]
