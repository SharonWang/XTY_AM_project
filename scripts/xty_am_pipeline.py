"""Reusable functions for the human lung Xenium AM and AT2 analysis.

The module performs no analysis at import time. Notebooks select paths, load
data, and call these functions explicitly. The published Macrophages label is
treated as a broad candidate pool, not an automatic alveolar-macrophage call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    TwoSlopeNorm,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import false_discovery_control, wilcoxon


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
    report_rows: list[dict[str, Any]] = []
    for source_column, destination_column in column_map.items():
        incoming = subset_adata.obs.loc[matched, source_column].copy()
        valid_cells = incoming.index[incoming.notna()]
        if destination_column not in merged.obs.columns:
            dtype = "float64" if pd.api.types.is_numeric_dtype(incoming) else "object"
            fill = np.nan if dtype == "float64" else pd.NA
            merged.obs[destination_column] = pd.Series(
                fill, index=merged.obs_names, dtype=dtype
            )
        if isinstance(merged.obs[destination_column].dtype, pd.CategoricalDtype):
            merged.obs[destination_column] = merged.obs[
                destination_column
            ].astype(object)

        existing = merged.obs.loc[valid_cells, destination_column]
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
        merged.obs.loc[valid_cells, destination_column] = incoming_valid.to_numpy()
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
        score_matrix = score_matrix.toarray()
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


__all__ = [
    "CANONICAL_UNMEASURED_CHECKS", "CELLTYPE_PALETTE", "CONTEXT_GREY",
    "DARK_TEXT", "EXPRESSION_CMAP", "FOCUS_PALETTE", "MARKER_MODULES",
    "PROGRAM_CMAP", "REQUIRED_OBS_COLUMNS", "compute_program_scores",
    "configure_plot_style", "extract_marker_matrices",
    "marker_availability_table", "plot_focus_umap", "plot_full_umap",
    "plot_marker_dotplot", "plot_program_umap", "plot_spatial_celltypes",
    "plot_spatial_focus", "plot_spatial_programs", "save_figure",
    "select_representative_cores", "summarize_markers",
    "validate_xenium_metadata",
]
