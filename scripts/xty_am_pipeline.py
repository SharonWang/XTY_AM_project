"""Reusable functions for the human lung Xenium AM and AT2 analysis.

The module performs no analysis at import time. Notebooks select paths, load
data, and call these functions explicitly. The published Macrophages label is
treated as a broad candidate pool, not an automatic alveolar-macrophage call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse


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
