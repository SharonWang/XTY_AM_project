"""Behavioral tests for the reusable Xenium analysis functions."""

from __future__ import annotations

import inspect

import anndata as ad
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts import xty_am_pipeline as pipeline


def test_select_representative_cores_keeps_complete_alveolar_fields():
    """A missing target cell type must prevent a core entering spatial review."""
    obs = pd.DataFrame(
        {
            "core_id": (
                ["c1"] * 6 + ["c2"] * 9 + ["c3"] * 12
                + ["c4"] * 15 + ["c5"] * 6
            ),
            "donor_id": (
                ["d1"] * 6 + ["d2"] * 9 + ["d3"] * 12
                + ["d4"] * 15 + ["d5"] * 6
            ),
            "tissue_annotation": ["A"] * 42 + ["V"] * 6,
            "celltype_final": (
                ["Macrophages", "AT1", "AT2", "B", "B", "B"]
                + ["Macrophages", "AT1", "AT2"] * 3
                + ["Macrophages", "AT1", "AT2"] * 4
                + ["Macrophages", "AT1", "AT2"] * 5
                + ["Macrophages", "AT1", "AT2", "B", "B", "B"]
            ),
        }
    )

    selected, summary = pipeline.select_representative_cores(
        obs,
        n_cores=3,
        tissue_code="A",
        required_celltypes=("Macrophages", "AT1", "AT2"),
    )

    assert selected == ["c1", "c3", "c4"]
    assert set(summary.loc[selected, "tissue_annotation"]) == {"A"}
    assert (summary.loc[selected, ["Macrophages", "AT1", "AT2"]] > 0).all().all()


def test_extract_marker_matrices_preserves_gene_order_and_zero_pattern():
    """Raw columns must follow requested genes rather than raw storage order."""
    raw = sparse.csr_matrix(
        np.array([[0, 3, 0], [2, 0, 1], [0, 1, 4]], dtype=np.float32)
    )
    transformed = raw.copy()
    transformed.data = np.log1p(transformed.data)
    adata = ad.AnnData(
        X=transformed,
        obs=pd.DataFrame(index=["a", "b", "c"]),
        var=pd.DataFrame(index=["APOE", "AGER", "MARCO"]),
    )
    adata.raw = ad.AnnData(X=raw, obs=adata.obs.copy(), var=adata.var.copy())

    log_values, raw_values, concordance = pipeline.extract_marker_matrices(
        adata,
        row_indices=np.array([0, 2]),
        genes=("MARCO", "APOE"),
    )

    np.testing.assert_array_equal(raw_values, np.array([[0, 0], [4, 0]]))
    np.testing.assert_allclose(log_values, np.log1p(raw_values))
    assert concordance == 1.0


def test_marker_summary_gives_each_donor_equal_weight():
    """A donor with many cells must not dominate the displayed marker mean."""
    obs = pd.DataFrame(
        {
            "donor_id": ["D1"] * 10 + ["D2"],
            "core_id": ["D1.c1"] * 10 + ["D2.c1"],
            "celltype_final": ["Macrophages"] * 11,
        }
    )
    expression = np.array([[10.0]] * 10 + [[0.0]])
    raw_counts = np.array([[1.0]] * 10 + [[0.0]])

    aggregate, by_core, by_donor = pipeline.summarize_markers(
        expression,
        raw_counts,
        obs,
        genes=("MARCO",),
        marker_modules={"AM evidence": ("MARCO",)},
        groups=("Macrophages",),
    )

    row = aggregate.iloc[0]
    assert row["mean_log_normalized_expression"] == pytest.approx(5.0)
    assert row["fraction_detected_raw_gt_0"] == pytest.approx(0.5)
    assert row["n_donors"] == 2
    assert len(by_core) == 2
    assert len(by_donor) == 2


def test_program_scores_retain_cell_donor_and_core_provenance():
    """Per-cell scores must remain traceable to donor and spatial core."""
    expression = pd.DataFrame(
        {
            "MARCO": [1.0, 2.0, 10.0, 11.0],
            "APOE": [2.0, 3.0, 9.0, 10.0],
            "LYVE1": [8.0, 7.0, 1.0, 0.0],
        },
        index=["a", "b", "c", "d"],
    )
    obs = pd.DataFrame(
        {
            "donor_id": ["D1", "D1", "D2", "D2"],
            "core_id": ["D1.c1", "D1.c1", "D2.c1", "D2.c1"],
            "celltype_final": ["Macrophages"] * 4,
        },
        index=expression.index,
    )

    cells, by_core, by_donor = pipeline.compute_program_scores(
        expression,
        obs,
        program_modules={
            "AM evidence": ("MARCO", "APOE"),
            "IM alternative": ("LYVE1",),
        },
        candidate_label="Macrophages",
    )

    assert {"donor_id", "core_id", "AM evidence", "IM alternative"}.issubset(cells.columns)
    assert set(by_core["core_id"]) == {"D1.c1", "D2.c1"}
    assert set(by_donor["donor_id"]) == {"D1", "D2"}
    assert cells.loc["d", "AM evidence"] > cells.loc["a", "AM evidence"]


def test_plot_full_umap_returns_figure_and_plots_every_cell():
    """The overview helper must not silently subsample the deposited UMAP."""
    umap = np.array([[0, 0], [1, 1], [2, 0]], dtype=float)
    labels = np.array(["AT1", "AT2", "Macrophages"])

    fig = pipeline.plot_full_umap(
        umap,
        labels,
        palette=pipeline.CELLTYPE_PALETTE,
        title="Test atlas",
    )

    plotted = sum(len(collection.get_offsets()) for collection in fig.axes[0].collections)
    assert plotted == 3
    plt.close(fig)


def test_specialized_plot_helpers_return_expected_facets():
    """Removing a focus or program facet must break the plotting contract."""
    coordinates = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float)
    labels = np.array(["Macrophages", "AT1", "AT2", "B"])
    obs = pd.DataFrame(
        {
            "core_id": ["c1", "c1", "c2", "c2"],
            "celltype_final": labels,
        },
        index=["a", "b", "c", "d"],
    )
    scores = pd.DataFrame(
        {"AM evidence": [-1.0], "IM alternative": [0.5]},
        index=["a"],
    )

    focus_umap = pipeline.plot_focus_umap(coordinates, labels)
    program_umap = pipeline.plot_program_umap(
        coordinates,
        candidate_indices=np.array([0]),
        program_scores=scores,
    )
    spatial_focus = pipeline.plot_spatial_focus(
        coordinates,
        obs,
        core_ids=("c1", "c2"),
    )
    spatial_celltypes = pipeline.plot_spatial_celltypes(
        coordinates,
        obs,
        core_ids=("c1", "c2"),
    )
    spatial_program = pipeline.plot_spatial_programs(
        coordinates,
        obs,
        core_ids=("c1", "c2"),
        candidate_global_indices=np.array([0]),
        program_scores=scores,
    )

    assert len(focus_umap.axes) == 1
    assert len(program_umap.axes) >= 2
    assert len(spatial_focus.axes) == 2
    assert len(spatial_celltypes.axes) == 2
    assert len(spatial_program.axes) >= 4
    focus_legend = {text.get_text() for text in spatial_focus.legends[0].texts}
    celltype_legend = {
        text.get_text() for text in spatial_celltypes.legends[0].texts
    }
    assert focus_legend == {"Macrophages", "AT1", "AT2"}
    assert celltype_legend == {"Macrophages", "AT1", "AT2", "B"}
    for figure in (
        focus_umap,
        program_umap,
        spatial_celltypes,
        spatial_focus,
        spatial_program,
    ):
        plt.close(figure)


def test_public_functions_have_detailed_numpy_docstrings():
    """Removing parameter or return documentation must break the API contract."""
    public_functions = [
        pipeline.validate_xenium_metadata,
        pipeline.select_representative_cores,
        pipeline.extract_marker_matrices,
        pipeline.summarize_markers,
        pipeline.compute_program_scores,
        pipeline.save_figure,
        pipeline.plot_full_umap,
        pipeline.plot_focus_umap,
        pipeline.plot_marker_dotplot,
        pipeline.plot_program_umap,
        pipeline.plot_spatial_celltypes,
        pipeline.plot_spatial_focus,
        pipeline.plot_spatial_programs,
        pipeline.cluster_expression_summary,
        pipeline.add_human_gene_name,
        pipeline.gene_detection_by_group,
        pipeline.merge_obs_to_main,
        pipeline.assign_mhcii_single_signature,
        pipeline.plot_metadata_summary,
        pipeline.plot_macrophage_pct_by_tissue,
    ]

    for function in public_functions:
        docstring = inspect.getdoc(function)
        assert docstring, function.__name__
        assert "Parameters" in docstring, function.__name__
        assert "Returns" in docstring, function.__name__
