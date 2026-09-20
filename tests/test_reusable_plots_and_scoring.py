"""Tests for reusable UMAP plotting and dual-signature scoring."""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import pandas as pd
import pytest

from scripts import xty_am_pipeline as pipeline


def make_plot_adata() -> ad.AnnData:
    """Return a small AnnData object with deterministic UMAP groups."""
    adata = ad.AnnData(
        X=np.ones((6, 2), dtype=float),
        obs=pd.DataFrame(
            {"group": pd.Categorical(["A", "A", "A", "B", "B", "B"])},
            index=[f"cell_{index}" for index in range(6)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    adata.obsm["X_umap"] = np.array(
        [[0, 0], [0.5, 0.2], [1, 0], [3, 3], [3.5, 3.2], [4, 3]],
        dtype=float,
    )
    return adata


def make_expression_adata() -> ad.AnnData:
    """Return normalized-like expression values for signature tests."""
    return ad.AnnData(
        X=np.array(
            [
                [5, 4, 0, 0, 1, 2],
                [4, 5, 0, 0, 2, 1],
                [0, 0, 5, 4, 1, 2],
                [1, 1, 1, 1, 2, 2],
            ],
            dtype=float,
        ),
        obs=pd.DataFrame(index=[f"cell_{index}" for index in range(4)]),
        var=pd.DataFrame(index=["G1", "G2", "G3", "G4", "CTRL1", "CTRL2"]),
    )


def test_plot_anndata_group_umap_supports_highlight_and_split(tmp_path):
    """Highlight panels must keep the requested order and save successfully."""
    adata = make_plot_adata()
    output = tmp_path / "groups.pdf"
    fig, axes = pipeline.plot_anndata_group_umap(
        adata,
        group_col="group",
        split_by="group",
        split_categories=["A", "B"],
        palette={"A": "#E8928F", "B": "#7FAED2"},
        save=output,
    )
    assert len(axes) == 2
    assert output.exists()
    assert [axis.get_title() for axis in axes] == ["A", "B"]


def test_plot_anndata_group_umap_rejects_incomplete_palette():
    """Missing group colours must raise rather than silently miscolour cells."""
    with pytest.raises(KeyError, match="No colour supplied"):
        pipeline.plot_anndata_group_umap(
            make_plot_adata(), group_col="group", palette={"A": "#E8928F"}
        )


def test_score_two_signatures_assigns_groups_and_margin_ambiguity():
    """Scoring must annotate in place and expose the ambiguity category."""
    adata = make_expression_adata()
    result = pipeline.score_and_assign_two_signatures(
        adata,
        ["G1", "G2"],
        ["G3", "G4"],
        use_raw=False,
        scale=False,
        ambiguous=True,
        ambiguous_threshold=-100,
        min_score_difference=0.05,
        ctrl_size=1,
        n_bins=2,
        verbose=False,
    )
    assert result is adata
    assert {
        "Signature1_score", "Signature2_score", "Signature_group"
    }.issubset(adata.obs.columns)
    assert "Ambiguous" in adata.obs["Signature_group"].cat.categories


def test_score_two_signatures_rejects_raw_and_layer_together():
    """Conflicting expression sources must raise a focused error."""
    with pytest.raises(ValueError, match="either use_raw"):
        pipeline.score_and_assign_two_signatures(
            make_expression_adata(), ["G1"], ["G3"], use_raw=True, layer="counts"
        )
