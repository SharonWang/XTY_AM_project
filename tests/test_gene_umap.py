"""Tests for reusable continuous gene-expression UMAP plotting."""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts import xty_am_pipeline as pipeline


def _gene_umap_adata() -> ad.AnnData:
    """Create a small AnnData object with hand-checkable expression values."""
    adata = ad.AnnData(
        X=np.array(
            [
                [0.0, 4.0],
                [1.0, 0.0],
                [2.0, 2.0],
                [3.0, 0.0],
            ]
        ),
        obs=pd.DataFrame(index=[f"cell_{index}" for index in range(4)]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    adata.obsm["X_umap"] = np.array(
        [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]]
    )
    adata.layers["counts"] = sparse.csr_matrix(
        np.array(
            [
                [0.0, 8.0],
                [5.0, 0.0],
                [0.0, 4.0],
                [1.0, 0.0],
            ]
        )
    )
    return adata


def test_gene_umap_reports_expression_metadata_and_creates_parent(tmp_path):
    """Wrong cutoff counts, panel limits, or missing output directories must fail."""
    output = tmp_path / "nested" / "gene_umap.png"

    fig, axes, metadata = pipeline.plot_anndata_gene_umap(
        _gene_umap_adata(),
        genes=["G1", "G2"],
        color_quantiles=(0.0, 1.0),
        show_legend=True,
        save=output,
    )

    assert output.exists()
    assert len(axes) == 2
    summary = metadata.set_index("gene")
    assert summary.loc["G1", "n_expressing"] == 3
    assert summary.loc["G1", "fraction_expressing"] == pytest.approx(0.75)
    assert summary.loc["G1", "vmin"] == pytest.approx(1.0)
    assert summary.loc["G1", "vmax"] == pytest.approx(3.0)
    assert summary.loc["G2", "n_expressing"] == 2
    assert summary.loc["G2", "vmin"] == pytest.approx(2.0)
    assert summary.loc["G2", "vmax"] == pytest.approx(4.0)
    plt.close(fig)


def test_gene_umap_reads_sparse_layer_without_densifying_all_genes():
    """Reading X instead of the requested sparse layer must fail."""
    fig, axes, metadata = pipeline.plot_anndata_gene_umap(
        _gene_umap_adata(),
        genes="G1",
        layer="counts",
        color_quantiles=(0.0, 1.0),
    )

    row = metadata.iloc[0]
    assert len(axes) == 1
    assert row["expression_source"] == "layer:counts"
    assert row["n_expressing"] == 2
    assert row["vmin"] == pytest.approx(1.0)
    assert row["vmax"] == pytest.approx(5.0)
    plt.close(fig)


def test_gene_umap_rejects_gene_specific_limits_with_shared_colorbar():
    """A shared colorbar with different panel normalizations must fail."""
    with pytest.raises(ValueError, match="shared_color_scale"):
        pipeline.plot_anndata_gene_umap(
            _gene_umap_adata(),
            genes=["G1", "G2"],
            shared_color_scale=True,
            show_legend=True,
            vmin={"G1": 0.0, "G2": 1.0},
        )


def test_gene_umap_rejects_wrong_number_of_per_gene_limits():
    """An undersized vmin sequence must raise a clear validation error."""
    with pytest.raises(ValueError, match="vmin.*one value per gene"):
        pipeline.plot_anndata_gene_umap(
            _gene_umap_adata(),
            genes=["G1", "G2"],
            vmin=[0.0],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"color_quantiles": (0.9, 0.1)}, "color_quantiles"),
        ({"padding_fraction": -0.1}, "padding_fraction"),
        ({"dpi": 0}, "dpi"),
    ],
)
def test_gene_umap_rejects_invalid_plot_parameters(kwargs, message):
    """Invalid quantitative display settings must not produce a misleading plot."""
    with pytest.raises(ValueError, match=message):
        pipeline.plot_anndata_gene_umap(
            _gene_umap_adata(),
            genes=["G1"],
            **kwargs,
        )
