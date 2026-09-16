"""Behavioral tests for user-supplied Xenium utility functions."""

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


def test_cluster_expression_summary_reports_sparse_group_statistics():
    """Wrong sparse means, detection percentages, or group sizes must fail."""
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            [[1.0, 0.0], [3.0, 2.0], [0.0, 4.0]],
        ),
        obs=pd.DataFrame(
            {"cluster": ["A", "A", "B"]},
            index=["c1", "c2", "c3"],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )

    result = pipeline.cluster_expression_summary(
        adata,
        genes=["G1", "missing", "G2"],
        groupby="cluster",
    )

    a_g1 = result.query("cluster == 'A' and gene == 'G1'").iloc[0]
    b_g2 = result.query("cluster == 'B' and gene == 'G2'").iloc[0]
    assert a_g1["mean_expression"] == pytest.approx(2.0)
    assert a_g1["pct_expressing"] == pytest.approx(100.0)
    assert a_g1["n_cells"] == 2
    assert b_g2["mean_expression"] == pytest.approx(4.0)
    assert b_g2["pct_expressing"] == pytest.approx(100.0)


def test_add_human_gene_name_collapses_orthologues_and_records_fallback():
    """Dropping one-to-many orthologues or fallback provenance must fail."""
    de = pd.DataFrame({"names": ["H2-Ab1", "Marco", "Unknown"]})
    ortholog = pd.DataFrame(
        {
            "external_gene_name": ["HLA-DRA", "HLA-DPA1", "MARCO"],
            "mmusculus_homolog_associated_gene_name": [
                "H2-Ab1", "H2-Ab1", "Marco",
            ],
        }
    )

    result = pipeline.add_human_gene_name(de, ortholog_table=ortholog)

    assert result["Human_gene_name"].astype(str).tolist() == [
        "HLA-DPA1; HLA-DRA",
        "MARCO",
        "UNKNOWN",
    ]
    assert result["Human_gene_mapping_method"].astype(str).tolist() == [
        "orthologue_table",
        "orthologue_table",
        "uppercase_fallback",
    ]


def test_gene_detection_by_group_uses_raw_and_rejects_no_panel_overlap():
    """Using transformed values instead of raw detection must fail."""
    adata = ad.AnnData(
        X=np.ones((3, 1), dtype=float),
        obs=pd.DataFrame(
            {"cluster": ["A", "A", "B"]},
            index=["c1", "c2", "c3"],
        ),
        var=pd.DataFrame(index=["G1"]),
    )
    raw_source = ad.AnnData(
        X=sparse.csr_matrix([[0], [2], [0]]),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    adata.raw = raw_source

    result = pipeline.gene_detection_by_group(
        adata,
        genes=["G1"],
        groupby="cluster",
        use_raw=True,
    )

    detected = result.set_index("group")["pct_expressing"].to_dict()
    assert detected == {"A": 50.0, "B": 0.0}
    with pytest.raises(ValueError, match="None of the requested genes"):
        pipeline.gene_detection_by_group(
            adata,
            genes=["absent"],
            groupby="cluster",
        )


def test_merge_obs_to_main_aligns_names_and_blocks_conflicts():
    """Row-position merging or silent conflict replacement must fail."""
    main = ad.AnnData(
        X=np.zeros((3, 1)),
        obs=pd.DataFrame(index=["c1", "c2", "c3"]),
        var=pd.DataFrame(index=["G1"]),
    )
    subset = ad.AnnData(
        X=np.zeros((2, 1)),
        obs=pd.DataFrame(
            {"state": ["high", "low"]},
            index=["c3", "c1"],
        ),
        var=pd.DataFrame(index=["G1"]),
    )

    merged, report = pipeline.merge_obs_to_main(
        main,
        subset,
        columns={"state": "MHCII_group"},
        copy=True,
        source_name="AM",
    )

    assert merged.obs["MHCII_group"].tolist() == ["low", pd.NA, "high"]
    assert report.loc[0, "n_matched_cells"] == 2
    assert "MHCII_group" not in main.obs

    merged.obs.loc["c1", "MHCII_group"] = "different"
    with pytest.raises(ValueError, match="conflicting values"):
        pipeline.merge_obs_to_main(
            merged,
            subset,
            columns={"state": "MHCII_group"},
            overwrite=False,
        )


def test_assign_mhcii_single_signature_keeps_discordant_cells_ambiguous():
    """Collapsing score/detection discordance into hard states must fail."""
    adata = ad.AnnData(
        X=np.asarray(
            [
                [10, 10, 10, 1],
                [0, 0, 0, 0],
                [10, 10, 10, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        ),
        obs=pd.DataFrame(index=["hi", "lo", "score_only", "core_only"]),
        var=pd.DataFrame(index=["SIG1", "SIG2", "SIG3", "CD74"]),
    )
    adata.raw = adata.copy()

    result = pipeline.assign_mhcii_single_signature(
        adata,
        signature_genes=["sig1", "SIG2", "sig3"],
        core_genes=("CD74", "HLA-DQB1"),
        use_raw=False,
        copy=True,
    )

    assert result.obs["MHCII_group"].astype(str).tolist() == [
        "MHCIIhi",
        "MHCIIlo",
        "Ambiguous",
        "Ambiguous",
    ]
    assert result.obs["MHCII_core_n_detected"].tolist() == [1, 0, 0, 1]
    audit = result.uns["MHCII_single_signature"]
    assert audit["available_signature"] == ["SIG1", "SIG2", "SIG3"]
    assert audit["missing_core"] == ["HLA-DQB1"]


def test_plot_metadata_summary_returns_core_and_donor_units():
    """Counting cells instead of unique cores or donors must fail."""
    metadata = pd.DataFrame(
        {
            "donor_id": ["D1", "D1", "D1", "D2", "D3"],
            "core_id": ["D1.c1", "D1.c1", "D1.c2", "D2.c1", "D3.c1"],
            "tma_id": ["TMA1", "TMA1", "TMA1", "TMA2", "TMA2"],
            "age": [50, 50, 50, 60, 70],
            "pmi": [8, 8, 8, 10, 12],
            "sex": ["F", "F", "F", "M", "F"],
            "tissue_annotation": ["A", "A", "B", "V", None],
        }
    )

    output = pipeline.plot_metadata_summary(metadata)

    assert len(output["core_meta"]) == 4
    assert len(output["donor_meta"]) == 3
    assert output["cohort_summary"].set_index("Metric").loc["Cores", "Value"] == 4
    assert output["donor_inconsistency"].empty
    plt.close(output["fig"])


def test_plot_macrophage_pct_uses_paired_donor_summaries_and_fdr():
    """Treating multiple cores as independent abundance replicates must fail."""
    abundance = pd.DataFrame(
        {
            "donor_id": [
                "D1", "D1", "D1", "D2", "D2", "D3", "D3",
            ],
            "tissue_annotation": ["A", "A", "B", "A", "B", "A", "B"],
            "CellType": ["AM"] * 7,
            "percent_of_all_cells": [10, 14, 5, 20, 10, 30, 15],
        }
    )

    fig, axes, donor_summary, statistics = (
        pipeline.plot_macrophage_pct_by_tissue(
            abundance,
            tissue_order=("A", "B"),
            macrophage_order=("AM",),
            comparisons=(("A", "B"),),
        )
    )

    d1_a = donor_summary.query(
        "donor_id == 'D1' and tissue_annotation == 'A'"
    )["percent_of_all_cells"].iloc[0]
    assert d1_a == pytest.approx(12.0)
    assert statistics.loc[0, "n_paired_donors"] == 3
    assert 0 <= statistics.loc[0, "p_adj"] <= 1
    assert len(axes) == 1
    plt.close(fig)
