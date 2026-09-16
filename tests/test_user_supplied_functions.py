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


def test_merge_obs_to_main_is_transactional_across_multiple_columns():
    """A late conflict must not leave earlier columns partially written."""
    main = ad.AnnData(
        X=np.zeros((2, 1)),
        obs=pd.DataFrame(
            {"existing": ["keep", pd.NA]},
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["G1"]),
    )
    subset = ad.AnnData(
        X=np.zeros((2, 1)),
        obs=pd.DataFrame(
            {"new_value": [1, 2], "conflict": ["replace", "ok"]},
            index=["c1", "c2"],
        ),
        var=pd.DataFrame(index=["G1"]),
    )

    with pytest.raises(ValueError, match="conflicting values"):
        pipeline.merge_obs_to_main(
            main,
            subset,
            columns={"new_value": "new_value", "conflict": "existing"},
            copy=False,
            overwrite=False,
        )

    assert "new_value" not in main.obs
    assert main.obs["existing"].tolist()[0] == "keep"


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


def test_assign_mhcii_single_signature_preserves_sparse_scoring(monkeypatch):
    """Densifying the full sparse signature matrix must fail."""
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            [[10, 10, 10, 1], [0, 0, 0, 0], [8, 8, 8, 1]],
            dtype=float,
        ),
        obs=pd.DataFrame(index=["high1", "low", "high2"]),
        var=pd.DataFrame(index=["SIG1", "SIG2", "SIG3", "CD74"]),
    )
    adata.raw = adata.copy()

    def reject_dense_conversion(*args, **kwargs):
        raise AssertionError("sparse signature matrix was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_dense_conversion)
    result = pipeline.assign_mhcii_single_signature(
        adata,
        signature_genes=["SIG1", "SIG2", "SIG3"],
        use_raw=False,
    )

    assert result.obs["MHCIIhi_score"].notna().all()


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


def test_declared_public_api_contains_new_functions_and_palettes():
    """Forgetting new utilities in __all__ must fail API introspection."""
    required = {
        "cluster_expression_summary", "add_human_gene_name",
        "gene_detection_by_group", "merge_obs_to_main",
        "assign_mhcii_single_signature", "plot_metadata_summary",
        "plot_macrophage_pct_by_tissue", "TISSUE_PALETTE", "SEX_PALETTE",
        "TMA_PALETTE", "MACROPHAGE_SUBTYPE_PALETTE",
        "plot_am_at2_pct_by_donor", "plot_am_at2_spatial",
        "test_continuous_mhcii_at2_proximity",
        "calculate_nhood_enrichment_by_core", "summarize_nhood_by_donor",
        "plot_nhood_enrichment_donor_tissue",
        "assign_balanced_mhcii_score_groups",
        "calculate_knn_niche_continuum", "calculate_radius_niche_continuum",
        "plot_radius_core_correlations", "plot_knn_niche_continuum",
    }
    assert required.issubset(set(pipeline.__all__))


def _small_spatial_adata():
    """Create a tiny two-core AM/AT2 object for local smoke tests."""
    obs = pd.DataFrame(
        {
            "CellType_refined": ["AM", "AM", "AM", "AT2"] * 2,
            "MHCII_group": ["MHCIIlo", "Ambiguous", "MHCIIhi", "Unassigned"] * 2,
            "MHCIIhi_score": [0.0, 1.0, 2.0, np.nan] * 2,
            "core_id": ["C1"] * 4 + ["C2"] * 4,
            "donor_id": ["D1"] * 4 + ["D2"] * 4,
            "tissue_annotation": ["A"] * 4 + ["B"] * 4,
        },
        index=[f"cell_{index}" for index in range(8)],
    )
    adata = ad.AnnData(
        X=np.zeros((8, 1)),
        obs=obs,
        var=pd.DataFrame(index=["G1"]),
    )
    adata.obsm["spatial"] = np.asarray(
        [[0, 0], [1, 0], [2, 0], [3, 0],
         [0, 0], [1, 0], [2, 0], [3, 0]],
        dtype=float,
    )
    return adata


def test_balanced_groups_and_spatial_continuum_use_core_boundaries():
    """Balanced tails and spatial neighbors must be computed within cores."""
    adata = _small_spatial_adata()
    grouped, summary, cutoffs = pipeline.assign_balanced_mhcii_score_groups(
        adata,
        n_per_tail=1,
        min_am_cells=3,
        copy=True,
    )
    assert set(cutoffs["n_MHCIIhi"]) == {1}
    assert set(cutoffs["n_MHCIIlo"]) == {1}
    assert summary["n_cells"].sum() == 6

    knn = pipeline.calculate_knn_niche_continuum(
        grouped,
        target_types=["AM", "AT2"],
        k_neighbors=2,
        min_query_cells=3,
    )
    assert (knn["neighbor_long"]["cell_id"] !=
            knn["neighbor_long"]["neighbor_cell_id"]).all()
    assert knn["core_correlations"]["core_id"].nunique() == 2

    radius = pipeline.calculate_radius_niche_continuum(
        grouped,
        target_types=["AM", "AT2"],
        radii=(1.5,),
        min_query_cells=3,
    )
    assert set(radius["neighborhood_comp"]["core_id"]) == {"C1", "C2"}

    proximity = pipeline.test_continuous_mhcii_at2_proximity(
        grouped,
        min_am=3,
        min_at2=1,
        n_permutations=9,
    )
    assert set(proximity["core_results"]["core_id"]) == {"C1", "C2"}
    assert proximity["cell_distances"]["nearest_AT2_distance"].ge(0).all()


def test_core_summary_and_new_plotting_functions_smoke():
    """Small local data must exercise the supplied plotting return contracts."""
    adata = _small_spatial_adata()
    fig1, axes1, wide, long = pipeline.plot_am_at2_pct_by_donor(
        adata,
        annotate_cores=False,
    )
    assert len(wide) == 2
    assert set(long["CellType"]) == {"AM", "AT2"}
    plt.close(fig1)

    fig2, axes2, spatial = pipeline.plot_am_at2_spatial(
        adata,
        "C1",
        am_groups=("MHCIIhi", "MHCIIlo"),
    )
    assert len(spatial) == 4
    assert spatial["is_selected_AM"].sum() == 2
    assert spatial["is_AT2"].sum() == 1
    plt.close(fig2)

    core_results = pd.DataFrame(
        {
            "core_id": ["C1", "C2"],
            "donor_id": ["D1", "D2"],
            "tissue_annotation": ["A", "B"],
            "neighbor_celltype": ["MHCIIhi AM", "MHCIIhi AM"],
            "radius": [50, 50],
            "n_focus": [10, 12],
            "n_neighbor": [4, 6],
            "enrichment_zscore": [1.0, -0.5],
        }
    )
    summary = pipeline.summarize_nhood_by_donor(core_results)
    assert summary["n_cores"].tolist() == [1, 1]
    fig_nhood_1, fig_nhood_2 = pipeline.plot_nhood_enrichment_donor_tissue(
        summary,
        tissue_order=("A", "B"),
    )
    plt.close(fig_nhood_1)
    plt.close(fig_nhood_2)

    tissue_tests = pd.DataFrame(
        {
            "tissue_annotation": ["A", "B"],
            "neighbor_type": ["AT2", "AT2"],
            "median_rho": [0.4, -0.2],
            "FDR": [0.01, 0.2],
        }
    )
    fig3, axes3 = pipeline.plot_knn_niche_continuum(
        tissue_tests,
        tissue_order=("A", "B"),
    )
    assert len(axes3) == 2
    plt.close(fig3)

    radius_core = pd.DataFrame(
        {
            "core_id": ["C1", "C2"],
            "tissue_annotation": ["A", "B"],
            "neighbor_type": ["AT2", "AT2"],
            "radius": [50.0, 50.0],
            "rho": [0.4, -0.2],
        }
    )
    fig4, axes4, plotted = pipeline.plot_radius_core_correlations(
        radius_core,
        top_n=1,
    )
    assert len(plotted) == 2
    plt.close(fig4)


def test_squidpy_dependency_is_deferred_until_enrichment_call():
    """Importing the pipeline must not require Squidpy for unrelated steps."""
    if pipeline.sq.__class__.__name__ != "_MissingSquidpy":
        pytest.skip("Squidpy is installed in this environment")
    with pytest.raises(ImportError, match="requires squidpy"):
        pipeline.calculate_nhood_enrichment_by_core(
            _small_spatial_adata(),
            cluster_key="CellType_refined",
            focus_label="AT2",
            min_focus_cells=1,
            min_neighbor_cells=1,
            n_perms=2,
        )
