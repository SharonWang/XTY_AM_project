"""Tests for categorical Stage 2 nearest-AT2 analysis and figures."""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts import xty_am_pipeline as pipeline


def _nearest_group_adata(*, two_cores: bool = False) -> ad.AnnData:
    """Create cores where MHCII-high AMs are visibly closer to AT2 cells."""
    coordinates = np.array(
        [
            [0.0, 0.0], [0.0, 1.0],
            [9.0, 0.0], [9.0, 1.0],
            [10.0, 0.0], [10.0, 1.0],
        ]
    )
    celltypes = ["AM", "AM", "AM", "AM", "AT2", "AT2"]
    groups = ["MHCIIlo", "MHCIIlo", "MHCIIhi", "MHCIIhi", pd.NA, pd.NA]
    blocks = [("D1", "C1", "A", 40.0)]
    if two_cores:
        blocks.append(("D2", "C2", "B", 60.0))
    obs_parts = []
    spatial_parts = []
    for block_index, (donor, core, tissue, age) in enumerate(blocks):
        obs_parts.append(
            pd.DataFrame(
                {
                    "donor_id": donor,
                    "core_id": core,
                    "tissue_annotation": tissue,
                    "age": age,
                    "CellType_refined": celltypes,
                    "MHCII_group": groups,
                },
                index=[f"{core}_cell_{i}" for i in range(len(celltypes))],
            )
        )
        spatial_parts.append(coordinates + np.array([20.0 * block_index, 0.0]))
    obs = pd.concat(obs_parts)
    return ad.AnnData(
        X=np.zeros((len(obs), 1)),
        obs=obs,
        var=pd.DataFrame(index=["dummy"]),
        obsm={"spatial": np.vstack(spatial_parts)},
    )


def test_nearest_mhcii_groups_has_positive_effect_when_high_is_closer():
    """Reversing the low-minus-high sign or group counts must fail."""
    adata = _nearest_group_adata()

    first = pipeline.calculate_stage2_nearest_mhcii_groups_by_core(
        adata,
        min_group_cells=2,
        min_at2_cells=2,
        n_permutations=99,
        random_state=17,
    )
    second = pipeline.calculate_stage2_nearest_mhcii_groups_by_core(
        adata,
        min_group_cells=2,
        min_at2_cells=2,
        n_permutations=99,
        random_state=17,
    )

    row = first.iloc[0]
    assert row["effect"] > 0
    assert row["effect"] == pytest.approx(
        row["mhcii_low_distance"] - row["mhcii_high_distance"]
    )
    assert row["distance_difference_high_minus_low"] == pytest.approx(
        -row["effect"]
    )
    assert row["n_mhcii_low_used"] == 2
    assert row["n_mhcii_high_used"] == 2
    assert row["permutation_p_value"] == second.iloc[0]["permutation_p_value"]


def test_nearest_mhcii_worker_restores_groups_in_multicore_runner():
    """Dropping MHCII groups from lightweight workers must not empty results."""
    adata = _nearest_group_adata(two_cores=True)
    group_lookup = adata.obs["MHCII_group"].copy()

    result = pipeline.run_stage2_multicore(
        pipeline.calculate_stage2_nearest_mhcii_groups_worker,
        adata,
        n_jobs=1,
        random_state=23,
        group_lookup=group_lookup,
        group_col="MHCII_group",
        min_group_cells=2,
        min_at2_cells=2,
        n_permutations=19,
    )

    assert result["core_id"].tolist() == ["C1", "C2"]
    assert (result["effect"] > 0).all()
    assert result["permutation_p_value"].notna().all()


def test_nearest_mhcii_groups_rejects_inconsistent_core_tissue():
    """Silently taking the first of several tissue labels must fail."""
    adata = _nearest_group_adata()
    adata.obs.loc[adata.obs.index[0], "tissue_annotation"] = "B"

    with pytest.raises(ValueError, match="multiple tissue"):
        pipeline.calculate_stage2_nearest_mhcii_groups_by_core(
            adata,
            min_group_cells=2,
            min_at2_cells=2,
        )


def test_stage2_nearest_extremes_uses_donors_for_inference(tmp_path):
    """Treating six cores as six inferential replicates must fail."""
    results = pd.DataFrame(
        {
            "donor_id": ["D1", "D1", "D2", "D2", "D3", "D3"],
            "core_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "method": "nearest_at2_mhcii_group",
            "mhcii_low_distance": [12.0, 10.0, 9.0, 11.0, 15.0, 13.0],
            "mhcii_high_distance": [7.0, 6.0, 5.0, 6.0, 8.0, 7.0],
        }
    )
    output = tmp_path / "nearest.png"

    plotted = pipeline.plot_stage2_nearest_extremes(
        results,
        output_file=output,
        show=False,
        random_state=31,
    )

    assert output.exists()
    assert len(plotted["donor_summary"]) == 3
    assert plotted["statistics"].loc[0, "n_donors"] == 3
    assert plotted["statistics"].loc[0, "n_cores"] == 6
    assert plotted["statistics"].loc[
        0, "proximity_effect_low_minus_high"
    ] > 0
    plt.close(plotted["fig"])


def test_stage2_radius_caterpillar_fisher_averages_within_donor(tmp_path):
    """Arithmetic averaging correlations or treating cores as donors must fail."""
    radius_results = pd.DataFrame(
        {
            "donor_id": ["D1", "D1", "D2", "D2", "D3", "D3"],
            "core_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "scale": ["50 um"] * 6,
            "effect": [0.2, 0.6, 0.1, 0.3, -0.2, 0.2],
            "tissue_annotation": ["A", "B", "A", "V", "B", "V"],
        }
    )
    donor_metadata = pd.DataFrame(
        {"donor_id": ["D1", "D2", "D3"], "age": [60, 40, 50]}
    )
    output = tmp_path / "radius.png"

    plotted = pipeline.plot_stage2_radius_50_caterpillar(
        radius_results,
        donor_metadata=donor_metadata,
        output_file=output,
        random_state=37,
    )

    assert output.exists()
    summary = plotted["donor_summary"]
    assert summary["donor_id"].tolist() == ["D2", "D3", "D1"]
    expected_d1 = np.tanh(np.mean(np.arctanh([0.2, 0.6])))
    assert summary.set_index("donor_id").loc["D1", "rho"] == pytest.approx(
        expected_d1
    )
    assert plotted["test_summary"].loc[0, "n_donors"] == 3
    plt.close(plotted["fig"])


def test_stage2_radius_caterpillar_rejects_duplicate_core_rows():
    """Duplicated core estimates must not silently receive extra weight."""
    rows = pd.DataFrame(
        {
            "donor_id": ["D1", "D2", "D3"],
            "core_id": ["C1", "C2", "C3"],
            "scale": [50, 50, 50],
            "effect": [0.2, 0.3, 0.4],
            "tissue_annotation": ["A", "B", "V"],
            "age": [40, 50, 60],
        }
    )
    duplicated = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="one row per donor/core"):
        pipeline.plot_stage2_radius_50_caterpillar(duplicated)
