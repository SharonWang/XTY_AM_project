"""Tests for Stage 2 nearest-score extremes and pooled donor plotting."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts import xty_am_pipeline as pipeline


def _nearest_extreme_core(scores=None, tissue="A"):
    """Return one core where high-score AMs are nearest to AT2 cells."""
    if scores is None:
        scores = np.arange(8, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n_am = len(scores)
    am_x = np.linspace(20.0, 1.0, n_am)
    coordinates = np.column_stack(
        [
            np.r_[am_x, [0.0, 0.0]],
            np.r_[np.zeros(n_am), [0.0, 1.0]],
        ]
    )
    obs = pd.DataFrame(
        {
            "donor_id": "D1",
            "core_id": "C1",
            "tissue_annotation": tissue,
            "age": 55.0,
            "CellType_refined": ["AM"] * n_am + ["AT2", "AT2"],
            "MHCIIhi_score": np.r_[scores, [np.nan, np.nan]],
        },
        index=[f"cell_{index}" for index in range(n_am + 2)],
    )
    return SimpleNamespace(obs=obs, obsm={"spatial": coordinates}, n_obs=len(obs))


def test_nearest_score_extremes_reports_positive_effect_when_high_is_closer():
    """Reversing the low-minus-high proximity effect or tail size must fail."""
    adata = _nearest_extreme_core()

    first = pipeline.calculate_stage2_nearest_extremes_by_core(
        adata,
        extreme_fraction=0.25,
        min_extreme_cells=2,
        min_at2_cells=2,
        n_permutations=99,
        random_state=41,
    )
    second = pipeline.calculate_stage2_nearest_extremes_by_core(
        adata,
        extreme_fraction=0.25,
        min_extreme_cells=2,
        min_at2_cells=2,
        n_permutations=99,
        random_state=41,
    )

    row = first.iloc[0]
    assert row["effect"] > 0
    assert row["effect"] == pytest.approx(
        row["mhcii_low_distance"] - row["mhcii_high_distance"]
    )
    assert row["n_extreme_per_group"] == 2
    assert row["permutation_p_value"] == second.iloc[0]["permutation_p_value"]


def test_nearest_score_extremes_does_not_expand_requested_tails():
    """Using the minimum count to enlarge a 25% tail must fail."""
    adata = _nearest_extreme_core(scores=np.arange(12, dtype=float))

    result = pipeline.calculate_stage2_nearest_extremes_by_core(
        adata,
        extreme_fraction=0.25,
        min_extreme_cells=5,
        min_at2_cells=1,
        n_permutations=0,
    )

    assert result.empty


def test_nearest_score_extremes_omits_ambiguous_boundary_ties():
    """Arbitrarily splitting equal scores across an extreme boundary must fail."""
    adata = _nearest_extreme_core(scores=[0, 1, 1, 1, 4, 5, 6, 7])

    result = pipeline.calculate_stage2_nearest_extremes_by_core(
        adata,
        extreme_fraction=0.25,
        min_extreme_cells=2,
        min_at2_cells=1,
        n_permutations=0,
    )

    assert result.empty


def _pooled_donor_summary():
    rows = []
    values = {
        "D1": {"A": (0.2, 0.4, 0.6), "B": (0.4, 0.6, 0.8)},
        "D2": {"A": (-0.1, 0.1, 0.3), "B": (0.1, 0.3, 0.5)},
        "D3": {"A": (0.0, 0.2, 0.4), "B": (0.2, 0.4, 0.6)},
    }
    methods = (
        ("Continuous MHCII kNN", 15),
        ("Continuous MHCII radius", 50),
        ("Continuous MHCII nearest AT2", 1),
    )
    for donor, tissues in values.items():
        for tissue, effects in tissues.items():
            for (method, scale), effect in zip(methods, effects):
                rows.append(
                    {
                        "donor_id": donor,
                        "tissue_annotation": tissue,
                        "method": method,
                        "scale": scale,
                        "donor_effect": effect,
                    }
                )
    return pd.DataFrame(rows)


def test_prepare_stage2_primary_pooled_uses_one_equal_weight_row_per_donor():
    """Leaving tissue rows as pseudoreplicates or using the wrong value column must fail."""
    result = pipeline.prepare_stage2_primary_pooled(_pooled_donor_summary())

    assert len(result) == 9
    assert not result.duplicated(["display_method", "donor_id"]).any()
    d1_knn = result.loc[
        result["display_method"].eq("kNN exposure")
        & result["donor_id"].eq("D1"),
        "effect",
    ].item()
    assert d1_knn == pytest.approx(0.3)


def test_prepare_stage2_primary_pooled_rejects_duplicate_tissue_rows():
    """Duplicated donor/tissue estimates must not silently receive extra weight."""
    data = _pooled_donor_summary()
    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate donor/tissue"):
        pipeline.prepare_stage2_primary_pooled(duplicated)


def test_plot_stage2_primary_pooled_returns_donor_level_tests(tmp_path):
    """Testing pooled tissue rows instead of three independent donors must fail."""
    output = tmp_path / "stage2_primary_pooled.png"

    fig, ax, plot_data, tests = pipeline.plot_stage2_primary_pooled(
        _pooled_donor_summary(),
        n_boot=200,
        random_state=43,
        output_file=output,
    )

    assert output.exists()
    assert len(plot_data) == 9
    assert tests["n_donors"].tolist() == [3, 3, 3]
    assert tests["q_value"].between(0, 1).all()
    plt.close(fig)
