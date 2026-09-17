"""Tests for multitype Stage 1 spatial-association functions."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scripts import xty_am_pipeline as pipeline


def _stage1_adata(*, conflicting_core_metadata: bool = False) -> ad.AnnData:
    """Return two small spatial cores containing AM, AT2, and background cells."""
    celltypes = ["AM", "Unknown", "AT2", "Fibroblast", "AM", "AT2"]
    obs = pd.DataFrame(
        {
            "CellType_refined": celltypes * 2,
            "core_id": ["C1"] * 6 + ["C2"] * 6,
            "donor_id": ["D1"] * 6 + ["D2"] * 6,
            "tissue_annotation": ["A"] * 6 + ["B"] * 6,
        },
        index=[f"cell_{index}" for index in range(12)],
    )
    if conflicting_core_metadata:
        obs.loc["cell_1", "donor_id"] = "conflicting_donor"

    adata = ad.AnnData(
        X=np.zeros((12, 1), dtype=float),
        obs=obs,
        var=pd.DataFrame(index=["G1"]),
    )
    core_coordinates = np.asarray(
        [
            [0.0, 0.0],   # AM
            [1.0, 0.0],   # excluded label retained as spatial background
            [2.0, 0.0],   # AT2
            [6.0, 0.0],   # Fibroblast
            [2.0, 1.0],   # AM
            [0.0, 1.0],   # AT2
        ]
    )
    adata.obsm["spatial"] = np.vstack([core_coordinates, core_coordinates])
    return adata


def test_contact_enrichment_deduplicates_symmetric_focal_target_pairs():
    """Reversing an undirected contact pair must not create a duplicate test."""
    result = pipeline.calculate_multitype_nhood_enrichment_by_core(
        _stage1_adata(),
        focal_types=("AM", "AT2"),
        target_types=("AM", "AT2", "Fibroblast"),
        radii=(3,),
        min_focal_cells=1,
        min_target_cells=1,
        n_permutations=9,
    )

    c1 = result.loc[result["core_id"].eq("C1")]
    unordered_pairs = {
        tuple(sorted((row.focal_type, row.target_type)))
        for row in c1.itertuples()
    }
    assert unordered_pairs == {
        ("AM", "AT2"),
        ("AM", "Fibroblast"),
        ("AT2", "Fibroblast"),
    }
    assert len(c1) == len(unordered_pairs)
    assert c1["FDR_within_core"].between(0, 1).all()


def test_knn_keeps_excluded_labels_as_spatial_background():
    """Dropping excluded cells must not change who occupies the nearest position."""
    result = pipeline.calculate_multitype_knn_niche_by_core(
        _stage1_adata(),
        k_values=(1,),
        focal_types=("AM",),
        target_types=("AT2",),
        min_focal_cells=1,
        min_target_cells=1,
        n_permutations=9,
    )

    c1 = result.loc[result["core_id"].eq("C1")].iloc[0]
    assert c1["direction"] == "AM → AT2"
    assert c1["mean_total_neighbors"] == pytest.approx(1.0)
    assert c1["observed"] == pytest.approx(0.5)


def test_radius_and_nearest_distance_keep_directional_results_separate():
    """AM-to-AT2 and AT2-to-AM results must remain distinct directional tests."""
    kwargs = {
        "adata": _stage1_adata(),
        "focal_types": ("AM", "AT2"),
        "target_types": {"AM": ("AT2",), "AT2": ("AM",)},
        "min_focal_cells": 1,
        "min_target_cells": 1,
        "n_permutations": 9,
    }
    radius = pipeline.calculate_multitype_radius_niche_by_core(
        radii=(1.5,),
        **kwargs,
    )
    nearest = pipeline.calculate_multitype_nearest_distance_by_core(**kwargs)

    expected_directions = {"AM → AT2", "AT2 → AM"}
    assert set(radius["direction"]) == expected_directions
    assert set(nearest["direction"]) == expected_directions
    assert nearest["effect"].notna().all()
    assert nearest["median_distance"].ge(0).all()


def test_stage1_rejects_invalid_scales_permutations_and_core_metadata():
    """Invalid spatial settings or mixed-donor cores must fail before inference."""
    adata = _stage1_adata()
    with pytest.raises(ValueError, match="k_values"):
        pipeline.calculate_multitype_knn_niche_by_core(
            adata,
            k_values=(0,),
        )
    with pytest.raises(ValueError, match="n_permutations"):
        pipeline.calculate_multitype_nearest_distance_by_core(
            adata,
            n_permutations=0,
        )
    with pytest.raises(ValueError, match="multiple donor"):
        pipeline.calculate_multitype_nhood_enrichment_by_core(
            _stage1_adata(conflicting_core_metadata=True),
            radii=(3,),
            min_focal_cells=1,
            min_target_cells=1,
            n_permutations=9,
        )


def test_stage1_summary_uses_one_donor_tissue_effect_and_adjusts_targets():
    """Core rows must collapse within donor before tissue-level testing."""
    rows = []
    for donor, effects in {
        "D1": (1.0, 3.0),
        "D2": (1.0,),
        "D3": (2.0,),
        "D4": (1.5,),
        "D5": (2.5,),
    }.items():
        for index, effect in enumerate(effects):
            rows.append(
                {
                    "method": "Contact enrichment",
                    "direction": "AM ↔ AT2",
                    "focal_type": "AM",
                    "target_type": "AT2",
                    "scale_type": "radius_um",
                    "scale": 25.0,
                    "core_id": f"{donor}.C{index}",
                    "donor_id": donor,
                    "tissue_annotation": "A",
                    "effect": effect,
                    "zscore": effect / 2,
                    "n_focal": 10,
                    "n_target": 20,
                }
            )

    donor_summary, tissue_tests = (
        pipeline.summarize_stage1_by_donor_and_tissue(
            pd.DataFrame(rows),
            min_donors=5,
        )
    )

    d1 = donor_summary.loc[donor_summary["donor_id"].eq("D1")].iloc[0]
    assert d1["donor_effect"] == pytest.approx(2.0)
    assert d1["n_cores"] == 2
    assert tissue_tests.loc[0, "n_donors"] == 5
    assert 0 <= tissue_tests.loc[0, "FDR"] <= 1


def test_multitype_stage1_public_functions_have_numpy_docstrings():
    """Every new public API must document both inputs and returned objects."""
    functions = [
        pipeline.calculate_multitype_nhood_enrichment_by_core,
        pipeline.calculate_multitype_knn_niche_by_core,
        pipeline.calculate_multitype_radius_niche_by_core,
        pipeline.calculate_multitype_nearest_distance_by_core,
        pipeline.summarize_stage1_by_donor_and_tissue,
    ]
    for function in functions:
        docstring = function.__doc__ or ""
        assert "Parameters" in docstring, function.__name__
        assert "Returns" in docstring, function.__name__
