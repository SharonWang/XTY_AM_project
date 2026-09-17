"""Tests for Stage 1 figures and Stage 2 AM–AT2 spatial analyses."""

from __future__ import annotations

import inspect
import warnings

import anndata as ad
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts import xty_am_pipeline as pipeline


def _stage2_adata(*, two_cores: bool = False, isolated_am: bool = False) -> ad.AnnData:
    """Return a tiny core where higher-score AMs have greater AT2 exposure."""
    am_x = np.arange(6, dtype=float) * 10.0
    celltypes = ["AM"] * 6 + ["Fibroblast"] * 3 + ["AT2"] * 3
    coordinates = np.vstack(
        [
            np.column_stack([am_x, np.zeros(6)]),
            np.column_stack([am_x[:3], np.full(3, 0.5)]),
            np.column_stack([am_x[3:], np.full(3, 0.5)]),
        ]
    )
    scores = np.asarray([0, 1, 2, 3, 4, 5] + [np.nan] * 6, dtype=float)

    if isolated_am:
        celltypes.append("AM")
        coordinates = np.vstack([coordinates, [100.0, 100.0]])
        scores = np.append(scores, 100.0)

    obs = pd.DataFrame(
        {
            "CellType_refined": celltypes,
            "MHCIIhi_score": scores,
            "core_id": "C1",
            "donor_id": "D1",
            "tissue_annotation": "A",
        },
        index=[f"C1_cell_{index}" for index in range(len(celltypes))],
    )

    if two_cores:
        obs2 = obs.copy()
        obs2.index = [f"C2_cell_{index}" for index in range(len(obs2))]
        obs2["core_id"] = "C2"
        obs2["donor_id"] = "D2"
        obs = pd.concat([obs, obs2])
        coordinates = np.vstack([coordinates, coordinates])

    adata = ad.AnnData(
        X=np.zeros((len(obs), 1), dtype=float),
        obs=obs,
        var=pd.DataFrame(index=["G1"]),
    )
    adata.obsm["spatial"] = coordinates
    return adata


def _stage1_plot_tables():
    """Return minimal donor and tissue summaries accepted by both Stage 1 plots."""
    donor_summary = pd.DataFrame(
        {
            "method": ["Contact enrichment"] * 3,
            "direction": ["AM ↔ AT2"] * 3,
            "focal_type": ["AM"] * 3,
            "target_type": ["AT2"] * 3,
            "scale_type": ["radius_um"] * 3,
            "scale": [50.0] * 3,
            "donor_id": ["D1", "D2", "D3"],
            "tissue_annotation": ["A"] * 3,
            "donor_effect": [0.2, 0.4, 0.6],
        }
    )
    tissue_tests = pd.DataFrame(
        {
            "method": ["Contact enrichment"],
            "direction": ["AM ↔ AT2"],
            "focal_type": ["AM"],
            "target_type": ["AT2"],
            "scale_type": ["radius_um"],
            "scale": [50.0],
            "tissue_annotation": ["A"],
            "median_effect": [0.4],
            "p_value": [0.03],
            "FDR": [0.04],
            "n_donors": [3],
        }
    )
    return donor_summary, tissue_tests


def test_stage1_plot_functions_return_figures_and_plotted_tables():
    donor_summary, tissue_tests = _stage1_plot_tables()
    config_a = [
        {
            "candidates": ("Contact enrichment",),
            "scale": 50,
            "label": "Contact\n50 µm",
        }
    ]
    figures, axes, tables = pipeline.plot_stage1A_niche_dotmap(
        tissue_tests,
        focal_types=("AM",),
        tissue_order=("A",),
        method_config=config_a,
        top_n=1,
    )
    assert set(figures) == {"AM"}
    assert set(axes) == {"AM"}
    assert tables["AM"]["target_type"].astype(str).tolist() == ["AT2"]

    config_b = [
        {
            "candidates": ("Contact enrichment",),
            "scale": 50,
            "title": "Contact enrichment",
            "subtitle": "Cells within 50 µm",
        }
    ]
    fig, plot_axes, plot_data = pipeline.plot_stage1B_primary(
        donor_summary,
        tissue_tests,
        tissue_order=("A",),
        method_config=config_b,
    )
    assert fig is not None
    assert plot_axes.size == 2
    assert len(plot_data) == 3
    plt.close("all")


def test_stage1_plot_functions_validate_required_columns():
    with pytest.raises(KeyError, match="Missing tissue_tests columns"):
        pipeline.plot_stage1A_niche_dotmap(pd.DataFrame({"method": []}))
    with pytest.raises(KeyError, match="Missing donor_summary columns"):
        pipeline.plot_stage1B_primary(pd.DataFrame(), pd.DataFrame())


def test_stage2_continuous_and_nearest_effects_have_consistent_direction():
    adata = _stage2_adata()
    common = dict(
        adata=adata,
        min_am_cells=6,
        min_at2_cells=3,
        n_permutations=19,
        random_state=7,
    )
    knn = pipeline.calculate_stage2_knn_continuum_by_core(
        k_values=(1,),
        **common,
    )
    radius = pipeline.calculate_stage2_radius_continuum_by_core(
        radii=(1,),
        **common,
    )
    nearest = pipeline.calculate_stage2_nearest_at2_by_core(**common)

    assert knn.loc[0, "effect"] > 0
    assert radius.loc[0, "effect"] > 0
    assert nearest.loc[0, "effect"] > 0
    assert nearest.loc[0, "effect"] == pytest.approx(
        -nearest.loc[0, "rho_score_distance_raw"]
    )


def test_stage2_radius_excludes_ams_with_no_neighbors():
    result = pipeline.calculate_stage2_radius_continuum_by_core(
        _stage2_adata(isolated_am=True),
        radii=(1,),
        min_am_cells=6,
        min_at2_cells=3,
        min_analyzed_am_cells=5,
        n_permutations=9,
    )
    row = result.iloc[0]
    assert row["n_am"] == 7
    assert row["n_am_analyzed"] == 6
    assert row["n_am_without_neighbors"] == 1
    assert row["pct_am_without_neighbors"] == pytest.approx(100 / 7)


def test_stage2_balanced_extremes_are_equal_sized_and_positive():
    result = pipeline.calculate_stage2_balanced_extremes_by_core(
        _stage2_adata(),
        radii=(1,),
        extreme_fraction=1 / 3,
        min_am_cells=6,
        min_at2_cells=3,
        min_extreme_cells=2,
        n_permutations=19,
        random_state=9,
    )
    row = result.iloc[0]
    assert row["n_high"] == row["n_low"] == 2
    assert row["effect"] > 0
    assert row["mean_high_at2_fraction"] > row["mean_low_at2_fraction"]


def test_stage2_balanced_extremes_skip_ambiguous_tied_boundaries():
    adata = _stage2_adata()
    adata.obs.loc[adata.obs["CellType_refined"].eq("AM"), "MHCIIhi_score"] = [
        0, 0, 0, 1, 1, 1
    ]
    result = pipeline.calculate_stage2_balanced_extremes_by_core(
        adata,
        radii=(1,),
        extreme_fraction=1 / 3,
        min_am_cells=6,
        min_at2_cells=3,
        min_extreme_cells=2,
        n_permutations=9,
    )
    assert result.empty
    assert "effect" in result.columns


@pytest.mark.parametrize(
    ("function_name", "kwargs", "message"),
    [
        ("calculate_stage2_knn_continuum_by_core", {"k_values": (0,)}, "k_values"),
        ("calculate_stage2_radius_continuum_by_core", {"radii": (-1,)}, "radii"),
        ("calculate_stage2_nearest_at2_by_core", {"n_permutations": 0}, "n_permutations"),
        (
            "calculate_stage2_balanced_extremes_by_core",
            {"extreme_fraction": 0.75},
            "extreme_fraction",
        ),
    ],
)
def test_stage2_rejects_invalid_settings(function_name, kwargs, message):
    with pytest.raises(ValueError, match=message):
        getattr(pipeline, function_name)(_stage2_adata(), **kwargs)


def test_spatial_multicore_runner_is_reproducible_across_worker_counts():
    kwargs = dict(
        radii=(1,),
        min_am_cells=6,
        min_at2_cells=3,
        n_permutations=9,
        verbose=0,
    )
    serial = pipeline.run_spatial_function_multicore(
        pipeline.calculate_stage2_radius_continuum_by_core,
        _stage2_adata(two_cores=True),
        n_jobs=1,
        random_state=42,
        **kwargs,
    )
    parallel = pipeline.run_spatial_function_multicore(
        pipeline.calculate_stage2_radius_continuum_by_core,
        _stage2_adata(two_cores=True),
        n_jobs=2,
        random_state=42,
        **kwargs,
    )
    pd.testing.assert_frame_equal(
        serial.sort_values("core_id").reset_index(drop=True),
        parallel.sort_values("core_id").reset_index(drop=True),
    )
    with pytest.raises(ValueError, match="n_jobs"):
        pipeline.run_spatial_function_multicore(
            pipeline.calculate_stage2_radius_continuum_by_core,
            _stage2_adata(),
            n_jobs=0,
        )


def test_stage2_multicore_wrapper_matches_generic_runner():
    kwargs = dict(
        radii=(1,),
        min_am_cells=6,
        min_at2_cells=3,
        n_permutations=9,
        verbose=0,
    )
    generic = pipeline.run_spatial_function_multicore(
        pipeline.calculate_stage2_radius_continuum_by_core,
        _stage2_adata(two_cores=True),
        n_jobs=1,
        random_state=3,
        **kwargs,
    )
    wrapped = pipeline.run_stage2_multicore(
        pipeline.calculate_stage2_radius_continuum_by_core,
        _stage2_adata(two_cores=True),
        n_jobs=1,
        random_state=3,
        **kwargs,
    )
    pd.testing.assert_frame_equal(generic, wrapped)


def test_stage2_summary_uses_donors_as_replicates():
    rows = []
    for donor, effects in {
        "D1": (0.2, 0.6),
        "D2": (0.3,),
        "D3": (0.4,),
        "D4": (0.5,),
        "D5": (0.7,),
    }.items():
        for core_number, effect in enumerate(effects):
            rows.append(
                {
                    "method": "Continuous MHCII radius",
                    "direction": "MHCII score → AT2 exposure",
                    "effect_type": "correlation",
                    "scale_type": "radius_um",
                    "scale": 25.0,
                    "core_id": f"{donor}.C{core_number}",
                    "donor_id": donor,
                    "tissue_annotation": "A",
                    "effect": effect,
                    "zscore": effect * 2,
                    "n_am": 10,
                    "n_at2": 20,
                    "n_am_analyzed": 10,
                }
            )

    donor_summary, tissue_tests = pipeline.summarize_stage2_by_donor_and_tissue(
        pd.DataFrame(rows),
        min_donors=5,
    )
    d1 = donor_summary.loc[donor_summary["donor_id"].eq("D1")].iloc[0]
    assert d1["donor_effect"] == pytest.approx(0.4)
    assert d1["n_cores"] == 2
    assert tissue_tests.loc[0, "n_donors"] == 5
    assert 0 <= tissue_tests.loc[0, "FDR"] <= 1


def test_stage2_empty_results_keep_documented_schema():
    result = pipeline.calculate_stage2_radius_continuum_by_core(
        _stage2_adata(),
        min_am_cells=100,
        min_at2_cells=100,
        n_permutations=9,
    )
    assert result.empty
    for column in ("method", "core_id", "effect", "p_association"):
        assert column in result.columns


def test_stage2_signed_statistics_do_not_apply_log_ratios():
    """Signed correlations and differences must not enter a ratio-scale helper."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.run_spatial_function_multicore(
            pipeline.calculate_stage2_radius_continuum_by_core,
            _stage2_adata(two_cores=True),
            n_jobs=1,
            random_state=42,
            verbose=0,
            radii=(1,),
            min_am_cells=6,
            min_at2_cells=3,
            n_permutations=9,
        )
    runtime_warnings = [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    assert runtime_warnings == []


def test_new_public_functions_have_numpy_docstrings_and_exports():
    names = (
        "plot_stage1A_niche_dotmap",
        "plot_stage1B_primary",
        "run_spatial_function_multicore",
        "calculate_stage2_balanced_extremes_by_core",
        "calculate_stage2_knn_continuum_by_core",
        "calculate_stage2_radius_continuum_by_core",
        "calculate_stage2_nearest_at2_by_core",
        "run_stage2_multicore",
        "summarize_stage2_by_donor_and_tissue",
    )
    for name in names:
        function = getattr(pipeline, name)
        docstring = inspect.getdoc(function)
        assert docstring is not None
        assert "Parameters" in docstring
        assert "Returns" in docstring
        assert name in pipeline.__all__
