"""Tests for Stage 2 core/donor influence and illustrative core selection."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import xty_am_pipeline as pipeline


def _core_results():
    """Return one Stage 2 hypothesis with unequal core cell counts."""
    return pd.DataFrame(
        {
            "core_id": ["D1.C1", "D1.C2", "D2.C1", "D3.C1"],
            "donor_id": ["D1", "D1", "D2", "D3"],
            "tissue_annotation": ["A"] * 4,
            "radius_um": [50.0] * 4,
            "rho": [0.2, 0.8, 0.4, -0.1],
            "n_am_tested": [10, 1000, 20, 20],
        }
    )


def _native_stage2_results():
    """Return the schema emitted by native Stage 2 radius functions."""
    data = _core_results().rename(
        columns={"radius_um": "scale", "rho": "effect", "n_am_tested": "n_am_analyzed"}
    )
    data["method"] = "Continuous MHCII radius"
    data["direction"] = "MHCII score → AT2 exposure"
    data["scale_type"] = "radius_um"
    data["effect_type"] = "correlation"
    return data


def test_rank_stage2_core_contributions_matches_primary_equal_core_estimator():
    core_ranking, donor_ranking, summary = pipeline.rank_stage2_core_contributions(
        _core_results(), tissue="A", radius=50, min_donors_for_test=3
    )
    expected_d1 = np.mean([0.2, 0.8])
    d1 = donor_ranking.loc[donor_ranking["donor_id"].eq("D1")].iloc[0]
    assert d1["donor_rho"] == pytest.approx(expected_d1)
    assert summary["mean_donor_rho"] == pytest.approx(
        np.mean([expected_d1, 0.4, -0.1])
    )
    assert set(core_ranking["core_weight"]) == {1.0}
    removes_donor = core_ranking.set_index("core_id")["removes_donor"]
    assert not bool(removes_donor["D1.C1"])
    assert bool(removes_donor["D2.C1"])
    assert summary["core_aggregation"] == "equal_core_arithmetic"
    assert summary["test_alternative"] == "greater"

    _, fisher_donors, fisher_summary = pipeline.rank_stage2_core_contributions(
        _core_results(), donor_aggregation="fisher_z", test_alternative="two-sided"
    )
    fisher_d1 = fisher_donors.loc[fisher_donors["donor_id"].eq("D1")].iloc[0]
    assert fisher_d1["donor_rho"] == pytest.approx(
        np.tanh(np.mean(np.arctanh([0.2, 0.8])))
    )
    assert fisher_summary["core_aggregation"] == "equal_core_fisher_z"
    assert fisher_summary["test_alternative"] == "two-sided"


def test_rank_stage2_core_contributions_requires_one_hypothesis_row_per_core():
    duplicated = pd.concat([_core_results(), _core_results().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row remains per core"):
        pipeline.rank_stage2_core_contributions(duplicated)


def test_rank_stage2_core_contributions_accepts_native_correlation_schema_only():
    core_ranking, donor_ranking, summary = pipeline.rank_stage2_core_contributions(
        _native_stage2_results()
    )
    assert len(core_ranking) == 4
    assert len(donor_ranking) == 3
    assert summary["method"] == "Continuous MHCII radius"
    assert summary["scale_type"] == "radius_um"
    noncorrelation = _native_stage2_results()
    noncorrelation["effect_type"] = "difference"
    with pytest.raises(ValueError, match="correlation"):
        pipeline.rank_stage2_core_contributions(noncorrelation)


def test_select_supportive_cores_uses_distinct_donors_and_context_examples():
    core_ranking, _, _ = pipeline.rank_stage2_core_contributions(_core_results())
    selected = pipeline.select_supportive_cores(
        core_ranking, n_supportive=2, include_typical=True, include_discordant=True
    )
    supportive = selected.loc[selected["selection_reason"].eq("Strong supportive core")]
    assert supportive["donor_id"].nunique() == len(supportive)
    assert len(supportive) <= 2
    assert "Discordant negative core" in set(selected["selection_reason"])
    assert selected["selection_is_inferential"].eq(False).all()
    none_supportive = pipeline.select_supportive_cores(
        core_ranking,
        n_supportive=0,
        include_typical=False,
        include_discordant=False,
    )
    assert none_supportive.empty


def test_core_contribution_and_spatial_plots_write_files(tmp_path):
    core_ranking, _, _ = pipeline.rank_stage2_core_contributions(_core_results())
    figure_path = tmp_path / "core_contributions.pdf"
    fig, axes = pipeline.plot_core_contributions(core_ranking, save=figure_path)
    assert figure_path.exists()
    assert len(axes) == 2
    fig.clf()

    native = _native_stage2_results()
    native["tissue_annotation"] = "B"
    native["scale"] = 100.0
    native_ranking, _, _ = pipeline.rank_stage2_core_contributions(
        native, tissue="B", radius=100
    )
    dynamic_fig, dynamic_axes = pipeline.plot_core_contributions(native_ranking)
    assert "tissue B" in dynamic_fig._suptitle.get_text()
    assert "100" in dynamic_fig._suptitle.get_text()
    assert "local AT2 fraction" not in dynamic_axes[0].get_xlabel()
    assert "tissue-A" not in dynamic_axes[1].get_xlabel()
    dynamic_fig.clf()

    obs = pd.DataFrame(
        {
            "core_id": ["C1"] * 6,
            "donor_id": ["D1"] * 6,
            "tissue_annotation": ["A"] * 6,
            "CellType_refined": ["AM", "AM", "AM", "AT2", "AT2", "Other"],
            "MHCIIhi_score": [-1.0, 0.0, 1.0, np.nan, np.nan, np.nan],
        },
        index=[f"cell_{index}" for index in range(6)],
    )
    adata = SimpleNamespace(
        obs=obs,
        obsm={"spatial": np.asarray([[0, 0], [1, 0], [2, 0], [0, 1], [2, 1], [1, 2]])},
    )
    output = pipeline.plot_mhcii_at2_spatial_core(
        adata, "C1", output_dir=tmp_path, show=False
    )
    assert output.exists()
    assert output.suffix == ".pdf"


def test_stage2_influence_public_functions_have_docstrings_and_exports():
    names = (
        "rank_stage2_core_contributions",
        "select_supportive_cores",
        "plot_core_contributions",
        "plot_mhcii_at2_spatial_core",
    )
    for name in names:
        function = getattr(pipeline, name)
        docstring = inspect.getdoc(function)
        assert docstring is not None
        assert "Parameters" in docstring
        assert "Returns" in docstring
        assert name in pipeline.__all__
