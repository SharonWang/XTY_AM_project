"""Behavioral tests for Stage 3 AM MHCII–AT2 VIM spatial analyses."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import xty_am_pipeline as pipeline


def _one_core(donor: str, core: str, *, negative_coupling: bool = True) -> pd.DataFrame:
    """Construct one small core with paired AM and AT2 coordinates."""
    rows = []
    mhcii = np.arange(4, dtype=float)
    vim = mhcii[::-1] if negative_coupling else mhcii.copy()
    for index, score in enumerate(mhcii):
        rows.append(
            {
                "donor_id": donor,
                "tissue_annotation": "A",
                "core_id": core,
                "CellType_refined": "AM",
                "x_centroid": index * 10.0,
                "y_centroid": 0.0,
                "MHCIIhi_score": score,
                "AT2_VIM_score": np.nan,
            }
        )
    for index, score in enumerate(vim):
        rows.append(
            {
                "donor_id": donor,
                "tissue_annotation": "A",
                "core_id": core,
                "CellType_refined": "AT2",
                "x_centroid": index * 10.0 + 0.5,
                "y_centroid": 0.0,
                "MHCIIhi_score": np.nan,
                "AT2_VIM_score": score,
            }
        )
    return pd.DataFrame(rows)


def make_stage3_adata(
    *,
    reuse_core_id: bool = False,
    negative_coupling: bool = True,
    include_unrelated: bool = False,
):
    """Return a minimal obs-only object containing one or two spatial cores."""
    first = _one_core("D1", "C1", negative_coupling=negative_coupling)
    frames = [first]
    if reuse_core_id or include_unrelated:
        second_core = "C1" if reuse_core_id else "C2"
        frames.append(_one_core("D2", second_core, negative_coupling=False))
    obs = pd.concat(frames, ignore_index=True)
    obs.index = [f"cell_{index}" for index in range(len(obs))]
    return SimpleNamespace(obs=obs, n_obs=len(obs))


def make_self_neighbor_adata():
    """Return geometry where every AM itself is the nearest all-cell point."""
    return make_stage3_adata(negative_coupling=False)


def make_tied_stage3_adata():
    """Return a core whose MHCII tail boundaries are completely tied."""
    adata = make_stage3_adata()
    mask = adata.obs["CellType_refined"].eq("AM")
    adata.obs.loc[mask, "MHCIIhi_score"] = 1.0
    return adata


def make_categorical_stage3_adata(
    *,
    discordant: bool = True,
    include_ambiguous: bool = False,
    one_am_group: bool = False,
):
    """Return categorical AM/AT2 states with known local pairing."""
    rows = []
    am_groups = ["MHCIIlo", "MHCIIlo", "MHCIIhi", "MHCIIhi"]
    if one_am_group:
        am_groups = ["MHCIIhi"] * 4
    vim_groups = (
        ["AT2_VIMhi", "AT2_VIMhi", "AT2_VIMlo", "AT2_VIMlo"]
        if discordant
        else ["AT2_VIMlo", "AT2_VIMlo", "AT2_VIMhi", "AT2_VIMhi"]
    )
    for index, group in enumerate(am_groups):
        rows.append(
            {
                "donor_id": "D1",
                "tissue_annotation": "A",
                "core_id": "C1",
                "CellType_refined": "AM",
                "x_centroid": index * 10.0,
                "y_centroid": 0.0,
                "MHCII_group": group,
                "AT2_VIM_group": pd.NA,
            }
        )
    for index, group in enumerate(vim_groups):
        rows.append(
            {
                "donor_id": "D1",
                "tissue_annotation": "A",
                "core_id": "C1",
                "CellType_refined": "AT2",
                "x_centroid": index * 10.0 + 0.5,
                "y_centroid": 0.0,
                "MHCII_group": pd.NA,
                "AT2_VIM_group": group,
            }
        )
    if include_ambiguous:
        rows.extend(
            [
                {
                    "donor_id": "D1", "tissue_annotation": "A", "core_id": "C1",
                    "CellType_refined": "AM", "x_centroid": 40.0, "y_centroid": 0.0,
                    "MHCII_group": "Ambiguous", "AT2_VIM_group": pd.NA,
                },
                {
                    "donor_id": "D1", "tissue_annotation": "A", "core_id": "C1",
                    "CellType_refined": "AT2", "x_centroid": 40.5, "y_centroid": 0.0,
                    "MHCII_group": pd.NA, "AT2_VIM_group": "Ambiguous",
                },
            ]
        )
    obs = pd.DataFrame(rows)
    obs.index = [f"categorical_{index}" for index in range(len(obs))]
    return SimpleNamespace(obs=obs, n_obs=len(obs))


def make_stage3_core_results() -> pd.DataFrame:
    """Return correlation effects with unequal core counts per donor."""
    return pd.DataFrame(
        [
            {"method": "radius_continuum", "scale_type": "radius_um", "scale": 50.0,
             "effect_name": "spearman_rho", "effect": 0.2, "donor_id": "D1",
             "tissue_annotation": "A", "core_id": "C1", "n_am_analyzed": 20},
            {"method": "radius_continuum", "scale_type": "radius_um", "scale": 50.0,
             "effect_name": "spearman_rho", "effect": 0.8, "donor_id": "D1",
             "tissue_annotation": "A", "core_id": "C2", "n_am_analyzed": 20},
            {"method": "radius_continuum", "scale_type": "radius_um", "scale": 50.0,
             "effect_name": "spearman_rho", "effect": 0.3, "donor_id": "D2",
             "tissue_annotation": "A", "core_id": "C3", "n_am_analyzed": 20},
            {"method": "radius_continuum", "scale_type": "radius_um", "scale": 50.0,
             "effect_name": "spearman_rho", "effect": 0.4, "donor_id": "D3",
             "tissue_annotation": "A", "core_id": "C4", "n_am_analyzed": 20},
        ]
    )


def make_multifamily_stage3_results() -> pd.DataFrame:
    """Return one test family spanning three tissues and three donors each."""
    rows = []
    for tissue_index, tissue in enumerate(("A", "B", "V")):
        for donor_index in range(3):
            rows.append(
                {
                    "method": "radius_continuum",
                    "scale_type": "radius_um",
                    "scale": 50.0,
                    "effect_name": "spearman_rho",
                    "effect": -0.2 - 0.05 * tissue_index - 0.01 * donor_index,
                    "donor_id": f"D{donor_index + 1}",
                    "tissue_annotation": tissue,
                    "core_id": f"{tissue}_C{donor_index + 1}",
                    "n_am_analyzed": 20,
                }
            )
    return pd.DataFrame(rows)


def test_stage3_reused_core_ids_remain_separate_by_donor():
    """Identical core labels in different donors must never be pooled."""
    result = pipeline.calculate_stage3_radius_continuum_by_core(
        make_stage3_adata(reuse_core_id=True),
        radii=(3,),
        min_at2_neighbors=1,
        min_am=3,
        n_permutations=19,
        random_state=11,
    )
    assert set(result["donor_id"]) == {"D1", "D2"}
    assert len(result) == 2


def test_stage3_all_cell_knn_excludes_focal_am_self():
    """The focal AM must not consume an all-cell kNN slot."""
    result = pipeline.calculate_stage3_knn_continuum_by_core(
        make_self_neighbor_adata(),
        k_values=(1,),
        neighbor_pool="all",
        min_at2_neighbors=1,
        min_am=3,
        n_permutations=19,
    )
    assert result.iloc[0]["median_at2_neighbors"] == 1


def test_stage3_balanced_extremes_do_not_split_tied_boundaries():
    """Equal scores at a tail boundary must not be assigned to opposite tails."""
    result = pipeline.calculate_stage3_balanced_extremes_by_core(
        make_tied_stage3_adata(),
        radii=(3,),
        min_at2_neighbors=1,
        min_extreme_cells=2,
        n_permutations=19,
    )
    assert result.empty


def test_stage3_negative_coupling_has_negative_effect():
    """Higher MHCII paired with lower VIM must retain a negative sign."""
    result = pipeline.calculate_stage3_nearest_at2_by_core(
        make_stage3_adata(negative_coupling=True),
        min_am=3,
        n_permutations=19,
        random_state=17,
    )
    assert result.iloc[0]["effect"] < 0


def test_stage3_multicore_seed_is_stable_when_unrelated_core_is_removed():
    """Unrelated cores must not change a retained core's random stream."""
    full = pipeline.run_stage3_multicore(
        pipeline.calculate_stage3_radius_continuum_by_core,
        make_stage3_adata(include_unrelated=True),
        n_jobs=1,
        radii=(3,),
        min_at2_neighbors=1,
        min_am=3,
        n_permutations=19,
    )
    subset = pipeline.run_stage3_multicore(
        pipeline.calculate_stage3_radius_continuum_by_core,
        make_stage3_adata(include_unrelated=False),
        n_jobs=1,
        radii=(3,),
        min_at2_neighbors=1,
        min_am=3,
        n_permutations=19,
    )
    columns = ["donor_id", "core_id", "effect", "p_value"]
    pd.testing.assert_frame_equal(
        full.loc[full["donor_id"].eq("D1"), columns].reset_index(drop=True),
        subset.loc[:, columns].reset_index(drop=True),
    )


def test_stage3_categorical_pair_effect_is_negative_for_discordant_pairing():
    """Discordant MHCII/VIM pairings must produce a negative log odds ratio."""
    result = pipeline.calculate_stage3_categorical_pair_enrichment_by_core(
        make_categorical_stage3_adata(discordant=True),
        radii=(3,),
        min_am_per_group=2,
        min_at2_per_group=2,
        n_permutations=19,
    )
    assert result.iloc[0]["effect"] < 0
    assert result.iloc[0]["effect_name"] == "log_concordance_odds_ratio"


def test_stage3_categorical_reports_ambiguous_cell_exclusion():
    """Ambiguous AM and AT2 states must be reported as exclusions."""
    result = pipeline.calculate_stage3_categorical_radius_by_core(
        make_categorical_stage3_adata(include_ambiguous=True),
        radii=(3,),
        min_at2_neighbors=1,
        min_am_per_group=2,
        n_balance_repeats=5,
        n_permutations=19,
    )
    row = result.iloc[0]
    assert row["n_am_excluded"] > 0
    assert row["n_at2_excluded"] > 0
    assert 0 < row["am_retained_fraction"] < 1


def test_stage3_categorical_nearest_requires_both_am_groups():
    """A categorical contrast is undefined when one AM state is absent."""
    result = pipeline.calculate_stage3_categorical_nearest_at2_by_core(
        make_categorical_stage3_adata(one_am_group=True),
        min_am_per_group=2,
        n_permutations=19,
    )
    assert result.empty


def test_stage3_summary_equal_weights_cores_and_supports_less_alternative():
    """Donors, not cells or cores, must be the tissue-level replicates."""
    donor, tissue = pipeline.summarize_stage3_by_donor_and_tissue(
        make_stage3_core_results(), alternative="less", min_donors=3
    )
    d1 = donor.loc[donor["donor_id"].eq("D1")].iloc[0]
    expected = np.tanh(np.mean(np.arctanh([0.2, 0.8])))
    assert d1["donor_effect"] == pytest.approx(expected)
    assert d1["n_cores"] == 2
    assert tissue.iloc[0]["alternative"] == "less"


def test_stage3_summary_fdr_is_scoped_by_method_effect_and_scale():
    """BH correction must operate across tissues only within one method family."""
    _, tissue = pipeline.summarize_stage3_by_donor_and_tissue(
        make_multifamily_stage3_results(), min_donors=3
    )
    assert "fdr_bh" in tissue
    assert set(tissue["fdr_family_size"]) == {3}


def test_stage3_plots_save_outputs(tmp_path):
    """Primary, sensitivity, and pair heatmap functions must write figures."""
    donor, tissue = pipeline.summarize_stage3_by_donor_and_tissue(
        make_multifamily_stage3_results(), min_donors=3
    )
    primary = tmp_path / "primary.pdf"
    sensitivity = tmp_path / "sensitivity.pdf"
    heatmap = tmp_path / "pairs.pdf"
    pipeline.plot_stage3_primary(donor, tissue, save=primary)
    pipeline.plot_stage3_scale_sensitivity(donor, save=sensitivity)
    pair_results = pd.DataFrame(
        {
            "method": ["categorical_pair_enrichment"] * 3,
            "donor_id": ["D1", "D2", "D3"],
            "tissue_annotation": ["A", "B", "V"],
            "scale": [50.0, 50.0, 50.0],
            "log2_oe_lo_lo": [0.2, 0.1, -0.1],
            "log2_oe_lo_hi": [-0.2, -0.1, 0.1],
            "log2_oe_hi_lo": [0.3, 0.2, -0.2],
            "log2_oe_hi_hi": [-0.3, -0.2, 0.2],
        }
    )
    pipeline.plot_stage3_categorical_pair_heatmap(pair_results, save=heatmap)
    assert primary.exists() and sensitivity.exists() and heatmap.exists()
