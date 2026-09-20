"""Behavioral tests for Stage 3 AM MHCII–AT2 VIM spatial analyses."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

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
