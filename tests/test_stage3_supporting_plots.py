"""Tests for Stage 3 spatial-core and exploratory age figures."""

from __future__ import annotations

import anndata as ad
import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import pandas as pd
import pytest

from scripts import xty_am_pipeline as pipeline


def make_spatial_plot_adata() -> ad.AnnData:
    """Return one core with AM scores and categorical AT2 VIM states."""
    celltypes = ["AM"] * 4 + ["AT2"] * 4 + ["Fibroblasts"]
    obs = pd.DataFrame(
        {
            "core_id": ["C1"] * 9,
            "donor_id": ["D1"] * 9,
            "tissue_annotation": ["A"] * 9,
            "CellType_refined": celltypes,
            "MHCIIhi_score": [0.1, 0.3, 0.7, 0.9] + [np.nan] * 5,
            "AT2_VIM_group": [pd.NA] * 4
            + ["AT2_VIMhi", "AT2_VIMhi", "AT2_VIMlo", "AT2_VIMlo"]
            + [pd.NA],
        },
        index=[f"spatial_{index}" for index in range(9)],
    )
    adata = ad.AnnData(X=np.zeros((9, 1)), obs=obs, var=pd.DataFrame(index=["G1"]))
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(9, dtype=float), np.zeros(9, dtype=float)]
    )
    return adata


def make_age_adata(*, inconsistent: bool = False) -> ad.AnnData:
    """Return three donors with four AMs each and known high-state counts."""
    rows = []
    for donor_index, (donor, age, n_high) in enumerate(
        (("D1", 30, 1), ("D2", 50, 2), ("D3", 70, 3))
    ):
        for cell_index in range(4):
            cell_age = age + (1 if inconsistent and donor == "D1" and cell_index == 3 else 0)
            rows.append(
                {
                    "donor_id": donor,
                    "core_id": f"C{donor_index + 1}",
                    "age": cell_age,
                    "tissue_annotation": "A",
                    "CellType_refined": "AM",
                    "MHCII_group": "MHCIIhi" if cell_index < n_high else "MHCIIlo",
                }
            )
    obs = pd.DataFrame(rows, index=[f"age_{index}" for index in range(len(rows))])
    return ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs, var=pd.DataFrame(index=["G1"]))


def test_spatial_core_plot_supports_at2_state_groups(tmp_path):
    """Grouped AT2 VIM states must render to the requested D-drive path."""
    output = pipeline.plot_mhcii_at2_spatial_core(
        make_spatial_plot_adata(),
        "C1",
        output_dir=tmp_path,
        at2_display="group",
        at2_group_col="AT2_VIM_group",
        show=False,
    )
    assert output.exists()
    assert output.suffix == ".pdf"


def test_spatial_core_plot_default_remains_all_at2(tmp_path):
    """The historical default must still draw all AT2 cells together."""
    output = pipeline.plot_mhcii_at2_spatial_core(
        make_spatial_plot_adata(), "C1", output_dir=tmp_path, show=False
    )
    assert output.exists()


def test_mhcii_hi_age_plot_uses_all_am_in_denominator(tmp_path):
    """MHCIIlo and unassigned states must remain in the AM denominator."""
    result = pipeline.plot_mhcii_hi_proportion_by_age(
        make_age_adata(), output_file=tmp_path / "age.pdf"
    )
    donor = result["donor_summary"].set_index("donor_id")
    assert donor.loc["D1", "n_all_am"] == 4
    assert donor.loc["D1", "n_mhcii_hi"] == 1
    assert result["output_file"].exists()


def test_mhcii_hi_age_model_rejects_inconsistent_donor_age():
    """One donor cannot contribute more than one age to the model."""
    with pytest.raises(ValueError, match="inconsistent within donor"):
        pipeline.plot_mhcii_hi_proportion_by_age(make_age_adata(inconsistent=True))
