"""Tests for Spatial CellChat export and continuous AM–AT2 LR analysis."""

from __future__ import annotations

import gzip
import inspect
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmread

from scripts import xty_am_pipeline as pipeline


def _lr_adata(*, two_cores: bool = True) -> ad.AnnData:
    """Return small cores where AM LR expression increases with MHCII score."""
    genes = ["LIG", "REC", "CD74", "OTHER"]
    celltypes = ["AM"] * 4 + ["AT2"] * 3
    scores = [0.0, 1.0, 2.0, 3.0] + [np.nan] * 3
    expression = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 1.0],
            [3.0, 3.0, 3.0, 1.0],
            [1.0, 2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    coordinates = np.asarray(
        [[0, 0], [10, 0], [20, 0], [30, 0], [0, 1], [15, 1], [30, 1]],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "CellType_refined": celltypes,
            "MHCIIhi_score": scores,
            "MHCII_extreme_group": [
                "MHCIIlo", "Ambiguous", "Ambiguous", "MHCIIhi",
                "Unassigned", "Unassigned", "Unassigned",
            ],
            "core_id": "C1",
            "donor_id": "D1",
            "tissue_annotation": "A",
        },
        index=[f"C1_{index}" for index in range(7)],
    )
    if two_cores:
        obs2 = obs.copy()
        obs2.index = [f"C2_{index}" for index in range(7)]
        obs2["core_id"] = "C2"
        obs2["donor_id"] = "D2"
        obs = pd.concat([obs, obs2])
        expression = np.vstack([expression, expression])
        coordinates = np.vstack([coordinates, coordinates])
    adata = ad.AnnData(
        X=sparse.csr_matrix(expression),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    adata.obsm["spatial"] = coordinates
    return adata


def test_balanced_extremes_are_equal_and_skip_tied_boundaries():
    obs = pd.DataFrame(
        {
            "MHCIIhi_score": np.arange(8, dtype=float),
            "CellType_refined": "AM",
            "core_id": "C1",
        },
        index=[f"cell_{i}" for i in range(8)],
    )
    groups = pipeline.assign_balanced_mhcii_extremes(
        obs, fraction=0.25, min_am=4
    )
    assert groups.value_counts().to_dict() == {
        "Ambiguous": 4,
        "MHCIIlo": 2,
        "MHCIIhi": 2,
    }

    obs["MHCIIhi_score"] = [0, 0, 0, 1, 2, 3, 3, 3]
    tied = pipeline.assign_balanced_mhcii_extremes(
        obs, fraction=0.25, min_am=4
    )
    assert set(tied) == {"Ambiguous"}


def test_add_cellchat_groups_preserves_non_am_and_separates_unassigned():
    adata = _lr_adata(two_cores=False)
    result = pipeline.add_cellchat_groups(adata)
    am_labels = set(result.iloc[:4].astype(str))
    assert am_labels == {"AM_MHCIIlo", "AM_MHCII_mid", "AM_MHCIIhi"}
    assert set(result.iloc[4:].astype(str)) == {"AT2"}
    adata.obs.loc["C1_1", "MHCII_extreme_group"] = "Unassigned"
    result = pipeline.add_cellchat_groups(adata)
    assert str(result.loc["C1_1"]) == "AM_unassigned"


def test_cellchat_export_writes_sparse_inputs_and_manifest(tmp_path):
    adata = _lr_adata(two_cores=False)
    pipeline.add_cellchat_groups(adata)
    report = pipeline.export_spatial_cellchat_inputs(
        adata,
        tmp_path,
        expression_scale="log1p_normalized",
        coordinate_units="micrometre",
    )
    expected = {
        "expression", "genes", "cells", "metadata", "coordinates", "manifest"
    }
    assert expected.issubset(report)
    assert all(report[key].exists() for key in expected)
    with gzip.open(report["expression"], "rb") as handle:
        exported = mmread(handle)
    assert exported.shape == (adata.n_vars, adata.n_obs)
    manifest = json.loads(report["manifest"].read_text(encoding="utf-8"))
    assert manifest["n_cells"] == adata.n_obs
    assert manifest["n_genes"] == adata.n_vars
    assert manifest["expression_orientation"] == "genes_by_cells"
    assert manifest["expression_scale"] == "log1p_normalized"
    assert manifest["input_coordinate_units"] == "micrometre"
    assert manifest["coordinate_scale_to_um"] == 1.0


def test_cellchat_export_requires_declared_scale_and_coordinate_conversion(tmp_path):
    adata = _lr_adata(two_cores=False)
    pipeline.add_cellchat_groups(adata)
    with pytest.raises(ValueError, match="expression_scale"):
        pipeline.export_spatial_cellchat_inputs(
            adata,
            tmp_path / "counts",
            expression_scale="raw_counts",
            coordinate_units="micrometre",
        )
    with pytest.raises(ValueError, match="coordinate_scale_to_um"):
        pipeline.export_spatial_cellchat_inputs(
            adata,
            tmp_path / "pixels",
            expression_scale="log1p_normalized",
            coordinate_units="pixel",
        )


def test_cellchat_export_rejects_negative_expression_and_missing_metadata(tmp_path):
    adata = _lr_adata(two_cores=False)
    pipeline.add_cellchat_groups(adata)
    adata.X[0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        pipeline.export_spatial_cellchat_inputs(
            adata,
            tmp_path / "negative",
            expression_scale="log1p_normalized",
            coordinate_units="micrometre",
        )
    adata = _lr_adata(two_cores=False)
    with pytest.raises(KeyError, match="metadata"):
        pipeline.export_spatial_cellchat_inputs(
            adata,
            tmp_path / "missing",
            expression_scale="log1p_normalized",
            coordinate_units="micrometre",
        )


def test_lr_panel_audit_requires_simple_measured_pairs():
    pairs = pd.DataFrame(
        {
            "ligand": ["lig", "LIG", "LIG"],
            "receptor": ["rec", "REC_A", "MISSING"],
        }
    )
    audit = pipeline.audit_lr_panel(pairs, ["LIG", "REC"])
    assert audit["complete_pair"].tolist() == [True, False, False]
    assert audit["simple_pair"].tolist() == [True, False, True]


def test_radius_weighted_mean_validates_and_reports_neighbor_counts():
    means, counts = pipeline.radius_weighted_mean(
        query_coords=np.asarray([[0.0, 0.0], [10.0, 0.0]]),
        target_coords=np.asarray([[0.0, 1.0], [9.0, 0.0]]),
        target_values=np.asarray([2.0, 4.0]),
        radius=2,
        kernel="uniform",
    )
    np.testing.assert_allclose(means, [2.0, 4.0])
    np.testing.assert_array_equal(counts, [1, 1])
    with pytest.raises(ValueError, match="sigma"):
        pipeline.radius_weighted_mean(
            np.asarray([[0.0, 0.0]]),
            np.asarray([[0.0, 1.0]]),
            np.asarray([1.0]),
            radius=2,
            kernel="gaussian",
            sigma=0,
        )


def test_core_lr_analysis_supports_both_directions_and_flags_score_overlap():
    pairs = pd.DataFrame({"ligand": ["LIG"], "receptor": ["REC"]})
    result = pipeline.calculate_lr_for_core_arrays(
        am_xy=np.asarray([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=float),
        at2_xy=np.asarray([[0, 1], [15, 1], [30, 1]], dtype=float),
        mhcii_score=np.arange(4, dtype=float),
        am_expr={"LIG": np.arange(4, dtype=float), "REC": np.arange(4, dtype=float)},
        at2_expr={"LIG": np.ones(3), "REC": np.full(3, 2.0)},
        lr_pairs=pairs,
        radius=16,
        min_am=4,
        n_permutations=19,
        random_state=5,
        mhcii_signature_genes=("LIG",),
    )
    assert set(result["direction"]) == {"AM_to_AT2", "AT2_to_AM"}
    assert result["rho"].gt(0).all()
    am_to_at2 = result.loc[result["direction"].eq("AM_to_AT2")].iloc[0]
    assert bool(am_to_at2["am_gene_in_mhcii_signature"])
    assert 0 <= am_to_at2["empirical_p"] <= 1
    assert am_to_at2["pct_am_gene_expressing"] == pytest.approx(75.0)


def test_all_core_lr_is_order_stable_and_fdr_is_within_core_family():
    adata = _lr_adata(two_cores=True)
    pairs = pd.DataFrame(
        {"ligand": ["LIG", "OTHER"], "receptor": ["REC", "REC"]}
    )
    kwargs = dict(
        lr_pairs=pairs,
        radii=(16,),
        min_am=4,
        min_at2=3,
        n_permutations=19,
        random_state=11,
        n_jobs=1,
        mhcii_signature_genes=("LIG", "CD74"),
    )
    original = pipeline.calculate_continuous_spatial_lr(adata, **kwargs)
    order = np.r_[
        np.flatnonzero(adata.obs["core_id"].eq("C2")),
        np.flatnonzero(adata.obs["core_id"].eq("C1")),
    ]
    reordered = pipeline.calculate_continuous_spatial_lr(
        adata[order].copy(), **kwargs
    )
    sort_cols = ["core_id", "direction", "ligand", "receptor", "radius_um"]
    compare = sort_cols + ["rho", "empirical_p", "core_FDR"]
    pd.testing.assert_frame_equal(
        original[compare].sort_values(sort_cols).reset_index(drop=True),
        reordered[compare].sort_values(sort_cols).reset_index(drop=True),
    )
    testable = original["rho"].notna()
    assert original.loc[testable, "core_FDR"].between(0, 1).all()
    assert original.loc[~testable, "core_FDR"].isna().all()


def test_all_core_lr_rejects_missing_provenance_and_invalid_coordinate_scale():
    pairs = pd.DataFrame({"ligand": ["LIG"], "receptor": ["REC"]})
    adata = _lr_adata(two_cores=False)
    adata.obs.loc[adata.obs.index[0], "donor_id"] = pd.NA
    with pytest.raises(ValueError, match="missing core/donor/tissue"):
        pipeline.calculate_continuous_spatial_lr(
            adata, pairs, radii=(16,), min_am=4, min_at2=3,
            n_permutations=0,
        )
    adata = _lr_adata(two_cores=False)
    with pytest.raises(ValueError, match="coordinate_scale"):
        pipeline.calculate_continuous_spatial_lr(
            adata, pairs, radii=(16,), min_am=4, min_at2=3,
            n_permutations=0, coordinate_scale=0,
        )


def test_all_core_lr_thread_backend_matches_serial():
    adata = _lr_adata(two_cores=True)
    pairs = pd.DataFrame({"ligand": ["LIG"], "receptor": ["REC"]})
    kwargs = dict(
        lr_pairs=pairs, radii=(16,), min_am=4, min_at2=3,
        n_permutations=9, random_state=17,
    )
    serial = pipeline.calculate_continuous_spatial_lr(adata, n_jobs=1, **kwargs)
    threaded = pipeline.calculate_continuous_spatial_lr(
        adata, n_jobs=2, parallel_backend_name="threading", **kwargs
    )
    columns = [
        "core_id", "direction", "ligand", "receptor", "radius_um",
        "rho", "empirical_p", "core_FDR",
    ]
    sort_columns = columns[:5]
    pd.testing.assert_frame_equal(
        serial[columns].sort_values(sort_columns).reset_index(drop=True),
        threaded[columns].sort_values(sort_columns).reset_index(drop=True),
    )


def test_lr_donor_summary_equal_weights_cores_and_tests_donors():
    core_rows = []
    for donor, values in {
        "D1": ((0.2, 10), (0.8, 1000)),
        "D2": ((0.3, 20),),
        "D3": ((0.4, 20),),
        "D4": ((0.5, 20),),
        "D5": ((0.6, 20),),
    }.items():
        for index, (rho, n_cells) in enumerate(values):
            core_rows.append(
                {
                    "core_id": f"{donor}.C{index}",
                    "donor_id": donor,
                    "tissue_annotation": "A",
                    "direction": "AM_to_AT2",
                    "ligand": "LIG",
                    "receptor": "REC",
                    "radius_um": 50.0,
                    "rho": rho,
                    "n_am_tested": n_cells,
                    "pct_am_gene_expressing": 50.0,
                    "pct_at2_gene_expressing": 60.0,
                    "pct_am_with_at2_neighbors": 90.0,
                }
            )
    donor = pipeline.summarize_lr_by_donor(pd.DataFrame(core_rows))
    d1 = donor.loc[donor["donor_id"].eq("D1")].iloc[0]
    expected = np.tanh(np.mean(np.arctanh([0.2, 0.8])))
    assert d1["donor_rho"] == pytest.approx(expected)
    assert d1["n_cores"] == 2
    pd.testing.assert_frame_equal(donor, pipeline.summarise_lr_by_donor(pd.DataFrame(core_rows)))
    tests = pipeline.test_lr_across_donors(donor, min_donors=5)
    assert tests.loc[0, "n_donors"] == 5
    assert 0 <= tests.loc[0, "FDR"] <= 1


def test_spatial_lr_public_functions_have_docstrings_and_exports():
    names = (
        "assign_balanced_mhcii_extremes",
        "add_cellchat_groups",
        "export_spatial_cellchat_inputs",
        "audit_lr_panel",
        "radius_weighted_mean",
        "calculate_lr_for_core_arrays",
        "calculate_continuous_spatial_lr",
        "summarize_lr_by_donor",
        "summarise_lr_by_donor",
        "test_lr_across_donors",
    )
    for name in names:
        function = getattr(pipeline, name)
        docstring = inspect.getdoc(function)
        assert docstring is not None
        assert "Parameters" in docstring
        assert "Returns" in docstring
        assert name in pipeline.__all__
