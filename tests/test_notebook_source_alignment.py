"""Contract tests for notebook, source-module, and README alignment."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
PIPELINE_PATH = ROOT / "scripts" / "xty_am_pipeline.py"
README_PATH = ROOT / "README.md"

EXPECTED_NOTEBOOKS = (
    "01_Metadata_Summary.ipynb",
    "02_1_AM_MHCII_Scoring.ipynb",
    "02_MA_Refinement.ipynb",
    "02_Myeloid_Refinement.ipynb",
    "03_1_AT2_Vim_Scoring.ipynb",
    "03_Epi_Refinement.ipynb",
    "04_Merge_Obs_To_Main.ipynb",
    "05_1A_Unbiased_Niche_Discovery.ipynb",
    "05_1B_Focused_AMAT2_Validation.ipynb",
    "06_AM_MHCII_AT2_Spatial_Association.ipynb",
    "07_1_AM_MHCII_AT2_VIM_Categorical_Spatial_Association.ipynb",
    "07_AM_MHCII_AT2_VIM_Continuous_Spatial_Association.ipynb",
    "08_Spatial_Umap.ipynb",
    "09_MHCprop_Age_Correlation.ipynb",
)

SECTION_MARKERS = (
    "# Notebook 01 - Metadata summary",
    "# Notebooks 02-03 - Cell-type refinement and state scoring",
    "# Notebook 04 - Merge refined metadata",
    "# Notebooks 05.1A-05.1B - Stage 1 spatial niche analysis",
    "# Notebook 06 - AM MHCII-AT2 spatial association",
    "# Notebooks 07-07.1 - AM MHCII-AT2 VIM spatial association",
    "# Notebook 08 - Spatial maps and core influence",
    "# Notebook 09 - MHCII-high proportion and age",
    "# Reusable ligand-receptor and CellChat utilities",
)


def _module_contract():
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    all_names = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            all_names = set(ast.literal_eval(node.value))
            break
    assert all_names is not None, "The pipeline must define __all__."
    return source, functions, all_names


def _notebook_pipeline_references(notebook_path, pipeline_functions):
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    assert notebook.get("cells")
    assert all(cell.get("cell_type") in {"code", "markdown", "raw"} for cell in notebook["cells"])
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", code))
    return pipeline_functions & tokens


def test_notebook_inventory_is_the_supplied_stepwise_workflow():
    observed = tuple(sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb")))
    assert observed == EXPECTED_NOTEBOOKS


def test_pipeline_sections_follow_notebook_execution_order():
    source, _, _ = _module_contract()
    positions = [source.index(marker) for marker in SECTION_MARKERS]
    assert positions == sorted(positions)

    representative_functions = (
        "plot_metadata_summary",
        "plot_anndata_group_umap",
        "merge_obs_to_main",
        "calculate_multitype_nhood_enrichment_by_core",
        "calculate_stage2_knn_continuum_by_core",
        "calculate_stage3_knn_continuum_by_core",
        "rank_stage2_core_contributions",
        "plot_mhcii_hi_proportion_by_age",
        "assign_balanced_mhcii_extremes",
    )
    function_positions = [source.index(f"def {name}(") for name in representative_functions]
    assert function_positions == sorted(function_positions)


def test_notebook_pipeline_references_exist_and_are_exported():
    _, functions, exported = _module_contract()
    for notebook_name in EXPECTED_NOTEBOOKS:
        references = _notebook_pipeline_references(
            NOTEBOOK_DIR / notebook_name,
            functions,
        )
        assert references <= exported, (
            f"{notebook_name} uses non-exported pipeline functions: "
            f"{sorted(references - exported)}"
        )


def test_readme_lists_every_notebook_and_every_public_function_usage():
    readme = README_PATH.read_text(encoding="utf-8")
    for notebook_name in EXPECTED_NOTEBOOKS:
        assert f"`notebooks/{notebook_name}`" in readme

    assert "## Function-to-notebook index" in readme
    assert "| Function | Used in notebook(s) |" in readme

    _, functions, exported = _module_contract()
    public_functions = sorted(
        name for name in functions & exported if not name.startswith("_")
    )
    index_text = readme.split("## Function-to-notebook index", 1)[1]
    missing = [name for name in public_functions if f"| `{name}` |" not in index_text]
    assert not missing, f"README function index is missing: {missing}"
