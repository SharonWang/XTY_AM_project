import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "00_methodology_data_audit.ipynb"


def test_notebook_has_required_reader_facing_structure():
    """Fail if notebook 00 loses its reproducible top-to-bottom structure."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)

    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    required_headings = [
        "## tl;dr",
        "## Context & Methods",
        "## Data",
        "## Results",
        "## Takeaways",
    ]
    for heading in required_headings:
        assert heading in markdown
    for step in ("6a", "6b", "6c", "6d", "6e", "6f"):
        assert f"#### Step {step}" in markdown

    code_sources = [
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    ]
    for step in ("6a", "6b", "6c", "6d", "6e", "6f"):
        matching_cells = [
            source for source in code_sources if f"Step {step}:" in source
        ]
        assert len(matching_cells) == 1, step
    analysis_cells = [
        source for source in code_sources if "# Step 6" in source
    ]
    assert len(analysis_cells) == 6

    parameter_cells = [
        cell
        for cell in notebook.cells
        if "parameters" in cell.metadata.get("tags", [])
    ]
    assert len(parameter_cells) == 1


def execute_local_notebook():
    """Execute notebook 00 with the real local kernel and return its summary."""
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        # Match JupyterLab's usual execution context: the notebook directory.
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
    summary_path = (
        Path(os.environ["XTY_AM_OUTPUT_ROOT"]) / "notebook_00_summary.json"
    )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def test_local_notebook_executes_and_writes_validated_summary(
    tmp_path, monkeypatch
):
    """Fail if the notebook cannot audit the real data and publish key counts."""
    monkeypatch.setenv("XTY_AM_RUN_MODE", "local_test")
    monkeypatch.setenv("XTY_AM_OUTPUT_ROOT", str(tmp_path))
    summary = execute_local_notebook()

    assert summary["n_cells"] > 0
    assert summary["celltype_counts"]["Macrophages"] > 0
    assert summary["celltype_counts"]["AT1"] > 0
    assert summary["celltype_counts"]["AT2"] > 0
    assert summary["duplicate_cell_ids"] == 0
    assert summary["umap_cells_plotted"] == summary["n_cells"]
    assert len(summary["selected_cores"]) == 4
    assert summary["spatial_display_strategy"] == "four_representative_complete_alveolar_cores"
    assert summary["expression_zero_pattern_concordance"] > 0.999
    assert summary["pipeline_source_script"] == "scripts/xty_am_pipeline.py"

    expected_figures = [
        "umap_all_celltypes",
        "umap_focus_macrophage_AT1_AT2",
        "dotplot_annotation_markers",
        "umap_macrophage_identity_features",
        "spatial_all_celltypes",
        "spatial_focus_macrophage_AT1_AT2",
        "spatial_macrophage_identity_features",
    ]
    for stem in expected_figures:
        for suffix in (".png", ".pdf"):
            figure_path = tmp_path / "figures" / f"{stem}{suffix}"
            assert figure_path.exists(), figure_path
            assert figure_path.stat().st_size > 1_000, figure_path

    availability = summary["marker_availability"]
    required_measured = {
        "AGER", "SCEL", "SFTPD", "PLA2G4F", "MARCO", "APOE",
        "LYVE1", "CD163", "FCGR3A", "MS4A4A",
        "FCN1", "S100A12", "IL1B", "CLEC4E",
    }
    assert all(availability[gene] is True for gene in required_measured)
    assert availability["FABP4"] is False
    assert availability["SFTPC"] is False

    expected_tables = [
        "annotation_marker_summary.csv",
        "annotation_marker_summary_by_core.csv",
        "annotation_marker_summary_by_donor.csv",
        "macrophage_candidate_program_scores_local_or_full.csv",
        "macrophage_program_summary_by_core.csv",
        "macrophage_program_summary_by_donor.csv",
        "marker_availability.csv",
    ]
    for name in expected_tables:
        table_path = tmp_path / "tables" / name
        assert table_path.exists(), table_path
        assert table_path.stat().st_size > 100, table_path
