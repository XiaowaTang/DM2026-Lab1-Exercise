import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # 1. Create session and load premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Get build_dtm_tool from student's tools_dtm
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 3. Invoke build_dtm_tool
    result = build_dtm_tool.invoke({})

    # 4. Assertions based on known answer #4 in TEST_FIXTURE.md
    assert result["vocabulary_size"] == 5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert np.isclose(result["sparsity_pct"], 47.5, atol=1e-3)

    # 5. Verify session state references and artifacts
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
    # 1. Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Get tools from tools_dtm
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # 3. First run build_dtm_tool to populate session.feature_matrix
    build_dtm_tool.invoke({})

    # 4. Invoke term_frequency_tool
    result = term_frequency_tool.invoke({})

    # 5. Assertions based on known answer #14 in TEST_FIXTURE.md
    assert result["status"] == "success"
    assert result["total_terms"] == 5

    # Verify top_terms in summary
    top_terms_map = {item["term"]: item["frequency"] for item in result["top_terms"]}
    assert top_terms_map["always"] == 7
    assert top_terms_map["gamma"] == 7

    # Retrieve full_report from session.results_store to assert all frequencies
    result_id = result["result_id"]
    full_report = session.results_store[result_id]["full_report"]
    report_map = dict(zip(full_report["term"], full_report["frequency"]))

    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    assert report_map == expected_frequencies


def test_dtm_heatmap_tool():
    # 1. Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Get tools from tools_dtm
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # 3. First run build_dtm_tool to populate session.feature_matrix
    build_dtm_tool.invoke({})

    # 4. Invoke dtm_heatmap_tool with n_terms=2, n_documents=3
    result = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    # 5. Assertions based on known answer #21 in TEST_FIXTURE.md
    assert result["status"] == "success"
    assert result["terms"] == ["alpha", "always"]
    assert result["document_ids"] == [0, 1, 2]
    expected_slice = [[3, 1], [2, 1], [1, 1]]
    assert result["sliced_matrix"] == expected_slice

    # Verify pending_figure attribute was populated
    assert session.pending_figure is not None
