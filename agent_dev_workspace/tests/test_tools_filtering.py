import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    # 1. Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Get tools and build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Get variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 4. Invoke variance_filter_tool with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # 5. Assertions based on summary result
    assert result["status"] == "success"
    assert result["initial_terms"] == 5
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1
    assert "always" in result["top_removed_features"]
    assert set(result["top_kept_features"]) == {"alpha", "beta", "delta", "gamma"}

    # 6. Retrieve full_report stored in session.results_store to verify specific variances
    result_id = result["result_id"]
    full_report = session.results_store[result_id]["full_report"]
    assert full_report.columns[0] == "feature"
    assert full_report.columns[1] == "variance"

    # Extract full report records into a lookup map
    var_map = {row["feature"]: (row["variance"], row["keep"]) for _, row in full_report.iterrows()}

    # Known answer #5 expected variances and keep indicators for threshold=0.15
    assert var_map["alpha"][0] == pytest.approx(1.1875, abs=1e-3)
    assert var_map["alpha"][1] is True

    assert var_map["always"][0] == pytest.approx(0.109375, abs=1e-3)
    assert var_map["always"][1] is False  # Removed (0.109375 < 0.15)

    assert var_map["beta"][0] == pytest.approx(0.234375, abs=1e-3)
    assert var_map["beta"][1] is True

    assert var_map["delta"][0] == pytest.approx(0.25, abs=1e-3)
    assert var_map["delta"][1] is True

    assert var_map["gamma"][0] == pytest.approx(1.109375, abs=1e-3)
    assert var_map["gamma"][1] is True


def test_pearson_filter_tool():
    # 1. Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Get pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # 4. Invoke pearson_filter_tool with target_class="catB"
    result = pearson_filter_tool.invoke({"target_class": "catB"})

    # 5. Assertions based on known answer #6 in TEST_FIXTURE.md
    assert result["status"] == "success"
    assert result["target_class"] == "catB"
    assert result["n_terms"] == 5

    # Check top_correlated_features in summary
    assert len(result["top_correlated_features"]) == 5

    # Check full_report stored in results_store
    result_id = result["result_id"]
    full_report = session.results_store[result_id]["full_report"]
    assert full_report.columns[0] == "feature"
    assert full_report.columns[1] == "pearson_r"

    corr_map = {row["feature"]: row["pearson_r"] for _, row in full_report.iterrows()}

    # Known answer #6 Pearson r values for catB target
    assert corr_map["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert corr_map["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corr_map["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corr_map["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corr_map["gamma"] == pytest.approx(0.8307, abs=1e-3)
    
    # Verify sorted by absolute correlation descending (|pearson_r|)
    abs_corrs = full_report["pearson_r"].abs().tolist()
    assert abs_corrs == sorted(abs_corrs, reverse=True)


def test_spearman_filter_tool():
    # 1. Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Get spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # 4. Invoke spearman_filter_tool with target_class="catB"
    result = spearman_filter_tool.invoke({"target_class": "catB"})

    # 5. Assertions based on known answer #6 in TEST_FIXTURE.md
    assert result["status"] == "success"
    assert result["target_class"] == "catB"
    assert result["n_terms"] == 5

    # Check top_correlated_features in summary
    assert len(result["top_correlated_features"]) == 5

    # Check full_report stored in results_store
    result_id = result["result_id"]
    full_report = session.results_store[result_id]["full_report"]
    assert full_report.columns[0] == "feature"
    assert full_report.columns[1] == "spearman_r"

    corr_map = {row["feature"]: row["spearman_r"] for _, row in full_report.iterrows()}

    # Known answer #6 Spearman r values for catB target
    assert corr_map["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert corr_map["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corr_map["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corr_map["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corr_map["gamma"] == pytest.approx(0.9363, abs=1e-3)

    # Verify sorted by absolute correlation descending (|spearman_r|)
    abs_corrs = full_report["spearman_r"].abs().tolist()
    assert abs_corrs == sorted(abs_corrs, reverse=True)
