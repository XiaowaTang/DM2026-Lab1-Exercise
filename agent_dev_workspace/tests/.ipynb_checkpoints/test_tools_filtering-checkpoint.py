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

    # 5. Assertions based on known answer #5 in TEST_FIXTURE.md
    assert result["status"] == "success"
    assert result["initial_terms"] == 5
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1

    # Extract var_df records into a lookup map
    var_map = {row["feature"]: (row["variance"], row["keep"]) for row in result["var_df"]}

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
