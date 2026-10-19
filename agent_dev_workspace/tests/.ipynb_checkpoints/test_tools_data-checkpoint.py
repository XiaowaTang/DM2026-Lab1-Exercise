import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_inspect_data_tool():
    # 1. Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Get inspect_data_tool from tools_data
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # 3. Invoke inspect_data_tool with n_rows=3
    result = inspect_data_tool.invoke({"n_rows": 3})

    # 4. Assertions based on known answer #22 in TEST_FIXTURE.md
    assert result["origin_total_rows"] == 8
    assert result["inspected_rows"] == 3
    assert len(result["rows"]) == 3

    # Verify content of row 0
    assert result["rows"][0]["text"] == "always alpha alpha alpha beta"
