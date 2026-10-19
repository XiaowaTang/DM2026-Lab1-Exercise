import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool():
    # Create a fresh session state for the test
    session = SessionState()

    # Load premade tools from the premade_tools directory
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Execute load_dataset_tool on sample_fixture.csv
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Assertions based on known answer #1 in TEST_FIXTURE.md
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}

    # Verify session state dataframe holds the loaded dataset
    assert session.dataframe is not None
    assert len(session.dataframe) == 8
