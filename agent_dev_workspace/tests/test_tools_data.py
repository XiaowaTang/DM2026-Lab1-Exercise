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


def test_tokenize_tool():
    # 1. Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Get tokenize_tool from tools_data
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # 3. Invoke tokenize_tool
    result = tokenize_tool.invoke({})

    # Expected token lists per known answer #10 in TEST_FIXTURE.md
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    # 4. Assertions
    assert result["status"] == "success"
    assert result["n_documents"] == 8
    assert session.dataframe["unigrams"].tolist() == expected_tokens
    assert session.artifacts["unigrams"] == expected_tokens


def test_list_files_tool():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Known answer #11 in TEST_FIXTURE.md: list_files_tool(subdirectory="newdataset")
    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert result["status"] == "success"
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]


def test_check_missing_tool():
    # 1. Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Get check_missing_tool from tools_data
    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # 3. Invoke check_missing_tool
    result = check_missing_tool.invoke({})

    # 4. Assertions based on known answer #2 in TEST_FIXTURE.md
    assert result["status"] == "success"
    assert result["n_missing"] == 1
    assert result["missing_indices"] == [7]
    assert result["has_missing"] == [False, False, False, False, False, False, False, True]


def test_check_duplicates_tool():
    # 1. Test check_duplicates_tool without dropping
    session1 = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools1 = make_tools(session1)
    check_duplicates_tool1 = next(t for t in data_tools1 if t.name == "check_duplicates_tool")

    result1 = check_duplicates_tool1.invoke({"drop": False})

    # Known answer #3 in TEST_FIXTURE.md (without drop): 1 duplicate, 8 remaining rows
    assert result1["status"] == "success"
    assert result1["n_duplicates"] == 1
    assert result1["remaining_rows"] == 8
    assert result1["dropped"] is False

    # 2. Test check_duplicates_tool with drop=True on a fresh session state
    session2 = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools2 = make_tools(session2)
    check_duplicates_tool2 = next(t for t in data_tools2 if t.name == "check_duplicates_tool")

    result2 = check_duplicates_tool2.invoke({"drop": True})

    # Known answer #3 in TEST_FIXTURE.md (with drop): keep=False drops both copies, leaving 6 rows
    assert result2["status"] == "success"
    assert result2["n_duplicates"] == 1
    assert result2["remaining_rows"] == 6
    assert result2["dropped"] is True
    assert len(session2.dataframe) == 6


def test_describe_data_tool():
    # 1. Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Get describe_data_tool from tools_data
    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    # 3. Invoke describe_data_tool
    result = describe_data_tool.invoke({})

    # 4. Assertions based on known answer #18 in TEST_FIXTURE.md
    assert result["status"] == "success"

    # Overall stats assertions: count=8, mean=19.875, std=9.4330, min=0, 25%=17.75, 50%=20.5, 75%=25.25, max=30
    overall = result["overall"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    # Per-category stats assertions:
    # catA: count=4, mean=17.25, std=12.5, min=0, 25%=12.75, 50%=20.0, 75%=24.5, max=29
    # catB: count=4, mean=22.5, std=5.7446, min=18, 25%=18.0, 50%=21.0, 75%=25.5, max=30
    by_cat = result["by_category"]
    assert "catA" in by_cat
    assert "catB" in by_cat

    catA = by_cat["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == 29

    catB = by_cat["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == 30

    # Verify plot was assigned to session.pending_figure
    assert session.pending_figure is not None


def test_sample_data_tool():
    # 1. Create session and load sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Get sample_data_tool from tools_data
    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # 3. Invoke sample_data_tool with n=4, random_state=42
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # 4. Assertions based on known answer #13 in TEST_FIXTURE.md
    assert result["status"] == "success"
    assert result["n_sampled"] == 4
    assert result["original_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4
