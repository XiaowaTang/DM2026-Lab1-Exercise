import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tool_fpgrowth_known_answer_17():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Case 1: filtering_method="variance" keeps 0 terms
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["status"] == "success"
    assert res_var["n_patterns"] == 0
    assert "no terms survived" in res_var["note"]

    # Case 2: filtering_method="term_frequency" keeps 0 terms
    res_tf = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf["status"] == "success"
    assert res_tf["n_patterns"] == 0
    assert "no terms survived" in res_tf["note"]

    # Case 3: filtering_method="tfidf", algorithm="fpgrowth", catA, min_sup=2
    res_tfidf_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 2
    })
    assert res_tfidf_catA["status"] == "success"
    assert res_tfidf_catA["n_patterns"] == 1
    assert res_tfidf_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_tfidf_catA["patterns"][0]["support"] == 3

    # Case 4: filtering_method="tfidf", algorithm="fpgrowth", catB, min_sup=2
    res_tfidf_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 2
    })
    assert res_tfidf_catB["status"] == "success"
    assert res_tfidf_catB["n_patterns"] == 1
    assert res_tfidf_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_tfidf_catB["patterns"][0]["support"] == 4


def test_mine_patterns_tool_topk_and_maxfpgrowth_known_answer_20():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Case 1: algorithm="topk", k=1, catA
    res_topk_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_catA["status"] == "success"
    assert res_topk_catA["n_patterns"] == 1
    assert res_topk_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk_catA["patterns"][0]["support"] == 3

    # Case 2: algorithm="topk", k=1, catB
    res_topk_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_catB["status"] == "success"
    assert res_topk_catB["n_patterns"] == 1
    assert res_topk_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_topk_catB["patterns"][0]["support"] == 4

    # Case 3: algorithm="maxfpgrowth", min_sup=1 -> returns 0 patterns on single-term DB per PAMI design
    res_max_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catA["status"] == "success"
    assert res_max_catA["n_patterns"] == 0

    res_max_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catB["status"] == "success"
    assert res_max_catB["n_patterns"] == 0
