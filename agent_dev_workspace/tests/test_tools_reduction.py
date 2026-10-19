import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_pca_known_answer_7():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({
        "method": "pca",
        "n_components": 2
    })

    assert result["status"] == "success"
    assert result["method"] == "pca"
    assert result["n_documents"] == 8
    assert session.pending_figure is not None

    coords = session.artifacts["reduced_coordinates"]
    assert len(coords) == 8

    # Expected PCA 2D coordinates from known answer #7
    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ]

    for i in range(8):
        assert coords[i][0] == pytest.approx(expected_coords[i][0], abs=1e-2)
        assert coords[i][1] == pytest.approx(expected_coords[i][1], abs=1e-2)

    assert result["explained_variance_ratio"][0] == pytest.approx(0.7532, abs=1e-2)
    assert result["explained_variance_ratio"][1] == pytest.approx(0.1965, abs=1e-2)


def test_reduce_dimensions_tsne_known_answer_8():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })

    assert result["status"] == "success"
    assert result["method"] == "tsne"
    assert result["n_documents"] == 8
    assert session.pending_figure is not None

    coords = session.artifacts["reduced_coordinates"]
    assert len(coords) == 8


def test_reduce_dimensions_umap_known_answer_9():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    result = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    assert result["status"] == "success"
    assert result["method"] == "umap"
    assert result["n_documents"] == 8
    assert session.pending_figure is not None

    coords = session.artifacts["reduced_coordinates"]
    assert len(coords) == 8

    # Expected UMAP coordinates from known answer #9 (n_neighbors=3, random_state=42)
    expected_coords = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ]

    for i in range(8):
        assert coords[i][0] == pytest.approx(expected_coords[i][0], abs=1e-1)
        assert coords[i][1] == pytest.approx(expected_coords[i][1], abs=1e-1)


def test_binarize_labels_known_answer_23():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    result = binarize_labels_tool.invoke({})

    assert result["status"] == "success"
    assert result["categories"] == ["catA", "catB"]

    one_hot = result["one_hot"]
    assert len(one_hot) == 8

    # Expected one-hot binary encodings from known answer #23
    # catA -> [1, 0], catB -> [0, 1]
    expected_one_hot = [
        [1, 0],  # row 0: catA
        [1, 0],  # row 1: catA
        [1, 0],  # row 2: catA
        [0, 1],  # row 3: catB
        [0, 1],  # row 4: catB
        [0, 1],  # row 5: catB
        [0, 1],  # row 6: catB
        [1, 0],  # row 7: catA
    ]

    assert one_hot == expected_one_hot
    assert session.artifacts["one_hot"] == expected_one_hot
