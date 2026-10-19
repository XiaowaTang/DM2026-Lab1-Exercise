from typing import Dict, Any, List, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Reduces document-term matrix dimensionality to 2D coordinates and generates a category scatter plot.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', or 'umap', default 'pca').
            n_components: Number of dimensions to reduce to (default 2).
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors for UMAP (default 15).
            random_state: Random seed for reproducibility (default 42).

        Returns:
            Dict containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - method (str): Dimensionality reduction method used.
                - n_documents (int): Total number of documents reduced.
                - sample_coordinates (List[List[float]]): Sample reduced coordinates for first 10 documents.
                - explained_variance_ratio (Optional[List[float]]): Variance ratio explained (PCA only).
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "session.feature_matrix is None. Run build_dtm_tool first."}

        # Extract document categories for dynamic coloring
        labels = None
        if session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values

        if labels is None:
            labels = ["Uncategorized"] * session.feature_matrix.shape[0]

        labels = [str(lbl) for lbl in labels]
        dense_matrix = session.feature_matrix.toarray()
        method_clean = method.lower()
        explained_variance_ratio = None

        # Compute reduction coordinates
        if method_clean == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(dense_matrix)
            explained_variance_ratio = [float(round(v, 4)) for v in reducer.explained_variance_ratio_]

        elif method_clean == "tsne":
            n_samples = dense_matrix.shape[0]
            actual_perplexity = min(perplexity, max(1.0, float(n_samples - 1)))
            reducer = TSNE(
                n_components=n_components,
                perplexity=actual_perplexity,
                random_state=random_state
            )
            coords = reducer.fit_transform(dense_matrix)

        elif method_clean == "umap":
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(dense_matrix)

        else:
            return {"status": "error", "message": f"Unknown dimensionality reduction method '{method}'."}

        coords_list = [[float(round(x, 4)) for x in row] for row in coords]

        # Generate scatter plot colored by category dynamically
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=[c[0] for c in coords_list],
            y=[c[1] for c in coords_list],
            hue=labels,
            palette="tab10",
            s=100,
            ax=ax
        )
        ax.set_title(f"Dimensionality Reduction ({method_clean.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Set mandatory pending figure on session
        session.pending_figure = fig

        # Save artifacts and full report dataframe
        session.artifacts["reduced_coordinates"] = coords_list
        coords_df = pd.DataFrame(coords_list, columns=[f"Dim{i+1}" for i in range(n_components)])
        coords_df["category"] = labels

        summary = {
            "result_id": session.next_result_id("reduce_dimensions"),
            "status": "success",
            "method": method_clean,
            "n_documents": len(coords_list),
            "sample_coordinates": coords_list[:10],
            "explained_variance_ratio": explained_variance_ratio
        }

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state
            },
            summary,
            full_report=coords_df
        )
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Converts category labels in session.dataframe['category_name'] into one-hot binary encodings.

        Stores the one-hot binary encodings in session.dataframe['one_hot'] and session.artifacts['one_hot'].

        Returns:
            Dict[str, Any] containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - categories (List[str]): List of category labels corresponding to binary columns.
                - one_hot (List[List[int]]): One-hot binary encoding matrix per document.
        """
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"status": "error", "message": "session.dataframe is missing or contains no 'category_name' column."}

        # Call set_labels to ensure session.labels and session.categories are populated
        raw_labels = session.dataframe["category_name"].values
        session.set_labels(raw_labels)

        # Build categories list in order
        categories = session.categories if session.categories else sorted(list(set(raw_labels)))

        # One-hot encode each document label based on categories order
        one_hot_matrix: List[List[int]] = []
        for lbl in raw_labels:
            one_hot_row = [1 if str(lbl) == str(cat) else 0 for cat in categories]
            one_hot_matrix.append(one_hot_row)

        session.dataframe["one_hot"] = one_hot_matrix
        session.artifacts["one_hot"] = one_hot_matrix

        summary = {
            "result_id": session.next_result_id("binarize_labels"),
            "status": "success",
            "categories": [str(c) for c in categories],
            "one_hot": one_hot_matrix
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
