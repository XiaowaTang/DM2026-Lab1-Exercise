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
                - coordinates (List[List[float]]): Reduced n_componentsD coordinates per document.
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
            # Adjust perplexity if larger than n_samples - 1
            n_samples = dense_matrix.shape[0]
            actual_perplexity = min(perplexity, max(1.0, float(n_samples - 1)))
            reducer = TSNE(n_components=n_components, perplexity=actual_perplexity, random_state=random_state)
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

        # Save artifacts and summary
        session.artifacts["reduced_coordinates"] = coords_list

        summary = {
            "result_id": session.next_result_id("reduce_dimensions"),
            "status": "success",
            "method": method_clean,
            "coordinates": coords_list,
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
            summary
        )
        return summary

    return [reduce_dimensions_tool]
