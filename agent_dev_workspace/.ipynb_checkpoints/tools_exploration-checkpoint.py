from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_idx1: int, doc_idx2: int) -> Dict[str, Any]:
        """Computes cosine similarity between two documents in session.feature_matrix using raw counts.

        Args:
            doc_idx1: Zero-based row index of the first document.
            doc_idx2: Zero-based row index of the second document.

        Returns:
            Dict[str, Any] containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - doc_idx1 (int): Index of document 1.
                - doc_idx2 (int): Index of document 2.
                - cosine_similarity (float): Cosine similarity score between 0.0 and 1.0.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "session.feature_matrix is None. Run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]
        if doc_idx1 < 0 or doc_idx1 >= n_docs:
            return {"status": "error", "message": f"doc_idx1 {doc_idx1} is out of bounds (total documents: {n_docs})."}
        if doc_idx2 < 0 or doc_idx2 >= n_docs:
            return {"status": "error", "message": f"doc_idx2 {doc_idx2} is out of bounds (total documents: {n_docs})."}

        # Extract document count vectors from feature_matrix
        vec1 = session.feature_matrix[doc_idx1].toarray()
        vec2 = session.feature_matrix[doc_idx2].toarray()

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Handle zero vector case cleanly (e.g., empty text row) as 0.0 cosine similarity
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            sim_matrix = cosine_similarity(vec1, vec2)
            similarity = float(np.round(sim_matrix[0][0], 4))

        summary = {
            "result_id": session.next_result_id("cosine"),
            "status": "success",
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "cosine_similarity": similarity
        }

        session.store_result("cosine_similarity_tool", {"doc_idx1": doc_idx1, "doc_idx2": doc_idx2}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """Computes Pearson correlation matrix across top terms by variance in session.feature_matrix.

        Args:
            top_n: Maximum number of highest variance terms to include (default 20).

        Returns:
            Dict[str, Any] containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - top_terms (List[str]): Terms selected sorted descending by variance.
                - correlation_matrix (List[List[float]]): 2D list of Pearson correlation coefficients.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "session.feature_matrix or session.feature_names is None. Run build_dtm_tool first."
            }

        matrix_dense = session.feature_matrix.toarray()
        feature_names = list(session.feature_names)
        n_terms = len(feature_names)

        if n_terms == 0:
            return {"status": "error", "message": "session.feature_names is empty."}

        # 1. Compute variance per term across all documents
        variances = np.var(matrix_dense, axis=0, ddof=0)

        # 2. Sort indices by variance descending
        sorted_indices = np.argsort(-variances)
        num_to_keep = min(top_n, n_terms)
        selected_indices = sorted_indices[:num_to_keep]

        selected_terms = [feature_names[i] for i in selected_indices]
        selected_submatrix = matrix_dense[:, selected_indices]

        # 3. Compute Pearson correlation matrix across selected term columns
        corr_matrix = np.corrcoef(selected_submatrix, rowvar=False)

        # Handle case where only 1 term is selected (np.corrcoef returns scalar)
        if num_to_keep == 1:
            corr_matrix = np.array([[1.0]])

        # Handle NaNs from zero-variance columns cleanly
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix_rounded = np.round(corr_matrix, 4).tolist()

        # 4. Generate heatmap plot and attach to session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f" if num_to_keep <= 5 else ".2f",
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            cmap="vlag",
            center=0,
            ax=ax
        )
        ax.set_title(f"Feature Correlation Matrix (Top {num_to_keep} Terms by Variance)")
        plt.tight_layout()
        session.pending_figure = fig

        summary = {
            "result_id": session.next_result_id("feat_corr"),
            "status": "success",
            "top_terms": selected_terms,
            "correlation_matrix": corr_matrix_rounded
        }

        session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
