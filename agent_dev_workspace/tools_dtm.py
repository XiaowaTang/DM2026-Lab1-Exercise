from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool() -> Dict[str, Any]:
        """Build a Document-Term Matrix (DTM) from session.dataframe text column.

        Computes count vectorization on document text, stores feature_matrix,
        feature_names, and count_vectorizer artifact on session, and reports matrix sparsity.

        Returns:
            Dict containing result_id, dtm_shape, vocabulary_size, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "session.dataframe is missing or contains no 'text' column."}

        # Fill missing text values with empty strings to avoid vectorizer errors
        text_data = session.dataframe["text"].fillna("")

        # Fit CountVectorizer with default parameters
        vectorizer = CountVectorizer()
        feature_matrix = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state attributes
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())

        # Compute sparsity metrics matching Master's specification
        n_rows, n_cols = feature_matrix.shape
        total_elements = n_rows * n_cols
        non_zero = int(feature_matrix.nnz)
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        summary = {
            "result_id": session.next_result_id("dtm"),
            "status": "success",
            "dtm_shape": [n_rows, n_cols],
            "vocabulary_size": len(feature_names),
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "vocabulary_sample": feature_names[:10]
        }

        session.store_result("build_dtm_tool", {}, summary)
        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Aggregates total term frequencies across all documents in the document-term matrix.

        Requires session.feature_matrix and session.feature_names to be already computed via build_dtm_tool.

        Returns:
            Dict containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - total_terms (int): Total number of terms in vocabulary.
                - top_terms (List[Dict[str, Any]]): Top 10 terms sorted by frequency descending.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "session.feature_matrix or session.feature_names is None. Run build_dtm_tool first."
            }

        # Sum total counts across all documents (columns of DTM)
        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        term_frequencies = {term: int(count) for term, count in zip(session.feature_names, counts)}

        # Sort terms by frequency descending
        sorted_terms = sorted(term_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_terms = [{"term": term, "frequency": freq} for term, freq in sorted_terms]

        # Store full report dataframe for visualization/downstream tools
        report_df = pd.DataFrame(sorted_terms, columns=["term", "frequency"])

        summary = {
            "result_id": session.next_result_id("term_freq"),
            "status": "success",
            "total_terms": len(session.feature_names),
            "top_terms": top_terms[:10]
        }

        session.store_result("term_frequency_tool", {}, summary, full_report=report_df)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a raw positional heatmap slice of the Document-Term Matrix (DTM).

        Requires session.feature_matrix and session.feature_names to be already computed via build_dtm_tool.

        Args:
            n_terms: Number of terms (columns) to slice from the beginning (default 20).
            n_documents: Number of documents (rows) to slice from the beginning (default 20).

        Returns:
            Dict containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - sliced_matrix (List[List[int]]): Sliced raw term counts as a nested list.
                - terms (List[str]): Terms included in the slice.
                - document_ids (List[int]): Document indices included in the slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "session.feature_matrix or session.feature_names is None. Run build_dtm_tool first."
            }

        total_docs, total_terms = session.feature_matrix.shape

        # Bound slicing parameters
        actual_docs = min(max(1, n_documents), total_docs)
        actual_terms = min(max(1, n_terms), total_terms)

        # Slice matrix and convert to dense array
        matrix_slice = session.feature_matrix[:actual_docs, :actual_terms].toarray()
        sliced_terms = session.feature_names[:actual_terms]
        doc_indices = list(range(actual_docs))

        # Generate Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            matrix_slice,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=doc_indices,
            ax=ax
        )
        ax.set_title(f"Document-Term Matrix Heatmap (Top {actual_docs} Docs, Top {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        summary = {
            "result_id": session.next_result_id("dtm_heatmap"),
            "status": "success",
            "sliced_matrix": matrix_slice.tolist(),
            "terms": sliced_terms,
            "document_ids": doc_indices
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
