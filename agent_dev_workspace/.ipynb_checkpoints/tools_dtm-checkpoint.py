from typing import Dict, Any
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

    return [build_dtm_tool]
