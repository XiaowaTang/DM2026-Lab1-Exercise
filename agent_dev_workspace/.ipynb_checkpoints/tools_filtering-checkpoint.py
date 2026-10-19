from typing import Dict, Any, List
import pandas as pd
import numpy as np
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.15) -> Dict[str, Any]:
        """Flags near-constant terms by variance for removal based on session.feature_matrix.

        Args:
            threshold: Minimum variance threshold to keep a term (default 0.15).

        Returns:
            Dict containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - threshold (float): Minimum variance threshold applied.
                - initial_terms (int): Total number of terms before filtering.
                - n_kept (int): Number of terms kept.
                - n_removed (int): Number of terms removed.
                - var_df (List[Dict[str, Any]]): List of dict records with 'feature', 'variance', and 'keep'.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "session.feature_matrix or session.feature_names is None. Run build_dtm_tool first."
            }

        # Convert sparse matrix to dense array to compute population variance per column
        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0, ddof=0)

        # Build DataFrame with feature, variance, and keep boolean indicator
        df_list = []
        for term, var in zip(session.feature_names, variances):
            var_val = float(round(var, 6))
            df_list.append({
                "feature": term,
                "variance": var_val,
                "keep": bool(var >= threshold)
            })

        var_df = pd.DataFrame(df_list)

        # Per contract: full_report must have feature as 1st col, variance as 2nd col, sorted for visualization
        sorted_full_report = var_df.sort_values(by="variance", ascending=False).reset_index(drop=True)

        n_kept = int(var_df["keep"].sum())
        n_removed = len(var_df) - n_kept

        summary = {
            "result_id": session.next_result_id("variance_filter"),
            "status": "success",
            "threshold": threshold,
            "initial_terms": len(session.feature_names),
            "n_kept": n_kept,
            "n_removed": n_removed,
            "var_df": var_df.to_dict(orient="records")
        }

        # store_result with sorted full_report dataframe matching contract requirement
        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=sorted_full_report)
        return summary

    return [variance_filter_tool]
