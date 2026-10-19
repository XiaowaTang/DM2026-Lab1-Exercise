from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
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

    @tool
    def pearson_filter_tool(target_class: str) -> Dict[str, Any]:
        """Computes Pearson correlation between each DTM feature count and a binary target class indicator.

        Args:
            target_class: The category label to treat as the positive target class (one-vs-rest).

        Returns:
            Dict containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - target_class (str): The target class label used for correlation.
                - n_terms (int): Total number of terms evaluated.
                - correlation_df (List[Dict[str, Any]]): List of dict records with 'feature' and 'pearson_r'.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "session.feature_matrix or session.feature_names is None. Run build_dtm_tool first."
            }

        # Get document labels
        labels = None
        if session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values

        if labels is None:
            return {
                "status": "error",
                "message": "No document labels found in session. Ensure dataset with category labels is loaded."
            }

        # Create binary target indicator y (1 for target_class, 0 otherwise)
        y = np.array([1 if str(lbl) == str(target_class) else 0 for lbl in labels], dtype=float)

        if np.all(y == 0):
            return {
                "status": "error",
                "message": f"Target class '{target_class}' was not found in document labels."
            }

        dense_matrix = session.feature_matrix.toarray()
        records = []

        for idx, term in enumerate(session.feature_names):
            x = dense_matrix[:, idx].astype(float)
            if np.std(x) == 0 or np.std(y) == 0:
                r_val = 0.0
            else:
                r_val, _ = stats.pearsonr(x, y)
                if np.isnan(r_val):
                    r_val = 0.0

            r_val = float(round(r_val, 6))
            records.append({
                "feature": term,
                "pearson_r": r_val
            })

        corr_df = pd.DataFrame(records)

        # Per contract: full_report must have feature as 1st col, pearson_r as 2nd col, sorted by absolute correlation for visualization
        sorted_full_report = corr_df.reindex(
            corr_df["pearson_r"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

        summary = {
            "result_id": session.next_result_id("pearson_filter"),
            "status": "success",
            "target_class": target_class,
            "n_terms": len(session.feature_names),
            "correlation_df": corr_df.to_dict(orient="records")
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class}, summary, full_report=sorted_full_report)
        return summary

    @tool
    def spearman_filter_tool(target_class: str) -> Dict[str, Any]:
        """Computes Spearman rank correlation between each DTM feature count and a binary target class indicator.

        Args:
            target_class: The category label to treat as the positive target class (one-vs-rest).

        Returns:
            Dict containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - target_class (str): The target class label used for correlation.
                - n_terms (int): Total number of terms evaluated.
                - correlation_df (List[Dict[str, Any]]): List of dict records with 'feature' and 'spearman_r'.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "session.feature_matrix or session.feature_names is None. Run build_dtm_tool first."
            }

        # Get document labels
        labels = None
        if session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values

        if labels is None:
            return {
                "status": "error",
                "message": "No document labels found in session. Ensure dataset with category labels is loaded."
            }

        # Create binary target indicator y (1 for target_class, 0 otherwise)
        y = np.array([1 if str(lbl) == str(target_class) else 0 for lbl in labels], dtype=float)

        if np.all(y == 0):
            return {
                "status": "error",
                "message": f"Target class '{target_class}' was not found in document labels."
            }

        dense_matrix = session.feature_matrix.toarray()
        records = []

        for idx, term in enumerate(session.feature_names):
            x = dense_matrix[:, idx].astype(float)
            if np.std(x) == 0 or np.std(y) == 0:
                r_val = 0.0
            else:
                res = stats.spearmanr(x, y)
                r_val = res.statistic if hasattr(res, "statistic") else res[0]
                if np.isnan(r_val):
                    r_val = 0.0

            r_val = float(round(r_val, 6))
            records.append({
                "feature": term,
                "spearman_r": r_val
            })

        corr_df = pd.DataFrame(records)

        # Per contract: full_report must have feature as 1st col, spearman_r as 2nd col, sorted by absolute correlation for visualization
        sorted_full_report = corr_df.reindex(
            corr_df["spearman_r"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

        summary = {
            "result_id": session.next_result_id("spearman_filter"),
            "status": "success",
            "target_class": target_class,
            "n_terms": len(session.feature_names),
            "correlation_df": corr_df.to_dict(orient="records")
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class}, summary, full_report=sorted_full_report)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
