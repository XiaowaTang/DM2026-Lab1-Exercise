import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

# Ensure punkt is downloaded for word_tokenize
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    @tool
    def inspect_data_tool(n_rows: int = 10) -> Dict[str, Any]:
        """Peeks at the first n_rows of the working DataFrame to inspect columns, content, and data quality.

        Args:
            n_rows: The number of leading rows to inspect (default 10).

        Returns:
            Dict[str, Any] containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - origin_total_rows (int): Total number of rows in the DataFrame.
                - inspected_rows (int): Number of rows actually returned.
                - rows (List[Dict[str, Any]]): Serialized record dicts for the inspected rows.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "session.dataframe is not loaded."}

        origin_total_rows = len(session.dataframe)
        
        # Take n_rows (handle n_rows <= 0)
        actual_n = max(0, n_rows)
        inspected_df = session.dataframe.head(actual_n)
        inspected_rows = len(inspected_df)

        # Convert DataFrame records to clean JSON-serializable list of dicts
        # replacing NaN values with None
        clean_df = inspected_df.where(inspected_df.notna(), None)
        rows = clean_df.to_dict(orient="records")

        summary = {
            "result_id": session.next_result_id("inspect"),
            "status": "success",
            "origin_total_rows": origin_total_rows,
            "inspected_rows": inspected_rows,
            "rows": rows
        }

        session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary, full_report=inspected_df)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes each document's text into unigrams using nltk.word_tokenize on lowercased text.

        Stores unigrams in session.dataframe['unigrams'] and session.artifacts['unigrams'].

        Returns:
            Dict[str, Any] containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - n_documents (int): Total number of tokenized documents.
                - total_tokens (int): Total count of unigrams across all documents.
                - sample_unigrams (List[List[str]]): Sample of unigram token lists for the first few documents.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "session.dataframe is missing or contains no 'text' column."}

        unigrams_list: List[List[str]] = []
        total_tokens = 0

        for text in session.dataframe["text"]:
            if isinstance(text, str) and text.strip():
                tokens = nltk.word_tokenize(text.lower())
            else:
                tokens = []
            unigrams_list.append(tokens)
            total_tokens += len(tokens)

        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        summary = {
            "result_id": session.next_result_id("tokenize"),
            "status": "success",
            "n_documents": len(session.dataframe),
            "total_tokens": total_tokens,
            "sample_unigrams": unigrams_list[:3]
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and folders in a specified subdirectory of the repository root, skipping hidden files/folders.

        Args:
            subdirectory: Path relative to repository root (default '.').

        Returns:
            Dict[str, Any] containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - subdirectory (str): The requested subdirectory path.
                - directories (List[str]): Alphabetically sorted list of subdirectory names.
                - files (List[str]): Alphabetically sorted list of file names.
        """
        target_path = os.path.normpath(subdirectory)

        if not os.path.exists(target_path):
            return {
                "status": "error",
                "message": f"Subdirectory '{subdirectory}' does not exist."
            }

        if not os.path.isdir(target_path):
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' is not a directory."
            }

        entries = [e for e in os.listdir(target_path) if not e.startswith(".")]
        directories = [e for e in entries if os.path.isdir(os.path.join(target_path, e))]
        files = [e for e in entries if os.path.isfile(os.path.join(target_path, e))]

        directories.sort()
        files.sort()

        summary = {
            "result_id": session.next_result_id("list_files"),
            "status": "success",
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks session.dataframe text column for missing, null, empty, or whitespace-only text values.

        Returns:
            Dict[str, Any] containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - n_missing (int): Number of rows with missing text.
                - missing_indices (List[int]): List of document row indices with missing text.
                - has_missing (List[bool]): Boolean indicator list per document row.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "session.dataframe is missing or contains no 'text' column."}

        has_missing: List[bool] = []
        missing_indices: List[int] = []

        for idx, text in enumerate(session.dataframe["text"]):
            is_missing = pd.isna(text) or not isinstance(text, str) or not text.strip()
            has_missing.append(is_missing)
            if is_missing:
                missing_indices.append(idx)

        summary = {
            "result_id": session.next_result_id("missing"),
            "status": "success",
            "n_missing": len(missing_indices),
            "missing_indices": missing_indices,
            "has_missing": has_missing
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for duplicate document texts in session.dataframe, and optionally drops them using keep=False.

        Args:
            drop: If True, drops all copies of duplicated rows using keep=False (default False).

        Returns:
            Dict[str, Any] containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - n_duplicates (int): Total number of duplicate rows detected.
                - remaining_rows (int): Total rows remaining in session.dataframe after optional drop.
                - dropped (bool): True if duplicate rows were dropped, False otherwise.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "session.dataframe is missing or contains no 'text' column."}

        # Calculate number of duplicate rows (pandas default keep='first' counts redundant copies)
        duplicate_mask = session.dataframe.duplicated(subset=["text"], keep="first")
        n_duplicates = int(duplicate_mask.sum())

        if drop and n_duplicates > 0:
            # Per contract: drop ALL copies of duplicated rows using keep=False
            session.dataframe = session.dataframe.drop_duplicates(subset=["text"], keep=False).reset_index(drop=True)
            dropped = True
        else:
            dropped = False

        remaining_rows = len(session.dataframe)

        summary = {
            "result_id": session.next_result_id("duplicates"),
            "status": "success",
            "n_duplicates": n_duplicates,
            "remaining_rows": remaining_rows,
            "dropped": dropped
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes summary statistics on document text length and generates a box plot grouped by category.

        Returns:
            Dict[str, Any] containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - overall (Dict[str, float]): Overall summary statistics of text_length.
                - by_category (Dict[str, Dict[str, float]]): Summary statistics of text_length per category.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "session.dataframe is missing or contains no 'text' column."}

        # Compute character length of each document's text
        df = session.dataframe.copy()
        df["text_length"] = df["text"].apply(lambda x: len(x) if isinstance(x, str) else 0)

        # Compute overall stats
        overall_stats = df["text_length"].describe().to_dict()

        # Determine category column name (category_name or category)
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        by_category_stats: Dict[str, Dict[str, float]] = {}
        if cat_col:
            grouped = df.groupby(cat_col)["text_length"]
            for cat_name, group in grouped:
                by_category_stats[str(cat_name)] = group.describe().to_dict()

        # Generate box plot grouped by category
        fig, ax = plt.subplots(figsize=(8, 5))
        if cat_col:
            sns.boxplot(data=df, x=cat_col, y="text_length", ax=ax)
            ax.set_title("Document Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Document Text Length Distribution")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        # Store pending figure on session
        session.pending_figure = fig

        summary = {
            "result_id": session.next_result_id("describe"),
            "status": "success",
            "overall": overall_stats,
            "by_category": by_category_stats
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 4, random_state: Optional[int] = 42) -> Dict[str, Any]:
        """Subsamples n rows from session.dataframe with a specified random seed.

        Modifies session.dataframe in place to contain only the sampled rows.

        Args:
            n: Number of rows to sample (default 4).
            random_state: Seed for random number generator for reproducibility (default 42).

        Returns:
            Dict[str, Any] containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - n_sampled (int): Number of rows actually sampled.
                - original_indices (List[int]): Row indices from original dataframe that were sampled.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "session.dataframe is not loaded."}

        total_rows = len(session.dataframe)
        sample_count = min(n, total_rows)

        # Sample DataFrame
        sampled_df = session.dataframe.sample(n=sample_count, random_state=random_state)
        original_indices = sampled_df.index.tolist()

        # Update session.dataframe in place with reset index
        session.dataframe = sampled_df.reset_index(drop=True)

        summary = {
            "result_id": session.next_result_id("sample"),
            "status": "success",
            "n_sampled": sample_count,
            "original_indices": original_indices
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    return [inspect_data_tool, tokenize_tool, list_files_tool, check_missing_tool, check_duplicates_tool, describe_data_tool, sample_data_tool]
