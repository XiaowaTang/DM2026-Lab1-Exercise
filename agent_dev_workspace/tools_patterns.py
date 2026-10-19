import os
import tempfile
import builtins
import functools
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI modules
import PAMI.extras.convert.DF2DB as db_converter
import PAMI.frequentPattern.basic.FPGrowth as fpgrowth_module
import PAMI.frequentPattern.topk.FAE as topk_module
import PAMI.frequentPattern.maximal.MaxFPGrowth as maxfpgrowth_module


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mines frequent itemsets/patterns for a specific category using PAMI.

        Args:
            category_name: Category to filter session.dataframe rows.
            filtering_method: Vocabulary filter approach ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI algorithm to use ('fpgrowth', 'topk', or 'maxfpgrowth', default 'fpgrowth').
            min_sup: Minimum support threshold (int, required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-k patterns to find (int, required for 'topk').

        Returns:
            Dict containing:
                - result_id (str): Unique result identifier.
                - status (str): 'success' or 'error'.
                - category_name (str): The mined category.
                - filtering_method (str): The filtering approach used.
                - algorithm (str): Mining algorithm used.
                - n_patterns (int): Total number of mined patterns.
                - patterns (List[Dict[str, Any]]): List of dicts with 'pattern' (List[str]) and 'support' (int).
                - note (Optional[str]): Message if no terms or patterns survived.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "session.dataframe is None."}

        # Validate category column
        cat_col = "category_name" if "category_name" in session.dataframe.columns else ("category" if "category" in session.dataframe.columns else None)
        if cat_col is None:
            return {"status": "error", "message": "No category column found in session.dataframe."}

        # Filter dataframe to category_name
        cat_df = session.dataframe[session.dataframe[cat_col].astype(str) == str(category_name)].copy()
        if len(cat_df) == 0:
            return {"status": "error", "message": f"No documents found for category '{category_name}'."}

        # Validate algorithm inputs
        algorithm = algorithm.lower()
        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            return {"status": "error", "message": f"min_sup is required for algorithm '{algorithm}'."}
        if algorithm == "topk" and k is None:
            return {"status": "error", "message": "k is required for algorithm 'topk'."}

        # Step 1: Fresh category-local CountVectorizer with English stopwords removed
        texts = cat_df["text"].tolist()
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError:
            # Handle empty vocabulary case
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived English stopword removal"
            }

        feature_names = np.array(vectorizer.get_feature_names_out())
        dense_dtm = dtm.toarray()

        if dense_dtm.shape[1] == 0:
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }

        # Step 2: Vocabulary filtering method
        filtering_method = filtering_method.lower()
        kept_indices = []

        if filtering_method == "variance":
            variances = np.var(dense_dtm, axis=0, ddof=0)
            p5 = np.percentile(variances, 5)
            p95 = np.percentile(variances, 95)
            kept_indices = np.where((variances > p5) & (variances < p95))[0]

        elif filtering_method == "term_frequency":
            frequencies = np.sum(dense_dtm, axis=0)
            p5 = np.percentile(frequencies, 5)
            p95 = np.percentile(frequencies, 95)
            kept_indices = np.where((frequencies > p5) & (frequencies < p95))[0]

        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dense_dtm).toarray()
            mean_tfidf = np.mean(tfidf_matrix, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            kept_indices = np.where(mean_tfidf >= p20)[0]

        else:
            return {"status": "error", "message": f"Unknown filtering_method '{filtering_method}'."}

        # Step 3: Handle zero terms kept
        if len(kept_indices) == 0:
            summary = {
                "result_id": session.next_result_id("mine_patterns"),
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        kept_terms = feature_names[kept_indices]
        filtered_dtm = dense_dtm[:, kept_indices]

        # Convert filtered DTM to DataFrame for DF2DB
        filtered_df = pd.DataFrame(filtered_dtm, columns=kept_terms)

        # Step 4: Create transactional DB file safely using UTF-8 open patch
        original_open = builtins.open

        @functools.wraps(original_open)
        def custom_open(*args, **kwargs):
            if "encoding" not in kwargs and (len(args) < 4 or args[3] is None):
                kwargs["encoding"] = "utf-8"
            return original_open(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "transactions.db")

            try:
                builtins.open = custom_open
                converter = db_converter.DF2DB(filtered_df)
                converter.convert2TransactionalDatabase(db_file, ">=", 1)
            finally:
                builtins.open = original_open

            # Step 5: Run chosen PAMI mining algorithm
            if algorithm == "fpgrowth":
                miner = fpgrowth_module.FPGrowth(db_file, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = topk_module.FAE(db_file, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = maxfpgrowth_module.MaxFPGrowth(db_file, minSup=min_sup, sep="\t")
            else:
                return {"status": "error", "message": f"Unknown algorithm '{algorithm}'."}

            miner.mine()

            # Retrieve patterns directly via PAMI's getPatternsAsDataFrame()
            patterns = []
            try:
                patterns_df = miner.getPatternsAsDataFrame()
                if patterns_df is not None and not patterns_df.empty:
                    # Column names in PAMI dataframe are usually 'Patterns' (or 'itemSets') and 'Support' (or 'sup')
                    pat_col = [c for c in patterns_df.columns if "pattern" in c.lower() or "item" in c.lower()]
                    sup_col = [c for c in patterns_df.columns if "sup" in c.lower() or "count" in c.lower()]

                    p_col_name = pat_col[0] if pat_col else patterns_df.columns[0]
                    s_col_name = sup_col[0] if sup_col else patterns_df.columns[1]

                    for _, row in patterns_df.iterrows():
                        pat_val = row[p_col_name]
                        if isinstance(pat_val, str):
                            itemset = pat_val.strip().split()
                        elif isinstance(pat_val, (list, tuple)):
                            itemset = list(pat_val)
                        else:
                            itemset = [str(pat_val)]

                        patterns.append({
                            "pattern": itemset,
                            "support": int(row[s_col_name])
                        })
            except Exception:
                patterns = []

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        summary = {
            "result_id": session.next_result_id("mine_patterns"),
            "status": "success",
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns),
            "patterns": patterns,
            "note": "no patterns found" if len(patterns) == 0 else None
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary
        )
        return summary

    return [mine_patterns_tool]
