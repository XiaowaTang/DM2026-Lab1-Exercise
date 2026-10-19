from typing import Dict, Any
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def inspect_data_tool(n_rows: int = 10) -> Dict[str, Any]:
        """Peeks at the first n_rows of the working DataFrame to inspect columns, content, and data quality.

        Args:
            n_rows: The number of leading rows to inspect (default 10).

        Returns:
            Dict containing result_id, origin_total_rows, inspected_rows, and rows records.
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

    return [inspect_data_tool]
