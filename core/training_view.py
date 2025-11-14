"""
training_viewer.py

Simple Streamlit module to load, view, filter and download training issues
stored as JSON (list-of-dicts). Designed to be imported and called from your
main Streamlit app.

Usage (in your app):
from training_viewer import show_training_viewer
show_training_viewer(path="out/issues_normalized.json", max_rows=5000)

The function will render an expander with search, filters and a download button.
"""

from typing import Optional
import os
import pandas as pd
import streamlit as st


def _safe_read_json(path: str) -> pd.DataFrame:
    """Return a DataFrame loaded from JSON path. Empty DataFrame on error."""
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_json(path)
        # If JSON decodes to a list -> pandas may already give DataFrame
        if isinstance(df, list):
            df = pd.DataFrame(df)
        # Ensure it's a DataFrame
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        return df
    except Exception as e:
        st.warning(f"Could not read JSON training file: {e}")
        return pd.DataFrame()


def show_training_viewer(path: Optional[str] = None, max_rows: int = 5000, expanded: bool = False):
    """
    Render the training data viewer UI in Streamlit.

    Args:
        path: path to the JSON file containing training issues (list-of-dicts).
              If None or missing file -> shows "no training issues" message.
        max_rows: maximum rows to render in the DataFrame (prevents UI freezes).
        expanded: whether the expander should be open by default.
    """
    st.markdown("### Training data viewer")
    TRAINING_JSON_PATH = path or "out/issues_normalized.json"

    with st.expander("View training data (issues used for training / append)", expanded=expanded):
        df_train = _safe_read_json(TRAINING_JSON_PATH)
        if df_train.empty:
            st.info(f"No training issues found at: {TRAINING_JSON_PATH}")
            return

        st.write(f"Loaded **{len(df_train)}** training records from `{TRAINING_JSON_PATH}`")

        # Build quick filters for common fields if present
        cols_available = list(df_train.columns)

        # Two-column layout: search on left, column filters on right
        left, right = st.columns([2, 1])
        with left:
            search_txt = st.text_input("Search Issue key or Summary (substring)", value="", key="training_search")
        with right:
            # only show filters for columns present and with reasonable cardinality
            filters = {}
            for c in ("Priority", "Assignee", "Reporter", "Status"):
                if c in df_train.columns:
                    uniques = df_train[c].dropna().astype(str).unique()
                    # limit choices for extremely high-cardinality columns
                    if len(uniques) > 200:
                        sel = st.multiselect(f"{c} filter (many values)", options=sorted(list(set(uniques)))[:200], key=f"filter_{c}")
                    else:
                        sel = st.multiselect(f"{c} filter", options=sorted([str(x) for x in uniques]), key=f"filter_{c}")
                    if sel:
                        filters[c] = sel

        # Apply search (case-insensitive) across Issue key and Summary if present
        filtered = df_train
        if search_txt:
            mask = pd.Series(False, index=filtered.index)
            for col in ("Issue key", "Summary"):
                if col in filtered.columns:
                    mask = mask | filtered[col].astype(str).str.contains(search_txt, case=False, na=False)
            filtered = filtered[mask]

        # Apply other filters
        for c, sel in filters.items():
            filtered = filtered[filtered[c].astype(str).isin(sel)]

        st.write(f"Showing **{len(filtered)}** records after filtering")

        # Safety rendering: avoid overwhelming the browser
        if len(filtered) > max_rows:
            st.warning(f"Filtered set is large ({len(filtered)} rows). Showing first {max_rows} rows only.")
            display_df = filtered.head(max_rows)
        else:
            display_df = filtered

        # Reset index for display clarity
        st.dataframe(display_df.reset_index(drop=True))

        # Provide a download button for the filtered set
        try:
            csv_bytes = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download filtered as CSV",
                data=csv_bytes,
                file_name="training_issues_filtered.csv",
                mime="text/csv",
                key="download_training_csv"
            )
        except Exception as e:
            st.warning(f"Could not prepare download: {e}")
