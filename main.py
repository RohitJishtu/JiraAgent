import streamlit as st
import os
import sys
import json
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from utils.utilis import load_config,build_issue_record
from core.training_view import show_training_viewer
from core.ingest import load_csv
from core.Reference_Issue   import find_reference_issues
from core.Recommended_Actions import find_recommended_actions
CONFIG_PATH = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/config.yml"
Embedding = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/src/embedding"


# Try to import project modules; if not available, show friendly error in UI later
try:
    from core.ingest import load_csv, append_to_json_store
    from core.Index import add_index_new_Data
    from core.Reference_Issue import find_reference_issues
    _CORE_AVAILABLE = True
except Exception as e:
    _CORE_AVAILABLE = False
    _CORE_IMPORT_ERROR = str(e)


# ---------------- Streamlit UI Setup ----------------

st.set_page_config(page_title="QuickRef", layout="wide")

# ---------------- Sidebar (minimal) ----------------
with st.sidebar:
    st.header("Model")
    st.text("all-MiniLM-L6-v2")  # read-only display of model
    if _CORE_AVAILABLE:
        st.success("You are Live")
    else:
        st.error("Core modules missing")
    threshold = st.number_input(
        "Enter Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,    # default
        step=0.01
    )

# ---------------- Main ----------------
st.title("QuickRef : Jira Reference Ticket")
st.markdown("> When you submit an issue, it gets saved safely, processed if needed, and compared with past issues. Then we suggest the most relevant reference tickets for you.")



# initialize state
if "issue_frozen" not in st.session_state:
    st.session_state.issue_frozen = False
    st.session_state.issues = None
mode = st.radio("Input mode", ["Single issue", "Upload CSV"], index=0)
issues: List[Dict[str, Any]] = []

#---------------- Input Section - Single Issue ----------------

if mode == "Single issue":
    st.subheader("Enter Single Issue Details")
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.session_state.issue_frozen:
            st.text_area("Summary *", value=st.session_state.issues.get("Summary", ""), height=120, disabled=True)
        else:
            summary = st.text_area("Summary *", height=120, placeholder="Enter issue summary...")
    with col2:
        if st.session_state.issue_frozen:
            st.text_input("Issue Key *", value=st.session_state.issues.get("Issue key", ""), disabled=True)
        else:
            issue_key = st.text_input("Issue Key *", placeholder="E.g. D6NA**-413**")

    # Second row: optional metadata in one line (issue id, assignee, reporter, priority)
    col3, col4, col5, col6 = st.columns([1, 1, 1, 1])
    with col3:
        if st.session_state.issue_frozen:
            st.text_input("Issue ID (optional)", value=st.session_state.issues.get("Issue id", ""), disabled=True)
        else:
            issue_id = st.text_input("Issue ID (optional)")
    with col4:
        if st.session_state.issue_frozen:
            st.text_input("Assignee (optional)", value=st.session_state.issues.get("Assignee", ""), disabled=True)
        else:
            assignee = st.text_input("Assignee (optional)")
    with col5:
        if st.session_state.issue_frozen:
            st.text_input("Reporter (optional)", value=st.session_state.issues.get("Reporter", ""), disabled=True)
        else:
            reporter = st.text_input("Reporter (optional)")
    with col6:
        if st.session_state.issue_frozen:
            st.selectbox("Priority", ["Low", "Medium", "High", "Critical"], index= ["Low","Medium","High","Critical"].index(st.session_state.issues.get("Priority","Medium")), disabled=True)
        else:
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"], index=1)

    # Bottom row: Submit and Clear side-by-side
    col7, col8 = st.columns([1, 1])
    with col7:
        submit = st.button("Submit", use_container_width=True)
    with col8:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        st.session_state.issue_frozen = False
        st.session_state.issues = None
        st.rerun()

    # handle submit -> freeze data
    if (not st.session_state.issue_frozen) and submit:
        # require summary and issue_key variables from user inputs
        if ('issue_key' in locals() and 'summary' in locals() and issue_key and summary):
            record = build_issue_record(issue_id or "", issue_key, summary, assignee, reporter, priority)
            # freeze and save
            st.session_state.issue_frozen = True
            st.session_state.issues = record
            st.success("✅ Data locked. Now you can run the pipeline.")
            st.rerun()
        else:
            st.warning("⚠️ Please provide both Issue Key and Summary.")

#---------------- Input Section - CSV Input ----------------
else:
    st.subheader("Upload CSV of issues (Jira export header recommended)")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write(f"Loaded {len(df)} rows from uploaded CSV.")
            st.dataframe(df.head(10))
            issues = df.to_dict(orient='records')
            st.success("Prepared uploaded CSV issues for pipeline run.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")

if st.session_state.get("issue_frozen") and st.session_state.get("issues"):
    issues = [st.session_state.issues]


#---------------- Main Pipeline Run  ----------------

st.markdown("---")
st.subheader("Lets find the matching issues")
st.markdown("> The app saves your input, optionally appends/indexes training data, and searches for relevant reference tickets.")

# Run Pipeline button (top-level)
run_button = st.button("Run Pipeline", use_container_width=False)

if run_button:
    # ensure we have prepared issues
    if not issues:
        st.error("No issues prepared. Provide a single issue or upload a CSV first.")
        st.stop()

    if not _CORE_AVAILABLE:
        st.error("Required project modules (core.*) are not importable in this environment. Ensure your repository root is on PYTHONPATH and restart Streamlit.")
        st.stop()

    # Step 1: Save to staging
    staging = "out/staging"
    os.makedirs(staging, exist_ok=True)
    staging_csv = os.path.join(staging, f"input_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv")
    try:
        pd.DataFrame(issues).to_csv(staging_csv, index=False)
        st.info(f"Saved input issues to staging: {staging_csv}")
    except Exception as e:
        st.warning(f"Could not save staging CSV: {e}")

    # Step 2: Load config (optional)
    # cfg_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH
    try:
        cfg = load_config(CONFIG_PATH)
    except Exception as e:
        print("Failed to load config:", e)
        raise SystemExit(1)

    # Step 3 & 4: Append to JSON store and index if TrainingModel enabled in config
    try:
        if cfg.get("TrainingModel"):
            st.info("Config requests training-mode append/index. Appending to JSON store...")
            try:
                append_summary = append_to_json_store(issues, out_path="out/issues_normalized.json", key_field="Issue key")
                st.write("Append summary:")
                st.json(append_summary)
            except Exception as e:
                st.warning(f"append_to_json_store failed: {e}")

            try:
                idx_summary = add_index_new_Data(issues, cfg)
                st.write("Index summary:")
                st.json(idx_summary)
            except Exception as e:
                st.warning(f"add_index_new_Data failed: {e}")
        else:
            st.info("TrainingModel not enabled in config; skipping append/index steps.")
    except Exception as e:
        st.warning(f"Append/index sequence raised: {e}")

    # Step 5: find reference issues (main AI   Functionality)  
    try:
        st.info("Running find_reference_issues to locate similar reference issues...")
        # if find_reference_issues expects a list of dicts, pass 'issues'; some implementations expect different args - adapt as needed.
        
        refs,potential_assignee = find_reference_issues(issues, embeddings_folder=Embedding,
                             model_path="all-MiniLM-L6-v2",
                             top_k=5, score_threshold=threshold or 0.5, debug=False)
    except Exception as e:
        st.error(f"find_reference_issues failed: {e}")
        st.stop()

    # print("potential_assignee potential_assignee)
    # Normalize and display results
    rows = []
    for r in refs:
        input_key = r.get("input_key") or r.get("Issue key") or ""
        input_summary = r.get("input_summary") or r.get("Summary", "")
        for m in r.get("references", []):
            rows.append({
                "input_key": input_key,
                "input_summary": input_summary,
                "match_score": m.get("score"),
                "potential_assignee": potential_assignee,
                "reference_issue_id":  m.get("key"),
                "reference_issue_summary": m.get("summary", "")
            })

    if not rows:
        input_key = issues[0].get("Issue key") 
        input_summary = issues[0].get("Summary")
        rows.append({
                "input_key": input_key,
                "input_summary":input_summary,
                "match_score": None,
                "potential_assignee": potential_assignee,
                "reference_issue_id":  'No Ref Found',
                "reference_issue_summary": "NA"
        })
        st.warning("No reference matches found with the given threshold/top_k.")
    result_df = pd.DataFrame(rows).sort_values(by="match_score", ascending=False)
    st.subheader("Reference matches (sorted by score)")
    st.dataframe(result_df)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download matches CSV", data=csv_bytes, file_name="reference_matches.csv", mime="text/csv")

    # Step 5: Recommended Action

    st.subheader("Recommended Actions")

    for issue in issues:
        issue_key = issue.get("Issue key") or issue.get("key") or issue.get("issue_key")
        if issue_key:
            action = find_recommended_actions(issue_key, json_path="out/issues_normalized.json")
            if action:
                st.markdown(f"{action}")
            else:
                st.markdown(f"**Issue {issue_key}:** No recommended actions found.")
        else:
            st.markdown("Issue key missing; cannot find recommended actions.")


    st.success("Pipeline run complete.")



    st.balloons() # Optional feel good animation
 
#----------------View All JSON Issues in table  ----------------


# Some change in this section WIP


#----------------View All JSON Issues in table  ----------------

show_training_viewer(path="out/issues_normalized.json", max_rows=50, expanded=False)