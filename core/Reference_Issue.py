import os
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import math
from core.Pot_Assignee import assign_lru

def _safe_key(issue: Dict[str, Any], key_field: str) -> str:
    return str(issue.get(key_field) or issue.get("Issue id") or issue.get("Summary") or "")

def find_reference_issues(
    issues: List[Dict[str, Any]],
    embeddings_folder: str,
    model_path: str = "all-MiniLM-L6-v2",
    embeddings_file: str = "embeddings.npy",
    meta_files: Optional[List[str]] = None,
    key_field: str = "Issue key",
    top_k: int = 3,
    score_threshold: float = 0.5,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Debug version — returns same structure plus diagnostics.
    """
    if meta_files is None:
        meta_files = ["annoy_meta.json", "meta.json", "annoy_meta.json"]

    emb_path = os.path.join(embeddings_folder, embeddings_file)
    meta_path = None
    for m in meta_files:
        cand = os.path.join(embeddings_folder, m)
        if os.path.exists(cand):
            meta_path = cand
            break

    if debug:
        print("DEBUG: emb_path =", emb_path)
        print("DEBUG: meta_path =", meta_path)

    # quick failsafe: report clearly if files missing
    if not os.path.exists(emb_path) or meta_path is None:
        if debug:
            print("Missing files:", {"emb_path_exists": os.path.exists(emb_path), "meta_path_found": meta_path})
        return [{
            "input_key": _safe_key(i, key_field),
            "input_summary": i.get("Summary", ""),
            "references": [],
            "diagnostics": {"missing_files": True, "emb_path_exists": os.path.exists(emb_path), "meta_path": meta_path}
        } for i in issues]

    # load stored embeddings and metadata
    stored_embs = np.load(emb_path)
    if stored_embs.ndim != 2:
        raise ValueError("embeddings.npy must be 2D")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if debug:
        print("Loaded stored_embs shape:", stored_embs.shape)
        try:
            sample_meta_keys = list(meta.keys())[:5]
        except Exception:
            sample_meta_keys = []
        print("Meta entries (first keys):", sample_meta_keys)
        if sample_meta_keys:
            for k in sample_meta_keys:
                print(" meta[{}] = {}".format(k, meta.get(k)))

    # ensure meta map indices correspond to embedding rows
    # collect numeric indices present in meta
    try:
        meta_indices = sorted([int(k) for k in meta.keys()])
    except Exception:
        meta_indices = []
    if debug:
        print("Meta numeric indices (first 10):", meta_indices[:10])

    # shape checks
    n_stored, stored_dim = stored_embs.shape
    if meta_indices and (meta_indices[-1] + 1 != n_stored or len(meta_indices) != n_stored):
        if debug:
            print("WARNING: meta indices/count don't match stored embeddings shape.")
            print("  meta_count =", len(meta_indices), "n_stored =", n_stored, "max_meta_idx =", (meta_indices[-1] if meta_indices else None))

    # normalize stored embeddings
    stored_embs = stored_embs.astype("float32")
    s_norms = np.linalg.norm(stored_embs, axis=1, keepdims=True)
    s_norms[s_norms == 0] = 1e-9
    stored_embs = stored_embs / s_norms

    # prepare lookup
    idx_to_meta = {int(k): v for k, v in meta.items()}

    # embed incoming summaries (batch)
    model = SentenceTransformer(model_path)
    summaries = [i.get("Summary", "") for i in issues]
    if not summaries:
        return []

    q_embs = model.encode(summaries, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    q_norms = np.linalg.norm(q_embs, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1e-9
    q_embs = q_embs / q_norms

    # dimension check
    q_n, q_dim = q_embs.shape
    if debug:
        print("Query embeddings shape:", q_embs.shape, "Stored embeddings shape:", stored_embs.shape)
    if q_dim != stored_dim:
        # this is a critical mismatch
        if debug:
            print("CRITICAL: model output dim (query) != stored embedding dim")
        # still compute dot but warn user
        diag_all = []
        for i, issue in enumerate(issues):
            diag_all.append({
                "input_key": _safe_key(issue, key_field),
                "diagnostics": {"dim_mismatch": True, "q_dim": q_dim, "stored_dim": stored_dim}
            })
        # return early with diagnostics
        return [{
            "input_key": _safe_key(issue, key_field),
            "input_summary": issue.get("Summary", ""),
            "references": [],
            "diagnostics": {"dim_mismatch": True, "q_dim": q_dim, "stored_dim": stored_dim}
        } for issue in issues]

    # compute similarity matrix
    sims = np.dot(q_embs, stored_embs.T)  # shape (M x N)

    results: List[Dict[str, Any]] = []
    for i, issue in enumerate(issues):
        input_key = _safe_key(issue, key_field)
        row = sims[i]
        diagnostics: Dict[str, Any] = {
            "row_size": int(row.size),
            "row_max": float(np.max(row)) if row.size else None,
            "row_min": float(np.min(row)) if row.size else None,
            "row_mean": float(np.mean(row)) if row.size else None,
            "n_above_threshold": int(np.sum(row >= score_threshold)) if row.size else 0,
            "top_candidates_before_filter": []
        }

        if row.size == 0:
            results.append({"input_key": input_key, "input_summary": issue.get("Summary", ""), "references": [], "diagnostics": diagnostics})
            continue

        top_idxs = np.argsort(-row)[: top_k * 2]
        # collect top candidates for debug (before threshold/self-skip)
        for idx in top_idxs[:min(len(top_idxs), 10)]:
            meta_entry = idx_to_meta.get(int(idx))
            diagnostics["top_candidates_before_filter"].append({
                "idx": int(idx),
                "score": float(row[int(idx)]),
                "meta_entry_sample": meta_entry if not debug else (meta_entry if isinstance(meta_entry, dict) else str(meta_entry))
            })

        refs = []
        for idx in top_idxs:
            meta_entry = idx_to_meta.get(int(idx))
            if not meta_entry:
                continue
            candidate_key = str(meta_entry.get("key") or meta_entry.get("Issue key") or meta_entry.get("Issue id") or "")
            # skip self-match
            if candidate_key and input_key and candidate_key == input_key:
                if debug:
                    # record self-skip in diagnostics
                    diagnostics.setdefault("self_skips", []).append({"idx": int(idx), "candidate_key": candidate_key})
                continue
            score = float(row[int(idx)])
            if score < score_threshold:
                # record threshold filter
                diagnostics.setdefault("threshold_filtered", []).append({"idx": int(idx), "score": score})
                continue
            refs.append({
                "idx": int(idx),
                "key": candidate_key,
                "issue_id": meta_entry.get("Issue id"),
                "summary": meta_entry.get("Summary"),
                "score": score
            })
            if len(refs) >= top_k:
                break

        if debug and not refs:
            best = float(np.max(row)) if row.size else 0.0
            print(f"[DEBUG] No refs >= {score_threshold:.2f} for '{input_key}'. Best score={best:.4f}")
            # also print the top few candidates
            print(" Top candidates (idx -> score):", [(int(idx), float(row[int(idx)])) for idx in top_idxs[:10]])

        diagnostics["returned_count"] = len(refs)   
        results.append({"input_key": input_key, "input_summary": issue.get("Summary", ""), "references": refs, "diagnostics": diagnostics})

        # Find Reference assignee
        # Make a method to find the reference assignee from the results
        # File Path : "out/issues_normalized.json",
        # Field Name : 
        # Task : Find Unique Assignees and One Time Load to the CSV and then 
        # Create a Queue , heapSort and then Assign the next in the liust to the issue 
        # finally results['potential_assignee'] retune that by appending in below  result 
        # results.append({"input_key": input_key, "input_summary": issue.get("Summary", ""), "references": refs, "diagnostics": diagnostics})
        ASSIGNEE_JSON = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/src/out/issues_normalized.json"
        ASSIGNEE_CSV  = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/src/out/assignees.csv"
        potential_assignee=assign_lru(ASSIGNEE_JSON,ASSIGNEE_CSV, verbose=True)    
        print("Before sending back :potential_assignee ", potential_assignee)
    return results,potential_assignee
