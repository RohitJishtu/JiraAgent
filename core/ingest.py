from __future__ import annotations
import csv
import json
import os
import sys
from typing import List, Dict, Any, Optional
import yaml  # pip install pyyaml
import tempfile

REQUIRED = [
    "Issue Type",
    "Issue key",
    "Issue id",
    "Summary",
]

PLACEHOLDERS = {"", "None", "none", "NULL", "null", "########", "N/A", "NA", "-"}

def mandatory_populated(row: Dict[str, Any]) -> bool:
    for f in REQUIRED:
        if f not in row:
            return False
        v = row.get(f)
        if v is None:
            return False
        s = str(v).strip()
        if s in PLACEHOLDERS:
            return False
    return True

def load_config(cfg_path: str) -> Dict[str, Any]:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg

def _choose_issues_csv_path(cfg: Dict[str, Any]) -> str:
    """
    Logic:
      - If ScoringModel == true: prefer cfg.scoring_issues_csv or source.scoring_issues_csv
      - Else: prefer cfg.training_issues_csv or source.training_issues_csv
      - Fallback: source.csv_path
    Raises FileNotFoundError if chosen path is missing.
    """
    source = cfg.get("source", {}) or {}
    scoring_enabled = bool(cfg.get("ScoringModel", False))
    chosen_csv = None
    reason = "unknown"

    if scoring_enabled:
        chosen_csv = cfg.get("scoring_issues_csv") or source.get("scoring_issues_csv")
        reason = "ScoringModel=True -> using scoring_issues_csv"
    else:
        chosen_csv = cfg.get("training_issues_csv") or source.get("training_issues_csv")
        reason = "ScoringModel=False -> using training_issues_csv"

    if not chosen_csv:
        chosen_csv = source.get("csv_path")
        reason = f"fallback -> using source.csv_path"

    if not chosen_csv:
        raise AssertionError("No CSV path found in config (scoring/training/source.csv_path missing)")

    if not os.path.exists(chosen_csv):
        raise FileNotFoundError(f"Selected CSV not found: {chosen_csv} (reason: {reason})")

    return chosen_csv

def load_csv(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load issues from the CSV chosen according to config (scoring/training fallback logic).
    Returns list of normalized rows where mandatory fields are populated.
    """
    path = _choose_issues_csv_path(cfg)

    issues: List[Dict[str, Any]] = []
    skipped = 0
    loaded = 0

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        if not headers:
            raise RuntimeError("CSV appears empty or missing headers.")
        if not any(h in headers for h in REQUIRED):
            print("Warning: CSV headers do not contain expected required fields. Proceeding but results may be empty.")

        for i, raw in enumerate(reader, start=1):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            if not mandatory_populated(row):
                skipped += 1
                continue
            issues.append(row)
            loaded += 1

    print(f"CSV load complete: loaded={loaded}, skipped={skipped}, source={path}")
    return issues

def append_to_json_store(
    new_issues: List[Dict[str, Any]],
    out_path: str = "out/issues_normalized.json",
    key_field: str = "Issue key",
) -> Dict[str, int]:
    """
    Append new_issues to a JSON file at out_path, avoiding duplicates using key_field.
    Returns a dict summary: {"loaded": X, "skipped_duplicates": Y, "existing": Z}

    Behavior:
      - If out_path doesn't exist, it's created and all new_issues are written.
      - If out_path exists, existing items are loaded and considered for deduplication.
      - Deduplication is based only on the value of key_field (string equality).
      - Writes atomically using a temp file + rename.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Load existing items
    existing = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fh:
            try:
                existing = json.load(fh) or []
            except json.JSONDecodeError:
                # If file is corrupt/empty, treat as empty store
                existing = []

    existing_by_key = {}
    for item in existing:
        k = item.get(key_field)
        if k is not None:
            existing_by_key[str(k)] = item

    added = 0
    skipped = 0

    # Iterate new items and add only those not already in existing_by_key
    for it in new_issues:
        k = it.get(key_field)
        if k is None:
            # If no key, treat it as new (or you can choose to skip)
            # Here we append it but it cannot be deduped reliably.
            added += 1
            existing.append(it)
            continue

        ks = str(k)
        if ks in existing_by_key:
            skipped += 1
            continue
        existing_by_key[ks] = it
        existing.append(it)
        added += 1

    # Write atomically: write to temp file then rename
    dirn = os.path.dirname(out_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_", dir=dirn, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, out_path)  # atomic on most OSes
    finally:
        # If temp file still exists for some reason, remove it
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return {"loaded": added, "skipped_duplicates": skipped, "existing": len(existing)}

# ---------- team members extraction/writing ----------
def extract_team_members_from_issues(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return list of {"name": <assignee-string>, "count": <num assigned>} sorted by count desc.
    Unassigned entries become "<unassigned>".
    """
    counts: Dict[str, int] = {}
    for it in issues:
        a = (it.get("Assignee") or "").strip() or "<unassigned>"
        counts[a] = counts.get(a, 0) + 1
    members = [{"name": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    return members

def save_team_members_csv(members: List[Dict[str, Any]], csv_dir: str) -> str:
    """
    Save team_members.csv (and team_members.json) into csv_dir. Returns path to CSV.
    """
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "team_members.csv")
    json_path = os.path.join(csv_dir, "team_members.json")
    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "count"])
        writer.writeheader()
        for m in members:
            writer.writerow({"name": m.get("name"), "count": m.get("count")})
    # write JSON as well
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(members, fh, ensure_ascii=False, indent=2)
    return csv_path

# # ---------- example main usage ----------
# if __name__ == "__main__":
#     # usage: python this_script.py [optional path-to-config.yaml]
#     cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
#     try:
#         cfg = load_config(cfg_path)
#     except Exception as e:
#         print("Failed to load config:", e)
#         raise SystemExit(1)

#     try:
#         issues = load_csv(cfg)
#     except Exception as e:
#         print("CSV load failed:", e)
#         raise SystemExit(2)

#     # If requested, extract and save team members next to the chosen CSV
#     try:
#         if cfg.get("LoadTeamMembers"):
#             chosen_csv = _choose_issues_csv_path(cfg)
#             csv_dir = os.path.dirname(chosen_csv) or "."
#             members = extract_team_members_from_issues(issues)
#             tm_csv_path = save_team_members_csv(members, csv_dir)
#             print(f"Saved team members to: {tm_csv_path}")
#         else:
#             print("LoadTeamMembers not set; skipping team members extraction.")
#     except Exception as e:
#         print("Failed to extract/save team members:", e)

#     # Append loaded issues into the normalized JSON store (keeps same API as before)
#     try:
#         out_store = cfg.get("out_store") or "out/issues_normalized.json"
#         summary = append_to_json_store(issues, out_path=out_store, key_field="Issue key")
#         print("Append summary:", summary)
#     except Exception as e:
#         print("Append to json store failed:", e)
#         raise SystemExit(3)
