import os
import json
import csv
from collections import OrderedDict
from typing import Dict, Any, Optional

def _extract_assignee(issue: Dict[str, Any]) -> Optional[str]:
    for k in ("assignee", "Assignee", "assigneeName", "owner", "assigned_to"):
        v = issue.get(k) or (issue.get("fields") or {}).get(k)
        if isinstance(v, dict):
            for sub in ("displayName", "name", "emailAddress", "accountId"):
                if v.get(sub):
                    return str(v[sub]).strip()
        elif v:
            s = str(v).strip()
            if s.lower() not in ("", "none", "null", "unassigned", "n/a"):
                return s
    return None

def _load_assignees_from_json(json_path: str) -> OrderedDict:
    lru = OrderedDict()
    if not os.path.exists(json_path):
        return lru

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # try line-delimited JSON
            f.seek(0)
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    data.append(parsed)
                except Exception:
                    continue

    # support dict with 'issues' or a top-level list
    issues = data.get("issues", data) if isinstance(data, dict) else data
    if not isinstance(issues, list):
        # If it's a dict mapping ids->issue dicts, try values()
        if isinstance(issues, dict):
            issues = list(issues.values())
        else:
            return lru

    for it in issues:
        if not isinstance(it, dict):
            continue
        a = _extract_assignee(it)
        if a and a not in lru:
            lru[a] = None
    return lru

def assign_lru(ASSIGNEE_JSON: str, ASSIGNEE_CSV: str, verbose: bool = False) -> Optional[str]:
    """
    Returns next least recently used assignee, creates CSV if missing.
    If verbose=True prints helpful debug messages.
    """
    lru = OrderedDict()
    if os.path.exists(ASSIGNEE_CSV):
        try:
            with open(ASSIGNEE_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if not r:
                        continue
                    a = (r.get("assignee") or "").strip()
                    if a and a.lower() not in ("", "none", "null", "unassigned", "n/a"):
                        if a not in lru:
                            lru[a] = None
        except Exception as e:
            if verbose:
                print("Warning: failed reading CSV:", e)
            lru = OrderedDict()

        if not lru and os.path.exists(ASSIGNEE_JSON):
            if verbose:
                print("CSV present but empty or invalid — attempting to rebuild from JSON.")
            lru = _load_assignees_from_json(ASSIGNEE_JSON)
    else:
        if verbose:
            print("CSV missing — building from JSON.")
        lru = _load_assignees_from_json(ASSIGNEE_JSON)

    if not lru:
        if verbose:
            print("No assignees found in CSV or JSON.")
        return None

    # Pick least recently used (first key)
    assignee = next(iter(lru))
    lru.move_to_end(assignee)

    # Persist updated order (safe dir handling)
    dirn = os.path.dirname(ASSIGNEE_CSV) or "."
    os.makedirs(dirn, exist_ok=True)
    with open(ASSIGNEE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["assignee"])
        for a in lru.keys():
            writer.writerow([a])

    if verbose:
        print("Returning assignee:", assignee)
    return assignee




if __name__ == "__main__":



#     # Get next LRU, create CSV if needed and print debug
#     pa = assign_lru(ASSIGNEE_JSON, ASSIGNEE_CSV, verbose=True)
#     print("Selected:", pa)

    # ASSIGNEE_JSON = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/src/data/out/issues_normalized.json"
    # ASSIGNEE_CSV = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/src/data/out/assignees.csv"

    ASSIGNEE_JSON = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/src/out/issues_normalized.json"
    ASSIGNEE_CSV  = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/src/out/assignees.csv"
    potential_assignee=assign_lru(ASSIGNEE_JSON,ASSIGNEE_CSV, verbose=True)    
    print("Before sending back :potential_assignee ", potential_assignee)