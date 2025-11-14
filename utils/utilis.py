# move Out 
import yaml
import os
import warnings
ignore_warnings = True
from datetime import datetime
from typing import List, Dict, Any

CONFIG_PATH = {}
def load_config(path: str = CONFIG_PATH) -> dict:
    """Read YAML config file and return as dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
    

# helper to build normalized issue record
# move Out 

def build_issue_record(issue_id: str, issue_key: str, summary: str, assignee: str = "", reporter: str = "", priority: str = "Medium") -> Dict[str, Any]:
    return {
        "Issue Type": "Story",
        "Issue key": issue_key,
        "Issue id": issue_id,
        "Summary": summary,
        "Assignee": assignee,
        "Reporter": reporter,
        "Priority": priority,
        "Status": "To Do",
        "Resolution": "",
        "Created": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "Affects Version/s": "",
        "Due Date": "",
        "Updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "Sprint": "",
        "Sprint (Extra)": "",
        "Custom field (Comments)": "",
        "Custom field (Parent Link)": ""
    }