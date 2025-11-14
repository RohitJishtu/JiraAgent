import os
import json


# Task 1 : 

# Crreate a function Find Recommended Action from the issue JSON Key 
# The Recommended Action for now will be taken directly from the comments field of the issue
# But later based on the chat completion Model it ll import json

def find_recommended_actions(issue_key: str, json_path: str) -> str:
    """
    Find the 'Custom field (Comments)' for a given issue key in a JSON file.
    Returns the comment text if found, else None.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and dict formats
    issues = data.get("issues", data) if isinstance(data, dict) else data

    for issue in issues:
        key = issue.get("Issue key") or issue.get("key") or issue.get("issue_key")
        if key and str(key).strip() == issue_key:
            return issue.get("Custom field (Comments)")

    return None
# reframe the comments fields and give proper actions based on this field Custom field (Comments)







