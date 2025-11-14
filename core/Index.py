# put this into your script alongside your other helpers/imports
import os
import json
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import warnings
ignore_warnings = True

# defaults (can be overridden by cfg)
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIM = 384
_DEFAULT_IDX_PATH = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/src/embedding/annoy_index.ann"
_DEFAULT_META_PATH = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/src/embedding/annoy_meta.json"
_DEFAULT_EMB_PATH = "/Users/swatisingh/Documents/Rohit/GIT/AgenticJIraAssignment/src/embedding/embeddings.npy" 
_DEFAULT_N_TREES = 50
_DEFAULT_KEY_FIELD = "Issue key"

def _l2_normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return a / norms

def _load_meta_embs(meta_path: str, emb_path: str, dim: int):
    if os.path.exists(meta_path) and os.path.exists(emb_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        embs = np.load(emb_path)
        if embs.ndim != 2 or embs.shape[1] != dim:
            raise ValueError("Saved embeddings have wrong shape/dim")
        return meta, embs
    return {}, np.zeros((0, dim), dtype=np.float32)

def _save_meta_embs(meta: dict, embs: np.ndarray, meta_path: str, emb_path: str):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    np.save(emb_path, embs)

def _build_and_save_annoy(embs: np.ndarray, idx_path: str, dim: int, n_trees: int):
    # rebuild index from scratch and save to disk
    if embs.shape[0] == 0:
        if os.path.exists(idx_path):
            try:
                os.remove(idx_path)
            except Exception:
                pass
        return
    idx = AnnoyIndex(dim, metric='angular')
    for i in range(embs.shape[0]):
        idx.add_item(i, embs[i].tolist())
    idx.build(n_trees)
    idx.save(idx_path)

def add_index_new_Data(issues: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Incrementally index new/changed issues.
    - First-time run (no meta/emb) -> build index from all `issues`.
    - Subsequent runs -> embed only new or changed issues and append them.
    Returns: {"status": str, "added": int, "index_size": int}
    """
    # read config overrides or use defaults
    model_path = cfg.get("ModelPath", _DEFAULT_MODEL)
    embed_dim = int(cfg.get("EmbedDim", _DEFAULT_DIM))
    idx_path = cfg.get("AnnoyIndexPath", _DEFAULT_IDX_PATH)
    meta_path = cfg.get("AnnoyMetaPath", _DEFAULT_META_PATH)
    emb_path = cfg.get("AnnoyEmbPath", _DEFAULT_EMB_PATH)
    n_trees = int(cfg.get("AnnoyNTrees", _DEFAULT_N_TREES))
    key_field = cfg.get("KeyField", _DEFAULT_KEY_FIELD)

    # load existing meta & embeddings (if any)
    meta, embs = _load_meta_embs(meta_path, emb_path, embed_dim)

    # quick mapping key->idx for existing metadata
    existing_key_to_idx = {v["key"]: int(k) for k, v in meta.items()} if meta else {}

    # prepare SentenceTransformer (will download first run)
    model = SentenceTransformer(model_path)

    # If no existing meta -> it's the first run -> build from all issues
    if not meta:
        texts = []
        metas = []
        for t in issues:
            key = str(t.get(key_field) or t.get("Issue id") or t.get("Summary") or "")
            if not key:
                continue
            texts.append(t.get("Summary", ""))
            metas.append({
                "key": key,
                "Issue id": t.get("Issue id"),
                "Issue key": t.get("Issue key"),
                "Summary": t.get("Summary"),
                "Assignee": t.get("Assignee")
            })

        if not texts:
            return {"status": "no_data", "added": 0, "index_size": 0}

        new_embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        new_embs = _l2_normalize(new_embs).astype("float32")

        embs = new_embs
        meta = {str(i): metas[i] for i in range(len(metas))}
        _save_meta_embs(meta, embs, meta_path, emb_path)
        _build_and_save_annoy(embs, idx_path, embed_dim, n_trees)
        return {"status": "built_from_all", "added": len(metas), "index_size": embs.shape[0]}

    # else: incremental run -> embed only new or changed items
    texts_to_embed = []
    metas_to_add = []
    for t in issues:
        key = str(t.get(key_field) or t.get("Issue id") or t.get("Summary") or "")
        if not key:
            continue
        summary = t.get("Summary", "")
        assignee = t.get("Assignee")
        if key not in existing_key_to_idx:
            # completely new ticket -> add
            texts_to_embed.append(summary)
            metas_to_add.append({"key": key, "Issue id": t.get("Issue id"), "Issue key": t.get("Issue key"),
                                 "Summary": summary, "Assignee": assignee})
        else:
            idx = existing_key_to_idx[key]
            stored = meta.get(str(idx))
            # if summary or assignee changed -> treat as changed and add new vector
            if not stored or stored.get("Summary") != summary or stored.get("Assignee") != assignee:
                texts_to_embed.append(summary)
                metas_to_add.append({"key": key, "Issue id": t.get("Issue id"), "Issue key": t.get("Issue key"),
                                     "Summary": summary, "Assignee": assignee})

    if not texts_to_embed:
        return {"status": "no_new_or_changed", "added": 0, "index_size": embs.shape[0]}

    # compute embeddings for only new/changed texts
    new_embs = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=False)
    new_embs = _l2_normalize(new_embs).astype("float32")

    # append embeddings and metadata
    if embs.size == 0:
        embs = new_embs
    else:
        embs = np.vstack([embs, new_embs])

    n_before = len(meta)
    for i, m in enumerate(metas_to_add):
        meta[str(n_before + i)] = m

    # persist and rebuild index
    _save_meta_embs(meta, embs, meta_path, emb_path)
    _build_and_save_annoy(embs, idx_path, embed_dim, n_trees)

    return {"status": "incremental_ok", "added": len(metas_to_add), "index_size": embs.shape[0]}
