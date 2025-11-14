# **QuickRef — AI-Powered Jira Reference Finder**

**QuickRef** is an AI-driven Streamlit application that helps you instantly find similar Jira issues, generate reference tickets, and recommend next actions.
It uses **SentenceTransformer embeddings**, **Annoy** for fast similarity search, and **incremental training** to continuously improve as new Jira issues are added.

---

## 🚀 Features

* 🔍 **Semantic reference search** using embeddings (`all-MiniLM-L6-v2`)
* ⚡ **Annoy** for fast approximate nearest-neighbor lookups
* 🧠 **Incremental training** — newly submitted issues automatically update the JSON store & Annoy index
* 📥 **Single Issue input** or **CSV upload**
* 🗃️ **Training dataset** persisted in `issues_normalized.json`
* 🎯 **Top-K reference matches** with similarity scores
* 👤 **Potential assignee predictions**
* 📄 **Recommended actions** derived from past issues
* 📊 **Training Viewer** built into the UI
* 🧱 Fully configurable via `config.yml`

---

## 🧩 Architecture Overview

```
User Input (Single Issue / CSV)
            │
            ▼
        Streamlit UI
            │
            ▼
   Issue Normalization (build_issue_record)
            │
            ▼
  ┌─────────────────────────────┐
  │  Training Mode (optional)   │
  │   ├── append_to_json_store  │ → updates JSON dataset
  │   └── add_index_new_Data    │ → updates Annoy index
  └─────────────────────────────┘
            │
            ▼
Embedding Model (all-MiniLM-L6-v2)
            │
            ▼
   ANN Search via Annoy (top_k matches)
            │
            ▼
find_reference_issues()
    → matches, scores, assignee
            │
            ▼
find_recommended_actions()
            │
            ▼
       UI Output + CSV Download
```

---

## 📁 Project Structure

```
QuickRef/
├── src/
│   ├── main.py                   # Streamlit application
│   ├── core/
│   │   ├── ingest.py             # CSV ingest + JSON appender
│   │   ├── Index.py              # Annoy index management
│   │   ├── Reference_Issue.py    # Semantic reference finder
│   │   ├── Recommended_Actions.py# Recommended action generator
│   │   ├── training_view.py      # Viewer for training data
│   ├── utils/
│   │   └── utilis.py             # config loader/helpers
│   ├── embedding/                # embedding + Annoy index dirs
├── out/
│   ├── issues_normalized.json     # training data
│   └── staging/                   # saved input CSVs
├── config.yml
└── README.md
```

---

## ⚙️ How It Works (Step-by-Step)

### **1. Input Selection**

Choose between:

* **Single Issue** (manual fields)
* **CSV Upload**

### **2. Normalize Issue**

Uses `build_issue_record()` to create a standardized issue dictionary.

### **3. Save to staging**

Every run stores user inputs in `out/staging/` with timestamp.

### **4. Training Mode (optional)**

When `TrainingModel: true` in `config.yml`:

* `append_to_json_store()` → adds issue to dataset
* `add_index_new_Data()` → embeds + updates Annoy index

### **5. Find Reference Issues**

Core logic via:

```
refs, potential_assignee = find_reference_issues(...)
```

This performs:

* Embedding via all-MiniLM-L6-v2
* Annoy search
* Similarity scoring
* Threshold filtering
* Potential assignee calculation

### **6. Show Results**

* Sorted table of matches
* CSV export

### **7. Recommended Actions**

`find_recommended_actions()` looks up past actions from the JSON dataset.

### **8. Training Viewer**

`show_training_viewer()` displays the entire training store in a searchable table.

---

## 🧪 Configuration (`config.yml`)

```yaml
ModelPath: "all-MiniLM-L6-v2"
EmbeddingFolder: "src/embedding"
TrainingModel: true

Annoy:
  num_trees: 50
  index_file: "src/embedding/index.ann"

Similarity:
  threshold: 0.55
  top_k: 5
```

---

## ▶️ Run the App

### **Install dependencies**

```
pip install -r requirements.txt
```

### **Launch Streamlit**

```
streamlit run src/main.py
```

---

## 🧠 Tech Stack

* Streamlit
* SentenceTransformers (`all-MiniLM-L6-v2`)
* Annoy
* Pandas
* Pydantic v2
* YAML

---

## 👤 Author

**Rohit Jishtu**
GitHub: [@RohitJishtu](https://github.com/RohitJishtu)
