# 📄 **Technical Design Document — Semantic Spreadsheet Search Engine**

### Author: Gaurav Kungwani

### Version: 1.0

### Date: 27 November 2025

---

## **1. Overview**

This project implements a **semantic search engine for spreadsheets** that enables users to search using natural language instead of exact row labels or cell addresses. The system supports both **CSV and Excel (.xlsx)** files and extracts structured information from spreadsheets — including **formulas, semantic relationships, and business meaning** — to enable intelligent retrieval.

Users can query using natural language like:

* *"expense ratio for year 1"*
* *"which formulas calculate percentages?"*
* *"lookup formulas in this spreadsheet"*
* *"operating expenses divided by revenue"*

The engine retrieves the most relevant rows using a **hybrid pipeline** combining:

* Embedding-based semantic search (vector similarity with FAISS)
* Large Language Model (Gemini) query rewriting and re-ranking

This allows the system to understand **intent**, **business concepts**, and **formula semantics**, not just keywords.

---

## **2. Semantic Understanding Approach**

### 🔹 2.1 Text Representation of Spreadsheet Rows

Each row is converted into a natural-language representation:

```
Sheet: Financial Ratios. Row label: Expense Ratio. 
Year 1: 0.0588 (ratio of Operating Expenses for Year 1 divided by Revenue 
for Year 1 from sheet 3-Year Forecast) [percentage calculation]
```

This representation includes:

| Component             | Included? | Purpose                                             |
| --------------------- | --------- | --------------------------------------------------- |
| Sheet name            | ✅         | Context for user                                    |
| Headers + cell values | ✅         | Core searchable content                             |
| Row label             | ✅         | Semantic anchor ("Operating Margin")                |
| Formula meaning       | ✅         | Enables reasoning ("ratio", "percentage", "lookup") |
| Formula type tags     | ✅         | Enables functional queries                          |

---

### 🔹 2.2 Business Domain Intelligence

The system applies lightweight **domain-aware heuristics**, including:

* **Synonym expansion (via Gemini)**
  Example: `"Q1 revenue"` → `"first quarter revenue, Jan–Mar revenue, quarterly sales"`
* **Formula intent extraction**
  Example Classification:

| Formula Pattern                     | Semantic Meaning               |
| ----------------------------------- | ------------------------------ |
| `A/B` or `/`                        | percentage calculation / ratio |
| `AVERAGE()`                         | average formula                |
| `SUMIF`, `IF`, `COUNTIF`            | conditional calculation        |
| `VLOOKUP`, `XLOOKUP`, `INDEX/MATCH` | lookup formula                 |

These tags allow meaning-based queries like:

> "Show lookup formulas"

even if the formula uses `INDEX(MATCH())`.

---

## **3. Query Processing Pipeline**

```
User Query
      ↓
Gemini Query Expansion
      ↓
Embedding → FAISS Top-N Retrieval
      ↓
LLM Re-Ranking (Top-K)
      ↓
Final Ranked Results
```

### Explanation:

| Stage                        | Purpose                                                                |
| ---------------------------- | ---------------------------------------------------------------------- |
| **Query Expansion (LLM)**    | Adds synonyms, time semantics (“Q1” = “Jan–Mar”), and business context |
| **Embedding Search (FAISS)** | Finds semantically similar rows based on vector similarity             |
| **LLM Re-ranking**           | Uses reasoning to reorder results based on intent                      |
| **Result Formatting**        | Structured JSON including sheet, row, extracted meaning                |

---

## **4. Architecture & Data Structures**

### High Level Architecture:

```
                      ┌───────────────────┐
Upload → Parse → Embed → FAISS Index → Persist (index + JSON)
                      └───────────────────┘
                                     ↓
                             /search request
                                     ↓
                    Expand Query → FAISS → LLM rerank → Response
```

### Stored Files:

| File             | Format       | Purpose                              |
| ---------------- | ------------ | ------------------------------------ |
| `index.faiss`    | Binary       | Vector similarity index              |
| `doc_store.json` | JSON         | Row metadata + natural language text |
| User input file  | `.csv/.xlsx` | Source content                       |

---

## **5. Performance Considerations**

| Concern            | Design Choice                                                  |
| ------------------ | -------------------------------------------------------------- |
| Low Latency Search | FAISS Approximate Nearest Neighbor search                      |
| Large spreadsheets | Only one sheet indexed at a time (user resets before reupload) |
| Memory footprint   | Index stores sentence embeddings, not raw text                 |
| LLM cost           | Re-ranking done only on top-N (~20), not full dataset          |
| Persistence        | Index and metadata cached to disk to avoid reprocessing        |

---

## **6. Trade-offs**

| Area                                 | Choice              | Reason                                                |
| ------------------------------------ | ------------------- | ----------------------------------------------------- |
| Using embeddings vs full-text search | Embeddings          | Needed for semantic similarity                        |
| LLM-assisted re-ranking              | Yes                 | Higher relevance accuracy                             |
| Limited formula interpretation       | Rule-based matching | Simpler & deterministic; LLM later improves reasoning |
| One spreadsheet per session          | Simpler state model | Avoids multi-spreadsheet cross-linking complexity     |

---

## **7. Challenges & Solutions**

| Challenge                                  | Solution                                                       |
| ------------------------------------------ | -------------------------------------------------------------- |
| Excel formulas referencing other sheets    | Parsed using openpyxl and mapped back to headers + sheet names |
| JSON serialization errors (`NaN`)          | Sanitized NaN → `None` before saving                           |
| Query/row semantic mismatch                | Gemini query rewriting + LLM re-ranking                        |
| Ambiguous queries ("most profitable year") | LLM re-ranking contextual understanding                        |

---

## **8. Future Enhancements**

* Multi-file indexing with incremental updates
* Fine-tuned LLM for spreadsheet reasoning

---

# ✔ Summary

This system blends:

* **Classic Information Retrieval (FAISS vector search)**
* **LLM-based reasoning**
* **Spreadsheet-specific formula analysis**

to create a search experience that feels more like asking a knowledgeable analyst rather than searching raw data.
