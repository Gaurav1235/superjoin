from typing import List, Dict, Any
import os
import uuid
import re
import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
import math
import openpyxl
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv() 

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
DOCS_PATH = os.path.join(DATA_DIR, "doc_store.json")

app = FastAPI(title="Semantic Spreadsheet Search")

# --- CORS (optional, but useful if you build a frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global state (MVP, fine for assignment / demo) ---

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Gemini client for query rewriting
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

genai.configure(api_key=GEMINI_API_KEY)

# choose a lightweight model, e.g. gemini-1.5-flash
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

faiss_index = None           # type: faiss.IndexFlatIP | None
doc_store: List[Dict[str, Any]] = []   # parallel to the index vectors
index_dim: int | None = None


# --- Pydantic models for responses ---

class SearchResult(BaseModel):
    score: float
    sheet: str
    row_index: int
    row_data: Dict[str, Any]
    text: str


class SearchResponse(BaseModel):
    query: str
    expanded_query: str
    faiss_results: List[SearchResult]
    final_results: List[SearchResult]


# --- Helper functions ---

# ---------- Formula + cell semantics helpers ----------

CELL_REF_RE = re.compile(
    r"(?:'(?P<sheet>[^']+)'[.!])?(?P<col>\$?[A-Z]+)(?P<row>\$?\d+)"
)


def column_letter_to_index(col_letters: str) -> int:
    """
    'A' -> 1, 'B' -> 2, ..., 'AA' -> 27
    """
    result = 0
    for ch in col_letters:
        result = result * 26 + (ord(ch) - ord("A") + 1)
    return result  # 1-based


def parse_cell_ref(ref: str, current_sheet: str):
    """
    Parse things like:
      B5
      $B$5
      '3-Year Forecast'!B5
      $'3-Year Forecast'.$B$5   (Google Sheets export sometimes)
    into (sheet_name, row_index, col_index).

    Returns None if it can't parse.
    """
    m = CELL_REF_RE.search(ref.strip())
    if not m:
        return None

    sheet = m.group("sheet") or current_sheet
    col_letters = m.group("col").replace("$", "")
    row = int(m.group("row").replace("$", ""))

    col = column_letter_to_index(col_letters)
    return sheet, row, col


def build_cell_semantics(xls: pd.ExcelFile) -> dict:
    """
    Build a mapping:
        (sheet_name, excel_row, excel_col) -> "RowLabel for ColumnHeader"

    Assumes:
      - First row in the sheet is the header row (row 1 in Excel).
      - First column is a row label (like 'Revenue', 'Operating Expenses', etc.).
    """
    semantics: dict[tuple[str, int, int], str] = {}

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)
        if df.empty:
            continue

        df.columns = [str(c) for c in df.columns]

        # position 0 = row label column
        row_label_col = df.columns[0]

        for idx, row in df.iterrows():
            excel_row = idx + 2  # df row 0 -> Excel row 2

            row_label = row.get(row_label_col)
            if pd.isna(row_label):
                continue

            for col_pos, col_name in enumerate(df.columns):
                excel_col = col_pos + 1  # df col 0 -> Excel col 1 (A)

                # we skip the first column as a “data” cell, it's the label
                if col_pos == 0:
                    continue

                col_header = col_name
                desc = f"{row_label} for {col_header}"
                semantics[(sheet_name, excel_row, excel_col)] = desc

    return semantics

def classify_formula_type(formula: str) -> list[str]:
    """
    Return high-level tags describing what kind of formula this is.
    Used so semantic search can answer queries like:
      - "percentage calculations"
      - "average formulas"
      - "conditional calculations"
      - "lookup formulas"
    """
    tags: list[str] = []
    f = formula.upper()

    # --- Percentage calculations ---
    # Heuristics:
    # - ratio-style formulas (a/b)
    # - OR explicit *100 or /100
    if "/" in f or "*100" in f or "/100" in f:
        tags.append("percentage calculation")

    # --- Average formulas ---
    if "AVERAGE(" in f:
        tags.append("average formula")
    # SUM(...) / COUNT(...)
    if "SUM(" in f and "COUNT(" in f and "/" in f:
        if "average" not in " ".join(tags):
            tags.append("average formula")

    # --- Conditional calculations ---
    if "IF(" in f or "IFS(" in f or "SUMIF(" in f or "COUNTIF(" in f or "SUMIFS(" in f or "COUNTIFS(" in f:
        tags.append("conditional calculation")

    # --- Lookup formulas ---
    if "VLOOKUP(" in f or "HLOOKUP(" in f or "XLOOKUP(" in f:
        tags.append("lookup formula")
    # INDEX/MATCH combo
    if "INDEX(" in f and "MATCH(" in f:
        if "lookup formula" not in " ".join(tags):
            tags.append("lookup formula")

    return tags


def describe_formula(formula: str, current_sheet: str, cell_semantics: dict) -> str | None:
    """
    For simple ratio formulas, generate a semantic explanation.
    Also append high-level tags like "percentage calculation",
    "average formula", "conditional calculation", "lookup formula"
    based on the formula text.
    """
    if not (isinstance(formula, str) and formula.startswith("=")):
        return None

    base_explanation = None

    # ----- Semantic ratio explanation (your previous behavior) -----
    expr = formula[1:]
    if "/" in expr:
        try:
            left_raw, right_raw = expr.split("/", 1)
        except ValueError:
            left_raw, right_raw = None, None

        if left_raw and right_raw:
            left_ref = parse_cell_ref(left_raw.strip(), current_sheet)
            right_ref = parse_cell_ref(right_raw.strip(), current_sheet)

            if left_ref and right_ref:
                left_desc = cell_semantics.get(left_ref)
                right_desc = cell_semantics.get(right_ref)

                if left_desc and right_desc:
                    sheet_left = left_ref[0]
                    sheet_right = right_ref[0]
                    if sheet_left == sheet_right:
                        sheet_phrase = f"from sheet {sheet_left}"
                    else:
                        sheet_phrase = f"from sheets {sheet_left} and {sheet_right}"
                    base_explanation = (
                        f"ratio of {left_desc} divided by {right_desc} {sheet_phrase}"
                    )

    # ----- Functional tags (percentage / average / conditional / lookup) -----
    tags = classify_formula_type(formula)
    tag_text = ""
    if tags:
        # e.g. "[percentage calculation, conditional calculation]"
        tag_text = " [" + ", ".join(tags) + "]"

    # If we had a semantic explanation, append tags to it.
    if base_explanation:
        return base_explanation + tag_text

    # If we had no semantic interpretation (not a simple ratio),
    # but still have tags, just return the tags.
    if tags:
        return "Formula type:" + tag_text

    # If no interpretation at all, return None so caller can fall back.
    return None

def detect_header_rows_with_llm(df_raw: pd.DataFrame, sheet_name: str) -> list[int]:
    """
    Use Gemini to detect which row indices (0-based) are header rows in this sheet.

    Strategy:
    - Build a compact preview of the sheet (row index + cell strings).
    - Ask LLM to return JSON: {"header_rows": [int, int, ...]}.
    - Fallback to [] on failure (caller can then use heuristics).
    """
    # Build a preview to avoid sending huge sheets
    max_rows = min(60, df_raw.shape[0])
    max_cols = min(20, df_raw.shape[1])

    preview_rows = []
    for idx in range(max_rows):
        row_vals = df_raw.iloc[idx, :max_cols].tolist()
        # convert to strings, keep empties as ""
        cells = ["" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v) for v in row_vals]
        # skip completely empty preview rows to save tokens
        if all(c == "" for c in cells):
            continue
        preview_rows.append({
            "row_index": int(idx),   # 0-based pandas index (Excel row = idx+1)
            "cells": cells,
        })

    if not preview_rows:
        return []

    preview_json = json.dumps(preview_rows, ensure_ascii=False)

    prompt = f"""
You are analyzing an Excel sheet named '{sheet_name}'.

You are given a list of rows from the sheet. Each row has:
- row_index: 0-based index (0 means Excel row 1, 1 means Excel row 2, etc.)
- cells: list of cell values in that row as strings.

Your task:
1. Decide which rows are HEADER rows for tables in this sheet.
2. A header row usually:
   - Contains column names (mostly text, not just numbers).
   - Is followed by rows that look like data (numbers, dates, repeated patterns).
3. There may be MULTIPLE tables in a single sheet.
4. Return ONLY JSON with this schema:

{{
  "header_rows": [<int>, <int>, ...]
}}

Rules:
- Use 0-based indices (same as row_index in the input).
- Do NOT include any explanation text outside the JSON.
- If no header rows found, return {{"header_rows": []}}.

Here are the rows:

{preview_json}
"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()

        # Extract JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in LLM output")

        json_str = raw[start : end + 1]
        obj = json.loads(json_str)
        hdrs = obj.get("header_rows", [])
        if not isinstance(hdrs, list):
            raise ValueError("header_rows is not a list")
        # ensure ints and sort
        header_rows = sorted(int(h) for h in hdrs)
        return header_rows
    except Exception as e:
        print(f"[LLM header detection] failed for sheet '{sheet_name}', falling back. Error:", e)
        return []  # caller falls back to heuristic

def create_documents_from_excel(file_path: str) -> list[dict]:
    xls = pd.ExcelFile(file_path)
    wb = openpyxl.load_workbook(file_path, data_only=False)

    docs: list[dict] = []

    for sheet_name in xls.sheet_names:
        ws = wb[sheet_name]

        # 1) Read raw sheet WITHOUT header
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        if df_raw.empty:
            continue

        # 2) Ask LLM which rows are headers (0-based indices)
        header_rows = detect_header_rows_with_llm(df_raw, sheet_name)

        # Fallback: if LLM found nothing, use first non-empty row as a single header
        if not header_rows:
            non_empty_rows = df_raw.notna().any(axis=1)
            if not non_empty_rows.any():
                continue  # sheet is empty
            header_rows = [int(non_empty_rows.idxmax())]

        header_rows = sorted(header_rows)

        # 3) For each header row, treat everything below it until the next header row as that table's data
        max_row_idx = df_raw.index.max()
        table_id = 0

        for i, header_idx in enumerate(header_rows):
            # next header (exclusive end); if this is last, then go to end of sheet
            if i + 1 < len(header_rows):
                next_header_idx = header_rows[i + 1]
            else:
                next_header_idx = max_row_idx + 1  # until end

            # slice this vertical segment
            # NOTE: we include header_idx itself here and then split header/data
            segment = df_raw.loc[header_idx: next_header_idx - 1]

            if segment.empty:
                continue

            # header row (first row of the segment)
                        # header row (first row of the segment)
            header_row = segment.iloc[0]

            # 🔥 keep any column that has data somewhere in this segment
            # (not just in the header row)
            nonempty_cols = segment.notna().any(axis=0)
            segment = segment.loc[:, nonempty_cols]

            # 🔥 KEEP ORIGINAL EXCEL COLUMN INDICES
            # these are 0-based: 0 -> col A, 1 -> col B, ...
            orig_excel_cols = list(segment.columns)

            # recompute header & data after column filtering
            header_row = segment.iloc[0]
            data_df = segment.iloc[1:]

            # we allow blank rows inside the table but skip them when creating docs
            # (blank rows do NOT break the table)
            if data_df.empty:
                table_id += 1
                continue

            # 🔧 build column names:
            # - if header cell is non-empty -> use it
            # - if header cell is empty but data is numeric -> call it "Value"
            # - otherwise -> generic "Column {i}"
            col_names = []
            for j, col_idx in enumerate(orig_excel_cols):
                header_val = header_row.iloc[j]
                if header_val is not None and not (
                    isinstance(header_val, float) and math.isnan(header_val)
                ):
                    col_names.append(str(header_val))
                else:
                    col_data = data_df.iloc[:, j]
                    if pd.api.types.is_numeric_dtype(col_data):
                        col_names.append("Value")
                    else:
                        col_names.append(f"Column {j + 1}")

            data_df.columns = col_names


            # 4) Iterate over rows in this table block
            for idx, row in data_df.iterrows():
                # idx is original pandas index -> Excel row = idx + 1
                excel_row = int(idx) + 1

                # skip fully blank rows (optional)
                if row.isna().all():
                    continue

                # sanitize NaNs in row_data
                row_raw = row.to_dict()
                row_dict: dict = {}
                for col_name, value in row_raw.items():
                    if isinstance(value, float):
                        try:
                            if math.isnan(value):
                                value = None
                        except Exception:
                            pass
                    row_dict[col_name] = value

                parts = [
                    f"Sheet: {sheet_name}",
                    f"Table: {table_id}",
                    f"Row: {excel_row}",
                ]

                # optional: row label from first column
                first_col = data_df.columns[0]
                row_label = row_dict.get(first_col)
                if row_label is not None:
                    parts.append(f"Row label: {row_label}")

                # iterate columns and optionally read formulas
                for (col_name, value), orig_col_idx in zip(row_dict.items(), orig_excel_cols):
                    if value is None:
                        continue

                    # Since df_raw/header mirror the worksheet structure,
                    # we map col_pos -> Excel column index.
                    # orig_col_idx is 0-based; Excel columns are 1-based
                    excel_col = int(orig_col_idx) + 1

                    cell = ws.cell(row=excel_row, column=excel_col)
                    formula = (
                        cell.value
                        if isinstance(cell.value, str) and cell.value.startswith("=")
                        else None
                    )

                    explanation = None
                    if formula is not None:
                        # NOTE: you can pass a real cell_semantics map here if you built one
                        explanation = describe_formula(formula, sheet_name, cell_semantics={})

                    if explanation:
                        parts.append(f"{col_name}: {value} ({explanation})")
                    else:
                        parts.append(f"{col_name}: {value}")

                text = ". ".join(parts)

                docs.append(
                    {
                        "id": f"{sheet_name}_t{table_id}_r{excel_row}",
                        "sheet": sheet_name,
                        "table": table_id,
                        "row_index": excel_row,
                        "row_data": row_dict,
                        "text": text,
                    }
                )

            table_id += 1  # next table for the next header

    return docs


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index using inner product.
    We will L2-normalize vectors so this approximates cosine similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Compute L2-normalized embeddings for a list of texts.
    """
    vectors = model.encode(texts, convert_to_numpy=True)
    # L2 normalize to make inner product == cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    vectors = vectors / norms
    return vectors

def save_state():
    """
    Save FAISS index and doc_store to disk.
    """
    global faiss_index, doc_store

    os.makedirs(DATA_DIR, exist_ok=True)

    if faiss_index is not None:
        faiss.write_index(faiss_index, INDEX_PATH)

    # doc_store can contain non-JSON-native types (like numpy types, timestamps, etc.)
    # default=str will convert them to strings.
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_store, f, ensure_ascii=False, default=str)


def load_state():
    """
    Load FAISS index and doc_store from disk if they exist.
    """
    global faiss_index, doc_store, index_dim

    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        try:
            faiss_index_local = faiss.read_index(INDEX_PATH)
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                doc_store_local = json.load(f)

            faiss_index = faiss_index_local
            doc_store = doc_store_local
            index_dim = faiss_index.d
            print(f"Loaded existing index with {len(doc_store)} docs, dim={index_dim}")
        except Exception as e:
            print("Failed to load existing index/doc_store:", e)
            faiss_index = None
            doc_store = []
            index_dim = None
    else:
        print("No existing index/doc_store found.")

load_state()

def llm_rerank(
    query: str,
    expanded_query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Re-rank FAISS candidates using Gemini.

    `candidates` is a list of dicts like:
      { "idx": <int>, "text": <str>, "sheet": <str>, "row_index": <int>, ... }

    Returns the same dicts, but re-ordered, truncated to top_k.
    """
    if not candidates:
        return []

    # Build a compact list to feed the LLM (truncate text to keep prompt small)
    llm_items = []
    for i, cand in enumerate(candidates):
        llm_items.append(
            {
                "id": i,
                "sheet": cand.get("sheet"),
                "row_index": cand.get("row_index"),
                "text": cand.get("text", "")[:600],  # truncate
            }
        )

    items_json = json.dumps(llm_items, ensure_ascii=False)

    prompt = f"""
You are re-ranking spreadsheet rows for a semantic search engine.

The user asked:
- Original query: {query}
- Expanded semantic query: {expanded_query}

You are given a list of candidate rows from the spreadsheet.
Each item has:
- id: an integer
- sheet: sheet name
- row_index: row index in that sheet
- text: textual description of the row (including column names and values)

Your task:
1. Read all candidates carefully.
2. Rank them from MOST relevant to LEAST relevant for answering the user's query.
3. Return ONLY a JSON list of ids in best-first order, like:
   [3, 0, 2, 1]

Here are the candidates as JSON:

{items_json}
"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()

        # Try to extract JSON list from response
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("No JSON list found in LLM output")

        json_str = raw[start : end + 1]
        id_list = json.loads(json_str)

        if not isinstance(id_list, list):
            raise ValueError("Parsed JSON is not a list")

        # Map ids back to candidates, preserving only valid ones
        id_to_cand = {i: cand for i, cand in enumerate(candidates)}
        reranked = []
        for i in id_list:
            if i in id_to_cand:
                reranked.append(id_to_cand[i])

        # If model returned fewer than candidates, append the rest in original order
        seen_idxs = {c["idx"] for c in reranked}
        for cand in candidates:
            if cand["idx"] not in seen_idxs:
                reranked.append(cand)

        return reranked[:top_k]

    except Exception as e:
        print("LLM re-ranking failed, using FAISS order:", e)
        # fallback: just return original FAISS order, truncated
        return candidates[:top_k]

def expand_query(query: str) -> str:
    """
    Use Gemini to rewrite and expand the user's query with
    synonyms, related business terms, and equivalent phrases.
    If anything fails, return the original query.
    """
    prompt = (
        "You rewrite short search queries to make them easier "
        "to match in a semantic search engine over spreadsheets.\n\n"
        "Task: Expand the user's query by adding synonyms, equivalent business terms, "
        "and time period equivalents (e.g., revenue/sales/income, "
        "Q1/first quarter/Jan–Mar), but keep it within 1–2 sentences.\n\n"
        "Important: Do NOT answer the query; only rewrite it as an expanded query "
        "suitable for semantic search.\n\n"
        f"Original query: {query}"
    )

    try:
        response = gemini_model.generate_content(prompt)
        expanded = response.text.strip()
        print(f"[query expansion] '{query}' -> '{expanded}'")
        return expanded
    except Exception as e:
        print("Query expansion with Gemini failed, using original query:", e)
        return query

# --- API endpoints ---

@app.post("/upload")
async def upload_spreadsheet(file: UploadFile = File(...)):
    """
    Upload an Excel file, parse it, embed the rows, and build a FAISS index.
    This resets the current index & doc store (simplest for MVP).
    """
    global faiss_index, doc_store, index_dim

    # validate file
    if not (file.filename.endswith(".xlsx") or file.filename.endswith(".xls") or file.filename.endswith(".csv")):
        raise HTTPException(status_code=400, detail="Please upload an .xlsx, .xls, or .csv file")

    # save file temporarily
    os.makedirs("data", exist_ok=True)
    temp_id = str(uuid.uuid4())
    if file.filename.endswith(".csv"):
        temp_path = os.path.join("data", f"{temp_id}.csv")
    else:
        temp_path = os.path.join("data", f"{temp_id}.xlsx")

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # parse into documents
    if temp_path.endswith(".csv"):
        df = pd.read_csv(temp_path)
        df.columns = [str(c) for c in df.columns]

        docs: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            row_raw = row.to_dict()
            row_dict = {}
            parts = [f"Sheet: {file.filename}", f"Row: {idx}"]
            for col_name, value in row_raw.items():
                if pd.isna(value):
                    row_dict[col_name] = None
                else:
                    row_dict[col_name] = value
                parts.append(f"{col_name}: {value}")
            text = ". ".join(parts)
            docs.append(
                {
                    "id": f"{file.filename}_{idx}",
                    "sheet": file.filename,
                    "row_index": int(idx),
                    "row_data": row_dict,
                    "text": text,
                }
            )
    else:
        docs = create_documents_from_excel(temp_path)

    if not docs:
        raise HTTPException(status_code=400, detail="No rows found in spreadsheet")

    # embed documents
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)

    # build FAISS index
    faiss_index = build_faiss_index(embeddings)
    index_dim = embeddings.shape[1]
    doc_store = docs

    save_state()

    return {
        "message": "Index built successfully",
        "num_documents": len(doc_store),
        "embedding_dim": index_dim,
        "file_name": file.filename,
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="Natural language query"),
    k: int = Query(5, ge=1, le=50),
):
    global faiss_index, doc_store

    if faiss_index is None or not doc_store:
        raise HTTPException(
            status_code=400,
            detail="No index available. Upload a spreadsheet first.",
        )

    # 1️⃣ Expand query with Gemini
    expanded_query = expand_query(query)

    # 2️⃣ FAISS search - get larger pool for reranking
    faiss_k = max(k, 20)
    query_vec = embed_texts([expanded_query])
    scores, indices = faiss_index.search(query_vec, faiss_k)
    scores = scores[0]
    indices = indices[0]

    # 3️⃣ Convert FAISS retrieved rows into structured format
    faiss_results_list = []
    candidate_payload = []  # used by reranker

    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(doc_store):
            continue
        doc = doc_store[idx]

        item = SearchResult(
            score=float(score),
            sheet=doc["sheet"],
            row_index=doc["row_index"],
            row_data=doc["row_data"],
            text=doc["text"],
        )
        faiss_results_list.append(item)

        candidate_payload.append(
            {
                "idx": len(candidate_payload),
                "faiss_score": float(score),
                "sheet": doc["sheet"],
                "row_index": doc["row_index"],
                "row_data": doc["row_data"],
                "text": doc["text"],
            }
        )

    # 4️⃣ LLM ranking
    reranked = llm_rerank(
        query=query,
        expanded_query=expanded_query,
        candidates=candidate_payload,
        top_k=k,
    )

    # 5️⃣ Convert reranked items back to SearchResult
    final_results_list = []
    for cand in reranked:
        final_results_list.append(
            SearchResult(
                score=cand.get("faiss_score", 0.0),
                sheet=cand.get("sheet"),
                row_index=cand.get("row_index"),
                row_data=cand.get("row_data"),
                text=cand.get("text"),
            )
        )

    return SearchResponse(
        query=query,
        expanded_query=expanded_query,
        faiss_results=faiss_results_list[:k],   # show top-k FAISS before reranking
        final_results=final_results_list,       # after LLM reranking
    )

@app.post("/reset")
async def reset_index():
    global faiss_index, doc_store, index_dim

    # Clear in-memory state
    faiss_index = None
    doc_store = []
    index_dim = None

    # Delete files if they exist
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    if os.path.exists(DOCS_PATH):
        os.remove(DOCS_PATH)

    return {"message": "Index and document store cleared. You can upload a new file now."}


@app.get("/")
async def root():
    return {"message": "Semantic Spreadsheet Search API. Use /upload then /search."}
