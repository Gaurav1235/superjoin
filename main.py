from typing import List, Dict, Any
import os
import uuid

import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json

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
    results: List[SearchResult]


# --- Helper functions ---

def create_documents_from_excel(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse the Excel file into a list of 'documents'.
    Each document corresponds to one row (across all sheets).
    """
    xls = pd.ExcelFile(file_path)
    docs: List[Dict[str, Any]] = []

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)

        # ensure columns are strings
        df.columns = [str(c) for c in df.columns]

        for idx, row in df.iterrows():
            row_dict = row.to_dict()

            # Build a text representation for semantic search
            # "Sheet: Customers. Row: 5. Customer: ACME, Country: Germany, Status: Churned, MRR: 10000."
            parts = [f"Sheet: {sheet_name}", f"Row: {idx}"]
            for col_name, value in row_dict.items():
                if pd.isna(value):
                    continue
                parts.append(f"{col_name}: {value}")

            text = ". ".join(parts)

            docs.append(
                {
                    "id": f"{sheet_name}_{idx}",
                    "sheet": sheet_name,
                    "row_index": int(idx),
                    "row_data": row_dict,
                    "text": text,
                }
            )

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
            row_dict = row.to_dict()
            parts = [f"Sheet: {file.filename}", f"Row: {idx}"]
            for col_name, value in row_dict.items():
                if pd.isna(value):
                    continue
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
async def search(query: str = Query(..., description="Natural language query"),
                 k: int = Query(5, ge=1, le=50)):
    """
    Search the current index using a natural language query.
    """
    global faiss_index, doc_store

    if faiss_index is None or not doc_store:
        raise HTTPException(status_code=400, detail="No index available. Upload a spreadsheet first.")

    # 1) Rewrite / expand query using Gemini
    expanded_query = expand_query(query)

    # 2) Embed the expanded query
    query_vec = embed_texts([expanded_query])

    # 3) Search FAISS
    scores, indices = faiss_index.search(query_vec, k)
    scores = scores[0]
    indices = indices[0]

    results: List[SearchResult] = []

    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(doc_store):
            continue
        doc = doc_store[idx]
        results.append(
            SearchResult(
                score=float(score),
                sheet=doc["sheet"],
                row_index=doc["row_index"],
                row_data=doc["row_data"],
                text=doc["text"],
            )
        )

    return SearchResponse(
        query=query,
        expanded_query=expanded_query,
        results=results,
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
