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

    # embed query
    query_vec = embed_texts([query])
    # search
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

    return SearchResponse(query=query, results=results)


@app.get("/")
async def root():
    return {"message": "Semantic Spreadsheet Search API. Use /upload then /search."}
