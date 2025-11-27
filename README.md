# superjoin

---

## 1. Prerequisites

* **Python** ≥ 3.10 (I’ve been using 3.11)
* **Git**
* Internet access (to:

  * download Python packages
  * call **Gemini API**)

---

## 2. Clone / copy the project

```bash
git clone https://github.com/Gaurav1235/superjoin.git
cd superjoin
```

Project structure (roughly):

```text
superjoin/
├─ main.py              # FastAPI app
├─ requirements.txt
├─ run_evals.py         # simple eval runner
├─ evals.jsonl          # eval cases
└─ data/                # created at runtime (index.faiss, doc_store.json, temp uploads)
```

---

## 3. Create a virtual environment & install dependencies

From inside the project folder:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Your `requirements.txt` should include things like:

```txt
fastapi
uvicorn
pandas
openpyxl
faiss-cpu
sentence-transformers
python-multipart
google-generativeai
requests      # for run_evals.py
python-dotenv # optional, if you use .env
```

(Adjust as per your actual file.)

---

## 4. Configure environment variables (Gemini)

The app uses **Google Gemini** for:

* Query expansion
* LLM re-ranking

You need a Gemini API key and then set:

```bash
export GEMINI_API_KEY="your_real_key_here"
```

If you prefer a `.env` file, create `.env`: (i have used this)

```env
GEMINI_API_KEY=your_real_key_here
```

…and make sure you call `load_dotenv()` in `main.py` before you configure `google.generativeai`.

---

## 5. Run the FastAPI server

From the project root (inside the venv):

```bash
uvicorn main:app --reload
```

By default it will run at:

* API root: `http://127.0.0.1:8000/`
* Docs/Swagger: `http://127.0.0.1:8000/docs`

---

## 6. Using the API

### 6.1 Upload a spreadsheet

1. Open `http://127.0.0.1:8000/docs` in a browser.
2. Find the `POST /upload` endpoint.
3. Click **Try it out**.
4. Choose a file:

   * `.xlsx` Excel file (multi-sheet, formulas supported), or
   * `.csv` (no formulas, but still indexed).
5. Execute.

You should see a response like:

```json
{
  "message": "Index built successfully",
  "num_documents": 42,
  "embedding_dim": 384,
  "file_name": "Sales Dashboard.xlsx"
}
```

This will also create/update:

* `data/index.faiss`
* `data/doc_store.json`

---

### 6.2 Run a search

Still in `/docs`:

1. Use `GET /search`

2. Enter e.g.:

   * `expense ratio for year 1`
   * `show percentage calculations`
   * `lookup formulas`
   * `operating expenses divided by revenue`

3. Execute.

The response includes:

* `query` – original query
* `expanded_query` – Gemini-expanded query
* `results` – list of rows (sheet, row_index, row_data, text)

Example:

```json
{
  "query": "expense ratio for year 1",
  "expanded_query": "...",
  "results": [
    {
      "score": 0.92,
      "sheet": "Financial Ratios",
      "row_index": 5,
      "row_data": { ... },
      "text": "Sheet: Financial Ratios. Row label: Expense Ratio. Year 1: 0.0588 (ratio of Operating Expenses for Year 1 divided by Revenue for Year 1 from sheet 3-Year Forecast) [percentage calculation]"
    }
  ]
}
```

---

### 6.3 Resetting the index (optional)

If you implemented `POST /reset`:

* Call `POST /reset` to clear in-memory index and delete `index.faiss` / `doc_store.json`.
* Then upload a new file via `POST /upload`.

If not, simply **uploading a new spreadsheet** already replaces the previous one (the index is rebuilt).

---

## 7. Running the evals on another system

1. Make sure the server is running (`uvicorn main:app --reload`).
2. In another terminal (same venv, same project folder):

```bash
python run_evals.py
```

The script:

* Reads `evals.jsonl`
* Sends each `query` to `/search`
* Checks whether expected keywords appear in the top-1 result
* Prints an overall summary + per-category stats

Typical output:

```text
Overall: 17/20 passed

[formula_understanding] 5/5 passed
[business_concepts] 4/5 passed
[cross_sheet] 4/5 passed
[general_semantic] 4/5 passed

Detailed results:
----
Category: formula_understanding
Query   : expense ratio for year 1
Passed  : True
Expected: ['Expense Ratio', 'ratio', 'Operating Expenses for Year 1', 'Revenue for Year 1']
Top Text: Sheet: Financial Ratios. Row label: Expense Ratio. Year 1: 0.0588 ...
...
```

If they change the port or host, they just need to update `API_URL` at the top of `run_evals.py`.

---

## 8. Quick checklist for someone new

1. Install Python (3.10+).
2. Clone repo & `cd` into it.
3. Create venv & `pip install -r requirements.txt`.
4. Set `GEMINI_API_KEY`.
5. Run `uvicorn main:app --reload`.
6. Visit `http://127.0.0.1:8000/docs`.
7. `POST /upload` → upload Excel/CSV.
8. `GET /search` → run semantic queries.
9. (Optional) `python run_evals.py` → see eval scores.
