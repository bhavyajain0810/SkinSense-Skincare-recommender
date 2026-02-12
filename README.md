## SkinSense: Skincare Recommender (RAG + LLM + Vector DB)

SkinSense is a demo‑ready, end‑to‑end skincare recommender that uses:

- **Streamlit UI**
- **ChromaDB** (persistent vector database)
- **SentenceTransformer** embeddings (`all-MiniLM-L6-v2`)
- **RAG retrieval + LLM generation**
- **SQLite logging + Plotly dashboard**
- A separate **LLM API server** exposing an **OpenAI‑compatible** `POST /v1/chat/completions` endpoint
- **Docker + Docker Compose** to run the Streamlit app and LLM API together

All suggestions are **cosmetic‑only**, **non‑medical**, and based on a local rules knowledge base. The project is designed to be easy to understand and run for students.

---

### Architecture overview

ASCII diagram of the main components:

```text
                +-----------------------------+
                |  Streamlit App (app.py)     |
                |  - Recommender tab          |
                |  - Dashboard tab            |
                +---------------+-------------+
                                |
                                | (1) user inputs + optional image
                                v
                 +--------------+--------------+
                 |  RAG Layer (rag/*)          |
                 |  - query builder            |
                 |  - ChromaDB retrieval       |
                 +--------------+--------------+
                                |
                                | (2) top‑k rule cards
                                v
                 +--------------+--------------+
                 | Prompt builder (utils/*)    |
                 | - build_prompt(...)         |
                 +--------------+--------------+
                                |
                                | (3) OpenAI‑compatible request
                                v
                 +--------------+--------------+
                 |  LLM API Server (llm_api)   |
                 |  /v1/chat/completions       |
                 |   - mock backend (default)  |
                 |   - ollama backend          |
                 |   - openai backend          |
                 +--------------+--------------+
                                |
                                | (4) markdown answer + citations
                                v
                +---------------+--------------+
                |  Streamlit App               |
                |  - Show routine + rule IDs   |
                |  - Save to SQLite logs       |
                +---------------+--------------+
                                |
                                | (5) Plotly charts from logs
                                v
                +-----------------------------+
                |  Dashboard (skin types,     |
                |  concerns, feedback, table) |
                +-----------------------------+
```

---

### Project structure

```text
skinsense-rag-skincare-recommender/
  app.py
  requirements.txt
  README.md
  .gitignore
  .env.example
  .dockerignore
  Dockerfile
  docker-compose.yml
  scripts/
    bootstrap.py
  knowledge_base/
    __init__.py
    generate_rules.py
    rules.json
  rag/
    __init__.py
    build_index.py
    retrieve.py
  utils/
    __init__.py
    db.py
    llm_client.py
    prompt_templates.py
    vision_attributes.py
  llm_api/
    main.py
    requirements.txt
    .env.example
    Dockerfile
  logs/
    .keep
```

---

### Knowledge base & RAG

- `knowledge_base/generate_rules.py` creates **~120 short rule cards** into `rules.json`.
- Each card looks like:

```json
{"id": "R001", "tags": "skin_type:oily concern:acne routine:am", "text": "..."}
```

- The rules cover:
  - **Skin types**: `oily`, `dry`, `combination`, `sensitive`, `normal`
  - **Concerns**: `acne`, `pigmentation`, `dullness`, `dryness`, `redness`, `texture`, `fine_lines`, `sun_protection`
  - General **safety cards**: patch‑test, introduce one new product, avoid over‑exfoliating, non‑medical disclaimer.

Vector index:

- `rag/build_index.py`:
  - Uses `chromadb.PersistentClient` with path `./chroma_db`
  - Embeddings from `sentence-transformers` (`all-MiniLM-L6-v2`)
  - Collection name: **`skincare_rules`**
  - Upserts `id`, `document`, `metadata` for each rule.
- `rag/retrieve.py`:
  - `get_collection(...)` loads the persistent Chroma collection.
  - `retrieve_rules(collection, query, k=8)` returns:

    ```python
    [
      {"id": "...", "document": "...", "metadata": {...}, "distance": 0.123},
      ...
    ]
    ```

---

### Streamlit app (`app.py`)

- Two tabs: **Recommender** and **Dashboard**.

**Recommender tab**

- Optional **image uploader** (jpg/png).
  - If `VISION_ATTR_URL` is set, `utils/vision_attributes.detect_from_image` is called.
  - If the service returns `skin_type`, `concerns`, or `notes`, they are used as **defaults** only (user can still edit).
- Manual inputs:
  - `skin_type` select: `oily`, `dry`, `combination`, `sensitive`, `normal`
  - `concerns` multiselect (up to 3)
  - `notes` free text
  - Slider for **k** retrieved cards (5–15, default 8)
- On **Generate routine**:
  1. Build a query string from attributes via `make_query(attrs)`.
  2. Retrieve top‑k rules via `retrieve_rules(...)`.
  3. Build a RAG prompt via `build_prompt(attrs, retrieved_rules)` with strict instructions:
     - Use only rule cards
     - No medical claims
     - Output Markdown with sections:
       - `## AM Routine`
       - `## PM Routine`
       - `## Extra Tips`
       - `## Why these suggestions?`
       - `## Citations` (must include rule IDs).
  4. Call the LLM via `utils.llm_client.call_llm` (OpenAI‑compatible /v1/chat/completions).
  5. If the LLM is not configured or fails:
     - Use `fallback_answer(rule_ids)` for a deterministic local answer.
  6. Log interaction to SQLite via `utils.db.insert_interaction`.
  7. Display:
     - The markdown answer
     - **Retrieved rule IDs**: e.g. `R012, R044`
- Feedback buttons:
  - 👍 `Helpful`
  - 👎 `Not helpful`
  - They call `db.update_feedback` and update the interaction’s `feedback` column.

**Dashboard tab**

- Reads all interactions from `logs/interactions.db`.
- Uses **Plotly** for:
  - **Skin type frequency** (bar chart)
  - **Top concerns frequency** (flattened from comma‑separated list)
  - **Feedback ratio** (helpful vs not_helpful vs none)
- Shows a **table of the last 25 interactions**:
  - id, timestamp, skin_type, concerns, retrieved_rule_ids, feedback.

---

### SQLite logging (`utils/db.py`)

- Database file: `logs/interactions.db`
- Table: `interactions`:

```sql
id INTEGER PRIMARY KEY AUTOINCREMENT,
ts TEXT NOT NULL,               -- UTC ISO timestamp
attributes_json TEXT NOT NULL,  -- JSON of skin_type, concerns, notes
retrieved_rule_ids TEXT NOT NULL, -- comma-separated IDs
response_md TEXT NOT NULL,      -- Markdown response
feedback TEXT NULL              -- 'helpful', 'not_helpful', or NULL
```

Helpers:

- `init_db()`
- `insert_interaction(attributes, retrieved_rule_ids, response_md) -> id`
- `update_feedback(id, feedback)`
- `fetch_all_interactions()`
- `fetch_recent_interactions(limit=25)`

---

### Prompt engineering (`utils/prompt_templates.py`)

- `make_query(attrs) -> str`:
  - Converts `{skin_type, concerns, notes}` into a compact search query.
- `build_prompt(attrs, retrieved_rules) -> str`:
  - Assembles the full system + user prompt:
    - Strictly uses provided rule cards as the only source of truth.
    - Enforces **cosmetic‑only**, non‑medical language.
    - Enforces Markdown output with sections and **citations**:

      ```markdown
      ## AM Routine
      ...

      ## PM Routine
      ...

      ## Extra Tips
      ...

      ## Why these suggestions?
      ...

      ## Citations
      Used: R001, R044, ...
      ```

---

### LLM client (`utils/llm_client.py`)

- Talks to the LLM API server (or any OpenAI‑compatible endpoint).
- Reads:
  - `LLM_BASE_URL` – e.g. `http://localhost:8001` or `http://llm_api:8001` in Docker
  - `LLM_API_KEY` – dummy is fine for the mock backend
  - `LLM_MODEL` – logical model name, e.g. `skinsense-local`
- Sends:

```json
POST {LLM_BASE_URL}/v1/chat/completions
Authorization: Bearer {LLM_API_KEY}
{
  "model": "skinsense-local",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<RAG prompt>"}
  ],
  "temperature": 0.4
}
```

- Returns the assistant message content.
- `fallback_answer(rule_ids)` generates a **local, deterministic**, safe markdown routine (used when the LLM is not available).

---

### Optional vision attributes (`utils/vision_attributes.py`)

- If `VISION_ATTR_URL` is set:
  - Sends: `{"image_base64": "..."}`
  - Expects: `{"skin_type": "...", "concerns": [...], "notes": "..."}`.
- Normalizes concerns to a list of strings.
- Returns `None` on any error, so the app remains robust if no service is available.

---

### LLM API server (`llm_api/main.py`)

Implements a FastAPI server with:

- `GET /health`:
  - Returns `{"status": "ok", "backend": "<mock|ollama|openai>"}`.
- `POST /v1/chat/completions`:
  - Accepts an **OpenAI‑compatible** body:

    ```json
    {
      "model": "skinsense-local",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
      ],
      "temperature": 0.4
    }
    ```

  - Returns an OpenAI‑compatible response with `choices[0].message.content`.

Backends (selected by `LLM_BACKEND` in `llm_api/.env`):

- **`mock` (default)**:
  - Deterministic skincare routine.
  - Extracts rule IDs from the prompt using regex (`R\d{3}`) and includes them in `## Citations  Used: Rxxx, Ryyy`.
- **`ollama`**:
  - Uses `OLLAMA_BASE_URL` (default `http://localhost:11434`) and `OLLAMA_MODEL`.
  - Calls `{OLLAMA_BASE_URL}/api/chat` with `{"model": OLLAMA_MODEL, "messages": [...]}`.
  - Wraps the result into OpenAI‑style response.
- **`openai`**:
  - Uses `OPENAI_BASE_URL` (default `https://api.openai.com`) and `OPENAI_API_KEY`.
  - Proxies the request to `/v1/chat/completions` and returns the raw JSON.

---

### Environment variables

**Root `.env.example`** (for Streamlit app):

```env
LLM_BASE_URL=http://localhost:8001
LLM_API_KEY=dummy
LLM_MODEL=skinsense-local
#VISION_ATTR_URL=http://localhost:9000/analyze
```

**`llm_api/.env.example`**:

```env
LLM_BACKEND=mock
#OLLAMA_BASE_URL=http://host.docker.internal:11434
#OLLAMA_MODEL=llama3.1
#OPENAI_BASE_URL=https://api.openai.com
#OPENAI_API_KEY=sk-...
```

You should create **real `.env` files** by copying these examples (no secrets are committed).

---

### Local setup (no Docker)

Requirements:

- Python 3.10+ recommended
- `git` (optional but useful)

Steps (Windows PowerShell examples, similar for macOS/Linux):

```powershell
cd "d:\Skinsense skincare recommmender\skinsense-rag-skincare-recommmender"

# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install dependencies for the Streamlit app
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install dependencies for the LLM API server
pip install -r llm_api\requirements.txt

# 4. Create env files (safe, no secrets included)
copy .env.example .env
copy llm_api\.env.example llm_api\.env
```

Now generate the knowledge base and Chroma index, start the LLM API, and start the app:

```powershell
# Terminal 1 – generate rules and build index (optional, bootstrap does this too)
python knowledge_base\generate_rules.py
python rag\build_index.py

# Terminal 2 – start the LLM API (port 8001)
cd "d:\Skinsense skincare recommmender\skinsense-rag-skincare-recommmender"
uvicorn llm_api.main:app --port 8001 --reload

# Terminal 3 – start Streamlit app (port 8501)
cd "d:\Skinsense skincare recommmender\skinsense-rag-skincare-recommmender"
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

> The **mock backend** works out‑of‑the‑box with `LLM_BACKEND=mock` and `LLM_API_KEY=dummy`.  
> No Ollama or OpenAI keys are required to get a working demo.

---

### Running with Docker & Docker Compose

Prerequisites:

- Docker Desktop or Docker Engine + Docker Compose

Steps:

```bash
cd skinsense-rag-skincare-recommender

# 1. Prepare env files (optional but recommended)
cp .env.example .env
cp llm_api/.env.example llm_api/.env

# 2. Build and run both services
docker compose up --build
```

This does the following:

- Builds the **llm_api** image and runs it on port **8001**.
- Builds the **app** image and runs Streamlit on port **8501**.
- Mounts:
  - `./logs` → `/app/logs` (SQLite interactions)
  - `./chroma_db` → `/app/chroma_db` (Chroma persistent store)
- The app container runs:
  - `python scripts/bootstrap.py` (ensures rules + Chroma index)
  - `streamlit run app.py --server.address=0.0.0.0 --server.port=8501`

Visit `http://localhost:8501` in your browser.

---

### Verification checklist

You can verify the pipeline end‑to‑end with these commands (from the project root):

1. **Generate rules**

   ```bash
   python knowledge_base/generate_rules.py
   ```

2. **Build Chroma index**

   ```bash
   python rag/build_index.py
   ```

3. **Start LLM API server**

   ```bash
   uvicorn llm_api.main:app --port 8001
   ```

4. **Start Streamlit app**

   ```bash
   streamlit run app.py
   ```

5. **Run everything via Docker Compose**

   ```bash
   docker compose up --build
   ```

If all of these complete without errors and you can see routines + dashboards in the browser, the project is working correctly.

---

### Safety disclaimer

- All outputs in this project are **cosmetic and educational** only.
- They are **not** medical advice, diagnosis, or treatment.
- Users should **consult a qualified professional** for any medical or serious skin concerns.
- Do not use this project for clinical or health‑critical decisions.

---

### Troubleshooting

- **`ModuleNotFoundError` for a package**
  - Make sure your virtual environment is active and you ran:
    - `pip install -r requirements.txt`
    - `pip install -r llm_api/requirements.txt`

- **ChromaDB / embedding errors**
  - Ensure `knowledge_base/rules.json` exists (run `python knowledge_base/generate_rules.py`).
  - Rebuild the index: `python rag/build_index.py`.
  - In Docker, the app runs `scripts/bootstrap.py` automatically.

- **`Could not connect to the vector database` in Streamlit**
  - Run `python rag/build_index.py` once, or `python scripts/bootstrap.py`.
  - Check that `./chroma_db` exists and is writable.

- **LLM endpoint errors in the UI**
  - If LLM is not configured or fails, the app automatically falls back to a local template.
  - To debug:
    - Confirm `LLM_BASE_URL` and `LLM_API_KEY` in `.env`.
    - Make sure the LLM API server is running (`uvicorn llm_api.main:app --port 8001`).
    - Check `GET http://localhost:8001/health` for backend info.

- **Docker build takes long or fails**
  - Large model downloads (sentence‑transformers) may take a while the first time.
  - Ensure you have a stable internet connection the first time you build/run.

- **Dashboard is empty**
  - Generate a few routines so interactions are logged to SQLite.
  - Then refresh the dashboard tab.

If you run into anything else, you can inspect:

- `logs/interactions.db` – using any SQLite viewer
- `docker compose logs` – for container‑level troubleshooting

