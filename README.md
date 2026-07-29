# ⚡ Vidullanka Plant AI Assistant

A Streamlit + LangChain + Groq RAG chatbot that answers questions using the official
Standard Operating Procedures (SOPs) of the **Wembiyagoda (WMB)** and **Batathota (BTO)**
Mini Hydro Power Plants.

It reads the two SOP PDFs, splits and embeds them locally (no cost, no API key needed
for embeddings), stores them in a FAISS vector index, and uses a **Groq**-hosted LLM to
answer questions grounded only in that SOP content — with page-level source citations.

---

# 🚀 Live Demo

**Access the deployed application here:**

🔗 **https://vidullanka-ai-agent-ilsye8bk7azmusu39mbmxr.streamlit.app/**

---

## 1. Project structure

```
plant_assistant/
├── app.py               # Streamlit UI (run this)
├── rag_engine.py         # RAG pipeline: loading, chunking, embedding, retrieval, Groq calls
├── data/
│   ├── WMB_SOP.pdf       # Wembiyagoda SOP (already included)
│   └── BTO_SOP.pdf       # Batathota SOP (already included)
├── faiss_index/          # auto-created on first run (the built vector index)
├── requirements.txt
├── .env.example           # copy to .env and add your Groq key
└── README.md
```

## 2. Get a free Groq API key

1. Go to <https://console.groq.com/keys>
2. Sign up (free) and click **Create API Key**
3. Copy the key — it starts with `gsk_...`

You can provide the key in **any one** of these ways:
- Paste it into the sidebar text box when the app is running (simplest, per-session), **or**
- Copy `.env.example` to `.env` and paste the key in there, **or**
- Create `.streamlit/secrets.toml` with:
  ```toml
  GROQ_API_KEY = "gsk_your_real_key_here"
  ```

## 3. Install & run

It's strongly recommended to use a virtual environment.

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) add your API key to .env
cp .env.example .env
# then edit .env and paste your real key

# 4. Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

On the **first run**, it will take ~30-90 seconds to read and embed both PDFs (you'll see
a "🔧 Indexing SOP documents..." spinner). After that, the index is saved to
`faiss_index/` and future runs load instantly. Use the **"🔄 Rebuild index"** button in the
sidebar any time you replace/update the PDFs in `data/`.

## 4. Using the app

- **Sidebar** — enter/confirm your Groq API key, pick a model, choose whether to search
  both plants or just one, and adjust how many SOP passages are retrieved per question.
- **Quick questions** — one-click buttons for the most common queries (startup,
  emergency stop, electrical fire, water leakage).
- **Chat box** — ask anything in plain English, e.g.:
  - "What is the MIV opening procedure?"
  - "Who needs to approve a plant shutdown at Batathota?"
  - "What PPE is required for HVCB cleaning?"
  - "Compare the crane operation rules for both plants."
- **Sources** — every answer has a collapsible "📄 Sources" panel showing exactly which
  plant/page the answer was pulled from, so you can always verify against the original SOP.

## 5. Adding more plants / SOPs later

1. Drop the new PDF into `data/`.
2. Add an entry to the `PLANTS` dict at the top of `rag_engine.py`, e.g.:
   ```python
   "XYZ": {
       "file": "XYZ_SOP.pdf",
       "name": "XYZ Mini Hydro Power Plant",
       "location": "Somewhere",
       "emoji": "🌊",
   },
   ```
3. Click **"🔄 Rebuild index"** in the sidebar (or delete the `faiss_index/` folder and
   restart the app).

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Couldn't build/load the SOP index` | Make sure `WMB_SOP.pdf` / `BTO_SOP.pdf` are actually inside `data/`. |
| `401 Unauthorized` / auth errors from Groq | Double-check the API key has no extra spaces and starts with `gsk_`. |
| `model_not_found` / `decommissioned` errors | Groq periodically retires models. Open the **Model** dropdown in the sidebar and pick a different one (e.g. `openai/gpt-oss-120b`), or check <https://console.groq.com/docs/models> for current model IDs and update `AVAILABLE_MODELS` in `rag_engine.py`. |
| Answers seem generic / not from the SOP | Increase "How many SOP passages to retrieve" in the sidebar, or click "🔄 Rebuild index" if you recently changed the PDFs. |
| First run is slow | Normal — it's embedding ~70 pages locally on CPU. Subsequent runs reuse the saved `faiss_index/`. |

## 7. Notes on cost & privacy

- **Embeddings** run 100% locally via `sentence-transformers` — free, no data leaves your machine.
- **Only the retrieved SOP snippets + your question** are sent to Groq for the final answer
  (not the entire PDFs).
- Groq's free tier is generous for this use case; see <https://console.groq.com/docs/rate-limits>
  for current limits.
