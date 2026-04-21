# ShakespeareChat — A Voice-Powered Chatbot That Speaketh as the Bard Himself

## The Players of This Noble Work

| Name | |
|---|---|
| Liggia Cruz | |
| Chao | |
| Emmanuel | |

---

## Overview

Hark! ShakespeareChat is a voice-powered chatbot forged for students of art history, wherein the spirit of William Shakespeare himself doth answer their questions. The student speaks aloud — asking about plays, sonnets, themes, historical context, or the literary artistry of the Bard — and receiveth an answer in the very voice and manner of Shakespeare.

The bot is a specialist of noble but narrow domain: it speaketh only of Shakespeare and his world — his plays, his sonnets, his life, the Elizabethan era, theatrical history, literary themes, and the art history surrounding his works. This restriction of domain is by design, for it ensureth answers most precise, immersive, and trustworthy for the student of art history.

---

## Pipeline — A Play in Four Acts

```
┌─────────────────────────────────────────────────────────┐
│                   THE ART HISTORY STUDENT               │
│        asketh a question about Shakespeare aloud        │
└───────────────────┬─────────────────────────────────────┘
                    │ an audio file (.ogg / .wav)
                    ▼
┌─────────────────────────────────────────────────────────┐
│  ACT I — STT  (stt/transcriber.py)                      │
│  Whisper doth hearken to the audio and returneth:       │
│  → text:     "What are the themes of Hamlet?"           │
│  → language: "en"                                       │
└───────────────────┬─────────────────────────────────────┘
                    │ text + language
                    ▼
┌─────────────────────────────────────────────────────────┐
│  ACT II — RAG  (rag/retriever.py)                       │
│  Searcheth through the Shakespeare scrolls and          │
│  art history texts for the most pertinent passages      │
│  → chunk 1: "To be, or not to be, that is the…"        │
│  → chunk 2: "Hamlet explores revenge, mortality…"       │
│  → chunk 3: "The Elizabethan theatre was shaped by…"    │
└───────────────────┬─────────────────────────────────────┘
                    │ text + language + chunks
                    ▼
┌─────────────────────────────────────────────────────────┐
│  ACT III — LLM  (llm/agent.py)                          │
│  Claude receiveth the question and context, and         │
│  composeth an answer in the very voice of Shakespeare   │
│  → "Ah, thou dost ask of Hamlet! In that great work…"  │
└───────────────────┬─────────────────────────────────────┘
                    │ answer as text
                    ▼
┌─────────────────────────────────────────────────────────┐
│  ACT IV — TTS  (tts/synthesizer.py)                     │
│  edge-tts doth transform the text into audio            │
│  with a voice most fitting for the Bard's own words     │
│  → output.mp3                                           │
└───────────────────┬─────────────────────────────────────┘
                    │ an audio file
                    ▼
┌─────────────────────────────────────────────────────────┐
│                   THE ART HISTORY STUDENT               │
│         heareth Shakespeare's answer in audio           │
└─────────────────────────────────────────────────────────┘
```

### Pre-processing (Performed But Once)

Before the bot may be consulted, the texts and documents must be indexed — a ritual performed only once:

```
Shakespeare plays, sonnets & art history PDFs → ingestor.py → ChromaDB (a vector store saved upon disk)
```

---

## The Structure of Folders — A Map of the Kingdom

```
shakespearechat/
├── main.py                  # orchestrates the full pipeline
├── config.py                # environment variables and configuration
├── requirements.txt
├── .env                     # API keys (never committed to git)
├── .gitignore
├── README.md
│
├── stt/
│   └── transcriber.py       # audio → text + language
│
├── rag/
│   ├── embedder.py          # text → vectors
│   ├── vector_store.py      # manages ChromaDB
│   ├── retriever.py         # seeketh the most relevant chunks
│   └── ingestor.py          # readeth documents and populateth the store
│
├── llm/
│   └── agent.py             # buildeth the Shakespeare prompt and calleth Claude's API
│
├── tts/
│   └── synthesizer.py       # text → audio
│
├── data/
│   └── docs/                # Shakespeare's works and art history texts (PDFs)
│
└── tests/
    ├── test_stt.py
    ├── test_rag.py
    └── test_llm.py
```

---

## Libraries Employed in This Endeavour

### STT — Speech to Text
| Library | Version | Purpose |
|---|---|---|
| `openai-whisper` | latest | Multilingual audio transcription, local and without cost per call |

### RAG — Retrieval-Augmented Generation
| Library | Version | Purpose |
|---|---|---|
| `sentence-transformers` | latest | Generateth multilingual embeddings with `paraphrase-multilingual-MiniLM-L12-v2` |
| `chromadb` | latest | Persistent vector store on disk for storing and seeking document chunks |
| `pypdf` | latest | Readeth and extracteth text from Shakespeare and art history PDFs |

### LLM — Large Language Model
| Library | Version | Purpose |
|---|---|---|
| `anthropic` | latest | Official SDK to summon the API of Claude (claude-sonnet-4-6), prompted to speak as Shakespeare |

### TTS — Text to Speech
| Library | Version | Purpose |
|---|---|---|
| `edge-tts` | latest | Neural voice synthesis via Microsoft Edge, free of charge, supporting multiple languages |

### Utilities
| Library | Version | Purpose |
|---|---|---|
| `python-dotenv` | latest | Loadeth environment variables from the `.env` file |

---

## AI Models Employed

| Model | Where it runneth | Cost | Purpose |
|---|---|---|---|
| `Whisper base` | Local (CPU/GPU) | Free | Audio transcription |
| `paraphrase-multilingual-MiniLM-L12-v2` | Local | Free | Embeddings for semantic search |
| `claude-sonnet-4-6` | Anthropic API | ~$0.005/question | Generating answers in Shakespeare's voice |
| Microsoft Neural Voices | edge-tts API | Free | Voice synthesis for the Bard's spoken word |

---

## Domain of Knowledge

ShakespeareChat answereth questions within these scholarly realms:

| Domain | Examples |
|---|---|
| **Shakespeare's Plays** | Themes, characters, plot, language, context of Hamlet, Macbeth, Othello, King Lear, etc. |
| **Shakespeare's Sonnets** | Interpretation, themes, structure, historical significance |
| **Elizabethan Era** | Theatre, society, politics, culture of 16th–17th century England |
| **Literary Analysis** | Dramatic devices, poetic forms, symbolism, narrative structure |
| **Art History Context** | Visual art, patronage, and culture surrounding Shakespeare's world |
| **Shakespeare's Life** | Biography, influences, legacy, the Globe Theatre |

---

## How to Run — Instructions for the Eager Student

### Backend (FastAPI)

```bash
# 1. Create and activate the virtual environment (from Agribot/)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set the API key in .env
ANTHROPIC_API_KEY=sk-ant-...

# 4. Add Shakespeare texts and art history PDFs to data/docs/ and run the ingestor (once only)
python -m rag.ingestor

# 5. Start the API server (from Agribot/)
python -m uvicorn api.server:app --reload
```

The API shall be served at `http://localhost:8000`.

### Frontend (React + Vite)

```bash
# In a separate terminal, from the frontend/ directory
cd ../frontend

npm install

npm run dev
```

The frontend shall open at `http://localhost:5173`.

---

## Troubleshooting — When Fortune Doth Not Favour Thee

### CORS error in the browser (`No 'Access-Control-Allow-Origin' header`)

This happeneth when an old backend process is still running on port 8000, and thy new server hath not truly taken its place. The stale process blocketh the request without the proper CORS headers.

**How to slay the old process:**

```bash
# Git Bash / Linux / Mac
taskkill //F //IM python.exe   # Windows Git Bash
# or
kill $(lsof -t -i:8000)        # Linux / Mac
```

Then restart the backend:

```bash
python -m uvicorn api.server:app --reload
```

---

## ML Pipeline (DVC) — The Tragedy Classifier

A reproducible machine-learning pipeline classifies Shakespeare text chunks as **Tragedy** or **Non-Tragedy** using [DVC (Data Version Control)](https://dvc.org).

### Pipeline DAG

```
 +---------+
 | prepare |
 +---------+
      *
      *
 +-----------+
 | featurize |
 +-----------+
      *
      *
  +-------+
  | train |
  +-------+
      *
      *
 +----------+
 | evaluate |
 +----------+
```

| Stage | Script | Description |
|---|---|---|
| `prepare` | `src/pipeline/prepare.py` | Chunk all `.txt` docs, label (Tragedy=1 / Non-Tragedy=0), balance classes |
| `featurize` | `src/pipeline/featurize.py` | TF-IDF vectorisation (5,000 features, bigrams), 75/25 stratified split |
| `train` | `src/pipeline/train.py` | Train 3 models: Naive Bayes, LR+SVD, LR+PCA |
| `evaluate` | `src/pipeline/evaluate.py` | Compute accuracy / precision / recall / F1; save confusion matrices |

### Reproduce the pipeline

```bash
# Ensure the 10 source .txt files are in data/docs/
dvc repro
```

DVC will skip stages whose inputs have not changed. To force a full re-run:

```bash
dvc repro --force
```

### Model metrics (`dvc metrics show`)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes + TF-IDF | **95.35%** | 93.73% | **97.21%** | **95.44%** |
| Logistic Regression + SVD | 94.56% | 94.67% | 94.42% | 94.55% |
| Logistic Regression + PCA | 94.36% | **94.77%** | 93.89% | 94.33% |

### GitHub Pages Report

Full report with confusion-matrix plots and the rendered notebook:
**[https://&lt;your-github-username&gt;.github.io/ShakespeareBot/](https://your-github-username.github.io/ShakespeareBot/)**

---

## Tests — That Truth May Be Proven

```bash
pytest tests/
```

- `test_stt.py` — testeth transcription with silent synthetic audio
- `test_rag.py` — testeth embeddings, insertion and vector search with an in-memory store
- `test_llm.py` — testeth the agent with a mocked API (consumeth no credits)
