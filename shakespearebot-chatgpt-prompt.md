# ChatGPT Image Prompt — ShakespeareBot Architecture

Paste the block below into ChatGPT (image generation).
If you also have `shakespearebot-pipeline.md`, attach it for extra fidelity — but the prompt is self-sufficient.

---

## PROMPT (copy everything between the lines)

---

Create a beautiful high-level technical architecture diagram for **ShakespeareBot** — a voice-powered AI chatbot that answers questions as William Shakespeare, combining RAG retrieval, an NLP tragedy classifier, an agentic LLM, and streaming audio output.

Preserve the exact workflow, MVC layer structure, component names, routing logic, and data flow described below. Do not add, remove, or change system behaviour.

**Style:** modern enterprise architecture infographic, polished SaaS engineering design, dark parchment/gold theme (deep navy or charcoal background, gold accent lines and labels, cream/ivory text), clean spacing, subtle depth, consistent iconography, premium visual hierarchy, balanced white space, elegant connector lines, minimal clutter. The overall aesthetic should feel like a Shakespearean scroll meets a modern tech blueprint.

---

### Structure — Four MVC Layers (left → right) + Data Layer (far right)

**LAYER 1 — VIEW (React 18 + Vite · localhost:5173)**
Components: Text Input, Mic Button (MediaRecorder API), SSE Stream Reader (sentence-by-sentence), Message Component (streaming cursor ▌), Audio Playback (base64 → mp3 via new Audio())
- User types or speaks a question
- SSE Stream Reader receives audio_chunk events and plays them sentence by sentence
- "Hearken Again" replay button on each message

**LAYER 2 — CONTROLLER (FastAPI + Uvicorn · localhost:8000)**
Components: CORS Middleware (allow: localhost:5173), Language Detection (langdetect), 4 endpoints:
- POST /api/chat/text/stream → SSE streaming (primary)
- POST /api/chat/audio/stream → SSE streaming (primary)
- POST /api/chat/text → non-streaming fallback
- POST /api/chat/audio → non-streaming fallback
- Conversation History: last 6 messages (3 turns), thread-safe threading.Lock
- Vite proxy forwards /api → :8000

**LAYER 3 — MODEL (three parallel sub-layers)**

*3A — NLP Classifier (src/classifier.py)*
- TF-IDF Vectorizer: 5,000 features, ngram(1,2), min_df=2
- Naive Bayes MultinomialNB α=0.1 · 95.35% accuracy
- Output: label (tragedy / non-tragedy) + confidence score (0.0–1.0)
- If confidence > 0.6: enriches ChromaDB search query toward tragedy or comedy/history corpus
- Passes query_type hint to AI Agent

*3B — AI Agent (llm/agent.py)*
- Claude claude-sonnet-4-6 via Anthropic HTTPS API · max_tokens=320
- Agentic loop: max 3 iterations
- 3 tools: search_shakespeare_texts (→ triggers RAG), answer_directly (→ skip RAG), ask_for_clarification (→ returns question to user)
- Shakespeare persona injected via system prompt (never changes)
- Receives: question + history + language + query_type + confidence

*3C — RAG Retrieval (rag/retriever.py)*
- Embedder: paraphrase-multilingual-MiniLM-L12-v2 (sentence-transformers, local CPU)
- ChromaDB cosine similarity search · fetch top 4 candidates (RAG_TOP_K=4)
- Word-overlap reranker → keep top 2 chunks
- Returns chunks prefixed with [Source: filename]

*3D — STT: Whisper tiny (stt/transcriber.py) — feeds into Controller*
- Local CPU · openai-whisper · multilingual auto-detect
- Returns: transcribed text + language code

*3E — TTS: edge-tts (tts/synthesizer.py) — feeds into View*
- Microsoft Neural Voices (free)
- Default: en-GB-RyanNeural (British male, theatrical)
- 8 languages: en, es, fr, pt, sw, ar, hi, ha
- Sentence splitter merges fragments < 20 chars
- Streams MP3 chunks as base64 SSE events

**LAYER 4 — DATA**
- ChromaDB PersistentClient: local disk (chroma_db/), collection: shakespeare_docs, cosine space
- Shakespeare .txt Docs: data/docs/ · 10 files · 4 tragedies, 3 comedies, 2 histories, 1 biography
- Trained ML Models: model/nb_tfidf.pkl · data/features/tfidf_vectorizer.pkl · DVC-tracked
- DVC ML Pipeline (offline): prepare → featurize → train → evaluate (produces model files)

---

### Directional arrows to show:
- User → Text Input OR Mic Button → POST /api/chat/[text|audio]/stream (via Vite proxy :5173 → :8000)
- Audio path: Controller → Whisper STT → text + language → Controller
- Controller → NLP Classifier → label + confidence → Controller (enriches search query)
- Controller → AI Agent (question + history + language + query_type)
- AI Agent → [search tool] → RAG Retriever → ChromaDB → top 2 chunks → Agent
- AI Agent → text response → Controller
- Controller → TTS sentence splitter → edge-tts → MP3 chunks → SSE stream → Frontend Audio Playback
- data/docs → ingestor (one-time) → ChromaDB
- DVC pipeline → model/nb_tfidf.pkl → NLP Classifier (loaded at startup)

---

### Special visual callouts to include:
- Badge on Claude: "claude-sonnet-4-6 · Anthropic API"
- Badge on Whisper: "tiny · local CPU · multilingual"
- Badge on TTS: "en-GB-RyanNeural · British theatrical voice"
- Badge on NLP: "95.35% accuracy · Naive Bayes"
- Badge on ChromaDB: "cosine similarity · 10 Shakespeare texts"
- Callout on agentic loop: "max 3 iterations · 3 tools"
- Callout on history: "_MAX_HISTORY=6 · thread-safe"
- Highlight the NLP Classifier → search query enrichment path as a gold decision branch
- Show SSE streaming path as a dashed animated-style arrow from TTS to frontend

### MVC labels:
- Explicitly label the three MVC bands: VIEW | CONTROLLER | MODEL
- Use a subtle horizontal band or sidebar label for each

### Color coding:
- VIEW layer: warm ivory / cream tones
- CONTROLLER layer: deep gold / amber tones
- MODEL layer: charcoal / slate with gold accents
- DATA layer: deep navy with subtle parchment texture
- SSE streaming path: dashed gold line
- RAG path: blue accent
- NLP classifier branch: green accent

### Icons to include:
- User icon (person) at the far left
- Browser/app icon for the React frontend
- Server/API icon for FastAPI
- Brain/AI icon for Claude and NLP Classifier
- Microphone icon for STT
- Speaker icon for TTS
- Database cylinder for ChromaDB
- Book/scroll icon for Shakespeare docs
- Pipeline/flow icon for DVC

**Tone:** technical, elegant, boardroom-ready, Shakespearean-meets-modern, architecture-review quality. Suitable for a university presentation to professor and peers.

---

## Assumptions baked into this prompt

None — all details verified from source code.
