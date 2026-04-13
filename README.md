# AgriBot — Voice-Powered Farming Assistant

## Time

| Nome | |
|---|---|
| Liggia Cruz | |
| Chao | |
| Emmanuel | |

---

## Visão Geral

AgriBot é um chatbot de voz especializado em agricultura, projetado para agricultores rurais em regiões em desenvolvimento. O agricultor faz perguntas falando no seu idioma nativo — espanhol, francês, suaíli, português, entre outros — e recebe a resposta em áudio, sem precisar ler ou digitar nada.

O bot é um especialista restrito: só responde sobre agricultura (solo, pragas, irrigação, épocas de plantio, armazenamento pós-colheita). Essa limitação de domínio é intencional — ela garante respostas mais precisas e confiáveis para o agricultor.

---

## Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                     AGRICULTOR                          │
│          fala em espanhol, francês, suaíli...           │
└───────────────────┬─────────────────────────────────────┘
                    │ arquivo de áudio (.ogg / .wav)
                    ▼
┌─────────────────────────────────────────────────────────┐
│  PASSO 1 — STT  (stt/transcriber.py)                    │
│  Whisper escuta o áudio e devolve:                      │
│  → texto:   "¿Cuándo debo plantar maíz?"                │
│  → idioma:  "es"                                        │
└───────────────────┬─────────────────────────────────────┘
                    │ texto + idioma
                    ▼
┌─────────────────────────────────────────────────────────┐
│  PASSO 2 — RAG  (rag/retriever.py)                      │
│  Busca nos PDFs da FAO os trechos mais relevantes       │
│  para a pergunta do agricultor                          │
│  → chunk 1: "O milho germina com solo acima de 10°C..." │
│  → chunk 2: "Plantio ideal: março a maio em regiões..." │
│  → chunk 3: "Espaçamento recomendado: 70cm entre..."    │
└───────────────────┬─────────────────────────────────────┘
                    │ texto + idioma + chunks
                    ▼
┌─────────────────────────────────────────────────────────┐
│  PASSO 3 — LLM  (llm/agent.py)                          │
│  Claude recebe a pergunta + contexto da FAO e           │
│  gera uma resposta prática em espanhol                  │
│  → "Plante maíz entre marzo y mayo cuando el suelo..."  │
└───────────────────┬─────────────────────────────────────┘
                    │ resposta em texto
                    ▼
┌─────────────────────────────────────────────────────────┐
│  PASSO 4 — TTS  (tts/synthesizer.py)                    │
│  edge-tts converte o texto em áudio com voz neural      │
│  → output.mp3  (voz masculina em espanhol)              │
└───────────────────┬─────────────────────────────────────┘
                    │ arquivo de áudio
                    ▼
┌─────────────────────────────────────────────────────────┐
│                     AGRICULTOR                          │
│              ouve a resposta em áudio                   │
└─────────────────────────────────────────────────────────┘
```

### Pré-processamento (roda uma vez)

Antes de usar o bot, os documentos da FAO precisam ser indexados:

```
PDFs da FAO → ingestor.py → ChromaDB (banco vetorial salvo em disco)
```

---

## Estrutura de Pastas

```
agribot/
├── main.py                  # orquestra o pipeline completo
├── config.py                # variáveis de ambiente e configurações
├── requirements.txt
├── .env                     # API keys (nunca sobe pro git)
├── .gitignore
├── README.md
│
├── stt/
│   └── transcriber.py       # áudio → texto + idioma
│
├── rag/
│   ├── embedder.py          # texto → vetores
│   ├── vector_store.py      # gerencia o ChromaDB
│   ├── retriever.py         # busca os chunks mais relevantes
│   └── ingestor.py          # lê PDFs e popula o banco
│
├── llm/
│   └── agent.py             # monta o prompt e chama a API do Claude
│
├── tts/
│   └── synthesizer.py       # texto → áudio
│
├── data/
│   └── docs/                # PDFs agrícolas da FAO
│
└── tests/
    ├── test_stt.py
    ├── test_rag.py
    └── test_llm.py
```

---

## Bibliotecas Utilizadas

### STT — Speech to Text
| Biblioteca | Versão | Função |
|---|---|---|
| `openai-whisper` | latest | Transcrição de áudio multilíngue local, sem custo por chamada |

### RAG — Retrieval-Augmented Generation
| Biblioteca | Versão | Função |
|---|---|---|
| `sentence-transformers` | latest | Gera embeddings multilíngues com o modelo `paraphrase-multilingual-MiniLM-L12-v2` |
| `chromadb` | latest | Banco vetorial persistente em disco para armazenar e buscar chunks dos PDFs |
| `pypdf` | latest | Leitura e extração de texto dos PDFs da FAO |

### LLM — Large Language Model
| Biblioteca | Versão | Função |
|---|---|---|
| `anthropic` | latest | SDK oficial para chamar a API do Claude (claude-sonnet-4-6) |

### TTS — Text to Speech
| Biblioteca | Versão | Função |
|---|---|---|
| `edge-tts` | latest | Síntese de voz neural via Microsoft Edge, gratuita, suporta EN/ES/FR/PT/SW/AR/HI/HA |

### Utilitários
| Biblioteca | Versão | Função |
|---|---|---|
| `python-dotenv` | latest | Carrega variáveis de ambiente do arquivo `.env` |

---

## Modelos de IA Utilizados

| Modelo | Onde roda | Custo | Função |
|---|---|---|---|
| `Whisper base` | Local (CPU/GPU) | Gratuito | Transcrição de áudio |
| `paraphrase-multilingual-MiniLM-L12-v2` | Local | Gratuito | Embeddings para busca semântica |
| `claude-sonnet-4-6` | API Anthropic | ~$0,005/pergunta | Geração da resposta agrícola |
| Vozes neurais Microsoft | API edge-tts | Gratuito | Síntese de voz multilíngue |

---

## Idiomas Suportados

| Código | Idioma | Voz TTS |
|---|---|---|
| `en` | Inglês | en-US-GuyNeural |
| `es` | Espanhol | es-ES-AlvaroNeural |
| `fr` | Francês | fr-FR-HenriNeural |
| `pt` | Português | pt-BR-AntonioNeural |
| `sw` | Suaíli | sw-KE-RafikiNeural |
| `ar` | Árabe | ar-SA-HamedNeural |
| `hi` | Hindi | hi-IN-MadhurNeural |
| `ha` | Hauçá | ha-NE-AbdullahNeural |

O idioma é detectado automaticamente pelo Whisper a partir do áudio do agricultor.

---

## Como Rodar

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar a API key no .env
ANTHROPIC_API_KEY=sk-ant-...

# 4. Adicionar PDFs da FAO em data/docs/ e rodar o ingestor (uma vez)
python -m rag.ingestor

# 5. Rodar o bot com um áudio
python main.py minha_pergunta.ogg resposta.mp3
```

---

## Testes

```bash
pytest tests/
```

- `test_stt.py` — testa transcrição com áudio sintético silencioso
- `test_rag.py` — testa embeddings, inserção e busca vetorial com banco em memória
- `test_llm.py` — testa o agent com mock da API (não consome créditos)