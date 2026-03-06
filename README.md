# 📈 AI Stock Market Intelligence

Automated financial news analysis powered by **BERT embeddings**, **semantic similarity filtering**, and **LLM-driven market insights**.

## 🔄 Pipeline Flow

```
News API  →  BERT Embeddings  →  Similarity Filtering  →  LLM Market Analysis  →  Streamlit UI
   ↓              ↓                      ↓                        ↓                     ↓
NEWSDATA.io   HuggingFace          Cosine Similarity       Mistral AI           Dark-Mode
  API         all-MiniLM-L6-v2     Score Threshold         Report Gen           Dashboard
```

**Refer PIPELINE JPG FOR DETAILED WORKFLOW**

### Step-by-Step

1. **📰 News API** — Fetches latest financial news from NEWSDATA.io based on user query
2. **🧠 BERT Embeddings** — Encodes articles using `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace Inference API
3. **🔍 Similarity Filtering** — Ranks articles by cosine similarity to query, filters low-relevance content
4. **🤖 LLM Market Analysis** — Sends top articles to Mistral AI for structured market intelligence report
5. **✨ Streamlit UI** — Premium dark-mode dashboard with glassmorphism, metrics, and interactive Q&A

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.9+ |
| Mistral API Key | For LLM-powered analysis |
| NEWSDATA.io API Key | Free tier at [newsdata.io](https://newsdata.io) |
| HuggingFace API Key | For BERT embeddings inference |

## Project Structure

```
stock_ai_streamlit/
├── app.py              # Streamlit dashboard (main entry)
├── news_fetcher.py     # NEWSDATA.io API client
├── embedder.py         # BERT embedding & similarity ranking (HuggingFace)
├── processor.py        # Text cleaning & merging
├── llm.py              # LLM report generation & Q&A (Mistral)
├── vector_store.py     # In-memory vector store (numpy)
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
└── README.md
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys in .env
OPENAI_API_KEY=your-mistral-api-key
OPENAI_BASE_URL=https://api.mistral.ai/v1
OPENAI_MODEL=mistral-small-latest
NEWSDATA_API_KEY=your-newsdata-api-key
HF_API_KEY=your-huggingface-api-key
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 3. Run the app
**Get into the directory***
python -m streamlit run app.py
```

## Features

- **🔍 Smart Search** — Search any market topic, stock, or sector
- **🧠 BERT Ranking** — Semantic similarity scoring to find most relevant articles
- **📊 Market Report** — AI-generated report with 7 structured sections
- **💬 Q&A** — Ask follow-up questions answered by AI using fetched news
- **📈 Metrics Dashboard** — Visual pipeline stats (articles, scores, timing)
- **🌙 Dark Mode** — Premium glassmorphism UI with smooth animations
- **⚡ API-Based** — No browser/Selenium required
