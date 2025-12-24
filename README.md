# DhammAI – RAG-based Search on Early Buddhist Texts

DhammAI is a small, educational **Retrieval-Augmented Generation (RAG)** project built on top of the SuttaCentral Bilara Data (English translations by Bhikkhu Sujato). It combines classic information retrieval (BM25), semantic search (FAISS + embeddings), and an OpenAI chat model to answer natural language questions with **context-based, source-linked answers**.

## Requirements

- Python 3.10+
- OpenAI API key

## Environment configuration (**important**)

Create a `.env` file in the project root based on `.env.example`.  
Set:

```env
OPENAI_API_KEY=your_api_key_here
```

The application uses this key for embeddings and answer generation.

If you want to use other OpenAI compatible providers, override the base url and the model name in the `.env` file too.
```env
OPENAI_BASE_URL=your_api_provider_base_url
OPENAI_API_KEY=your_api_provider_api_key
OPENAI_EMBEDDING_MODEL=your_api_provider_embedding_model_name
```

## Installation & Setup

**1. Create and activate virtual environment:**

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS
```

**2. Install dependencies:**

```
pip install -r requirements.txt
```

**3. Build indexes:**

```
python scripts/01_data_prep.py   # Verify data loading
python scripts/02_chunk.py       # Create text chunks
python scripts/03_index_build.py # Build BM25 + FAISS indexes
```

Run these once, or again if you modify the corpus or indexing parameters.

## Usage

### 1. CLI hybrid QA

```
python scripts/04_query_hybrid.py
```

Type an English question; the script:

- runs hybrid retrieval (BM25 + FAISS with Reciprocal Rank Fusion),
- selects top chunks,
- calls the OpenAI model to generate a concise answer with references.

### 2. Streamlit UI

```
streamlit run app/ui_streamlit.py
```

Then open the shown URL (by default `http://localhost:8501`).

The UI:

- lets you enter questions,
- displays the generated answer,
- shows the underlying context chunks and source links so you can see where the answer came from.


## License

- **Code**: MIT License (see LICENSE file)
- **Translations**: CC0 (public domain), source: [SuttaCentral](https://suttacentral.net/)
  - Translator: Bhikkhu Sujato
  - Only technical modifications (segmentation for indexing)
  - Details: LICENSES/TRANSLATIONS_CC0.txt

## Notes

- The project is intended as a school project, learning-oriented RAG pipeline, not a production system.
- Once indexes are built (`index/*`), you don't need to rebuild them unless the data or logic changes.
- All textual sources follow the original SuttaCentral / Bilara Data licensing.
