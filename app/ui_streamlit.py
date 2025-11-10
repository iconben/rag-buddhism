import json
from pathlib import Path

import streamlit as st

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

from lib.bm25_utils import BM25Index
from lib.faiss_utils import FaissIndex
from lib.rag_core import load_chunk_lookup, retrieve_hybrid, generate_answer

# index paths
INDEX_DIR = Path("index")
BM25_PATH = INDEX_DIR / "bm25.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
FAISS_META_PATH = INDEX_DIR / "faiss_meta.jsonl"
CHUNK_LOOKUP_PATH = INDEX_DIR / "chunk_lookup.json"


@st.cache_resource
def load_runtime():
    # load indexes and lookup once
    bm25 = BM25Index.load(str(BM25_PATH))
    faiss = FaissIndex.load(str(FAISS_INDEX_PATH), str(FAISS_META_PATH))
    chunk_lookup = load_chunk_lookup(str(CHUNK_LOOKUP_PATH))
    return bm25, faiss, chunk_lookup


def main():
    # streamlit UI
    st.set_page_config(page_title="Sutta RAG demo")
    st.title("RAG search: Early Buddhist texts")

    bm25, faiss, chunk_lookup = load_runtime()

    query = st.text_input("Question (English)")

    if st.button("Search & Answer") and query:
        # run hybrid retrieval + RAG
        contexts = retrieve_hybrid(query, bm25, faiss, chunk_lookup)
        answer = generate_answer(query, contexts)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Source contexts")
        for c in contexts:
            st.markdown(f"- [{c.chunk_id}] [{c.sc_link}]({c.sc_link})")
            st.caption(c.text[:400] + ("..." if len(c.text) > 400 else ""))


if __name__ == "__main__":
    main()
