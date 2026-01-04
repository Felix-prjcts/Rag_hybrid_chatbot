import streamlit as st
import numpy as np
import unicodedata
import re
import fitz  # PyMuPDF
from pathlib import Path

# Imports ML (Plus de rank_bm25 ici !)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import pipeline

# --- 1. Page Config ---
st.set_page_config(page_title="Dense RAG Only", layout="wide")
st.title("🧠 Pure Dense RAG (Vector Search Only)")
st.markdown("""
**Architecture:** Embeddings only (`all-MiniLM-L6-v2`).
**Logic:** Semantic similarity (Cosine Distance). No keyword matching (BM25).
""")

# --- 2. Core Functions ---

def clean_text(raw: str) -> str:
    normalized = unicodedata.normalize("NFKD", raw)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"\s+", " ", ascii_text)
    return cleaned.strip()

def extract_text_from_pdf_path(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def extract_text_from_upload(uploaded_file) -> str:
    with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
    return text

def make_chunks(texts_list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for filename, text in texts_list:
        for i, chunk in enumerate(splitter.split_text(text)):
            chunks.append({
                "doc_id": filename,
                "chunk_id": f'{filename}_{i}',
                "text": chunk,
                # Plus besoin d'index numérique pour BM25
            })
    return chunks

# --- 3. Load Models ---
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
    return embed_model, qa_model

embed_model, qa_model = load_models()

# --- 4. Session State ---
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "messages" not in st.session_state: st.session_state.messages = []

# --- 5. Sidebar ---
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_files = st.file_uploader("Upload files (PDF/TXT)", type=["txt", "pdf"], accept_multiple_files=True)
    st.markdown("---")
    
    use_demo = st.checkbox("Load 'Solar System' PDF", value=False)

    if st.button("Build Dense Index"):
        raw_docs = []
        
        # Load Demo
        if use_demo:
            demo_path = "data_demo/solar_system.pdf"
            try:
                text = extract_text_from_pdf_path(demo_path)
                raw_docs.append(("Solar_System_Demo.pdf", clean_text(text)))
                st.toast("Demo PDF loaded!", icon="🪐")
            except:
                st.error("Demo file not found.")
            
        # Load Uploads
        elif uploaded_files:
            for up_file in uploaded_files:
                try:
                    if up_file.type == "application/pdf":
                        raw_docs.append((up_file.name, clean_text(extract_text_from_upload(up_file))))
                    else:
                        raw_docs.append((up_file.name, clean_text(up_file.read().decode("utf-8"))))
                except: pass
            st.toast(f"{len(raw_docs)} files loaded!", icon="✅")
        
        # Indexing (ONLY Qdrant)
        if raw_docs:
            with st.spinner("Embedding & Indexing..."):
                chunks = make_chunks(raw_docs)
                
                client = QdrantClient(":memory:")
                dim = embed_model.get_sentence_embedding_dimension()
                client.recreate_collection("dense_rag", VectorParams(size=dim, distance=Distance.COSINE))
                
                # Vectorisation
                texts = [c["text"] for c in chunks]
                vectors = embed_model.encode(texts)
                
                points = [
                    PointStruct(id=i, vector=v.tolist(), payload=chunks[i])
                    for i, v in enumerate(vectors)
                ]
                client.upsert("dense_rag", points)
                st.session_state.vector_db = client

                st.success(f"Dense Index Ready! ({len(chunks)} vectors)")
                if not st.session_state.messages:
                    st.session_state.messages.append({"role": "assistant", "content": "Ready! I use only vector search."})

# --- 6. Chat Interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    if not st.session_state.vector_db:
        st.error("Please build the index first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching vectors..."):
                
                # --- PURE DENSE RETRIEVAL ---
                # 1. On vectorise la question
                query_vector = embed_model.encode([prompt])[0].tolist()
                
                # 2. On interroge Qdrant directement
                search_result = st.session_state.vector_db.query_points(
                    collection_name="dense_rag",
                    query=query_vector,
                    limit=4, # Top 4 vecteurs les plus proches
                    with_payload=True
                ).points
                
                # Plus de fusion RRF ! On prend juste les résultats.
                
                # --- CONTEXT ---
                context_text = "\n\n---\n\n".join([r.payload["text"] for r in search_result])
                
                full_prompt = (
                    "You are a helpful assistant. Answer the question using the context below.\n"
                    f"Context:\n{context_text}\n\n"
                    f"Question: {prompt}\n\nAnswer:"
                )
                
                response = qa_model(full_prompt, max_length=256)[0]["generated_text"]
                st.markdown(response)
                
                # Sources (Score Cosinus)
                with st.expander("🔎 View Dense Scores (Cosine Similarity)"):
                    for r in search_result:
                        # Le score ici est la "Similarité" (proche de 1 = très similaire)
                        st.markdown(f"**Score:** `{r.score:.4f}`")
                        st.caption(r.payload['text'])

        st.session_state.messages.append({"role": "assistant", "content": response})