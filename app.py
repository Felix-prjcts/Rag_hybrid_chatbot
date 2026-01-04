import streamlit as st
from rag import RAGEngine
import os

st.set_page_config(page_title="RAG Hybride Modularisé", layout="wide")
st.title(" Hybrid RAG Assistant (Demo)")

@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "bm25_db" not in st.session_state: st.session_state.bm25_db = None
if "chunks_db" not in st.session_state: st.session_state.chunks_db = []
if "messages" not in st.session_state: st.session_state.messages = []


with st.sidebar:
    st.header("📂 Data Source")
    
    # 1. Menu de sélection 
    source_choice = st.radio(
        "Choose Data Source:",
        ("Upload Files", "Demo: Solar System (Science)", "Demo: RGPD (Law)"),
        index=0
    )
    
    # Afficher l'uploader seulement si nécessaire
    uploaded_files = []
    if source_choice == "Upload Files":
        uploaded_files = st.file_uploader("Upload PDF/TXT", type=["txt", "pdf"], accept_multiple_files=True)
    
    st.markdown("---")

    if st.button("Build Index"):
        raw_docs = []
        
        # Demo Solaire
        if source_choice == "Demo: Solar System (Science)":
            demo_path = "data_demo/solar_system.pdf"
            if os.path.exists(demo_path):
                text = engine.extract_text_from_pdf(demo_path)
                raw_docs.append(("Solar_System.pdf", text))
                st.toast("Solar System loaded!", icon="🪐")
            else:
                st.error("File 'solar_system.pdf' missing in data_demo/")

        #Demo RGPD
        elif source_choice == "Demo: RGPD (Law)":
            demo_path = "data_demo/RGPD.pdf"
            if os.path.exists(demo_path):
                # On utilise la même fonction d'extraction du moteur
                text = engine.extract_text_from_pdf(demo_path)
                raw_docs.append(("RGPD.pdf", text))
                st.toast("GDPR/RGPD loaded!", icon="⚖️")
            else:
                st.error("File 'RGPD.pdf' missing in data_demo/")

        # Sinon Uploads Utilisateur
        elif source_choice == "Upload Files" and uploaded_files:
            for f in uploaded_files:
                if f.type == "application/pdf":
                    text = engine.extract_text_from_pdf(f)
                else:
                    text = f.read().decode("utf-8", errors="ignore")
                raw_docs.append((f.name, text))
            st.toast(f"{len(raw_docs)} files loaded!", icon="✅")

        # Lancement de l'indexation 
        if raw_docs:
            with st.spinner("Processing & Indexing..."):
                chunks = engine.process_documents(raw_docs)
                st.session_state.chunks_db = chunks
                
                v_db, s_db = engine.build_indices(chunks)
                st.session_state.vector_db = v_db
                st.session_state.bm25_db = s_db
                
                st.success(f"Index Ready with {len(chunks)} chunks!")
                
                # Message d'ex contextuel
                msg = "Ready!"
                if "RGPD" in source_choice:
                    msg = "Legal Assistant Ready! Try asking: 'Right to be forgotten' (Article 17)."
                elif "Solar" in source_choice:
                    msg = "Science Assistant Ready! Ask about planets."
                
                st.session_state.messages = [{"role": "assistant", "content": msg}]
        else:
            if source_choice == "Upload Files" and not uploaded_files:
                st.warning("Please upload a file first.")

# Interface Chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Your question..."):
    if not st.session_state.vector_db:
        st.error("Build index first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Recherche via Engine
                results = engine.search(
                    prompt, 
                    st.session_state.vector_db, 
                    st.session_state.bm25_db, 
                    st.session_state.chunks_db
                )
                
                # 2. Contexte & Génération
                context = "\n---\n".join([r["payload"]["text"] for r in results])
                answer = engine.generate_answer(prompt, context)
                
                st.markdown(answer)
                
                with st.expander("Sources"):
                    for r in results:
                        st.caption(f"Score: {r['score']:.4f} | {r['payload']['doc_id']}")
                        st.text(r['payload']['text'])
        
        st.session_state.messages.append({"role": "assistant", "content": answer})