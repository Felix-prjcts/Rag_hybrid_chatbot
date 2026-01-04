import numpy as np
import unicodedata
import re
import fitz  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import pipeline
from rank_bm25 import BM25Okapi

class RAGEngine:
    def __init__(self):

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""]
        )

    def clean_text(self, raw: str) -> str:
        """Nettoyage standard du texte"""
        normalized = unicodedata.normalize("NFKD", raw) #standardiser les entrées avant le traitement vectoriel
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii") #ignore caractere speciaux, aide pour bm25 et reuit bruit
        return re.sub(r"\s+", " ", ascii_text).strip() #tout en un ligne

    def extract_text_from_pdf(self, file_stream) -> str:
        # Si c'est un chemin (str) ou un fichier uploadé (bytes)
        try:
            if isinstance(file_stream, str):
                doc = fitz.open(file_stream)
            else:
                doc = fitz.open(stream=file_stream.getvalue(), filetype="pdf")
            
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"

    def process_documents(self, raw_docs):
        """
        Transforme une liste de tuples (nom_fichier, contenu) en chunks.
        Retourne une liste de dictionnaires (chunks)
        """
        chunks = []
        for filename, text in raw_docs:
            cleaned = self.clean_text(text)
            for i, chunk in enumerate(self.splitter.split_text(cleaned)):
                chunks.append({
                    "doc_id": filename,
                    "chunk_id": f'{filename}_{i}',
                    "text": chunk,
                    "index": len(chunks)
                })
        return chunks

    def build_indices(self, chunks):
        """Construit les index Qdrant (Dense) et BM25 (Sparse)"""#
        # Dense (Qdrant)
        client = QdrantClient(":memory:")
        dim = self.embed_model.get_sentence_embedding_dimension()
        client.recreate_collection(
            collection_name="demo_rag",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        
        texts = [c["text"] for c in chunks]
        vectors = self.embed_model.encode(texts)
        points = [
            PointStruct(id=i, vector=v.tolist(), payload=chunks[i])
            for i, v in enumerate(vectors)
        ]
        client.upsert(collection_name="demo_rag", points=points)

        # Sparse (BM25)
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        return client, bm25

    def search(self, query, client, bm25, chunks, k=60):
        """Execute la recherche hybride avec RRF."""
        # Dense Search
        q_vec = self.embed_model.encode([query])[0].tolist()
        dense_res = client.query_points(
            collection_name="demo_rag", query=q_vec, limit=10, with_payload=True
        ).points #

        # Sparse Search
        tk_query = self._tokenize(query)
        bm25_scores = bm25.get_scores(tk_query)
        sparse_indices = np.argsort(bm25_scores)[::-1][:10]

        # Fusion (RRF)
        fused_scores = {}
        
        # Dense RRF
        for rank, point in enumerate(dense_res): #
            c_id = point.payload['chunk_id']
            if c_id not in fused_scores:
                fused_scores[c_id] = {"score": 0.0, "payload": point.payload}
            fused_scores[c_id]["score"] += 1 / (k + rank + 1)

        # Sparse RRF
        for rank, idx in enumerate(sparse_indices):
            chunk_data = chunks[idx]
            c_id = chunk_data['chunk_id']
            if c_id not in fused_scores:
                fused_scores[c_id] = {"score": 0.0, "payload": chunk_data}
            fused_scores[c_id]["score"] += 1 / (k + rank + 1)

        # Sort
        final_results = list(fused_scores.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:3]

    def generate_answer(self, query, context):
        """Génère la réponse avec FLAN-T5 """
        
        #On coupe manuellement le contexte s'il est vraiment trop grand
        max_chars = 1600 # Environ 300-400 tokens
        safe_context = context[:max_chars]
        
        # 2. Le Prompt Optimisé
        prompt = (
            f"Question: {query}\n\n"                      
            "Using the text below, answer the question above.\n" 
            f"Context:\n{safe_context}\n\n"               # contexte (Si coupé c'est pas grave)
            "Answer:"                                     # Trigger
        )
        
        return self.qa_model(prompt, max_length=256)[0]["generated_text"]
    
    def generate_answer(self, query, context):
        """Genère la réponse avec le LLM (flan-t5)"""
        
        # 1. On coupe manuellement le contexte s'il est vraiment trop grand
        # (Sécurité supplémentaire pour garder de la place pour la réponse)
        max_chars = 1600 # Environ 300-400 tokens
        safe_context = context[:max_chars]
        
        prompt = (
            f"Question: {query}\n\n"                      
            "Using the text below, answer the question above.\n" 
            f"Context:\n{safe_context}\n\n"               # Contexte (Si coupé, c'est pas grave)
            "Answer:"                                     #Trigger
        )
        return self.qa_model(prompt, max_length=256)[0]["generated_text"]

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())