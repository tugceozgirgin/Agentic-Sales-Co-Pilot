import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.database.embedding import EmbeddingPipeline
from langchain_community.document_loaders import PyPDFLoader
from src.database.base_database import BaseDatabase
from openai import OpenAI

class SemanticDatabase(BaseDatabase):
    """
    Simplified semantic database that reads from a single PDF file.
    Uses Faiss for vector similarity search.
    """
    
    def __init__(self, 
                 pdf_path: str = "src/snippets.pdf",
                 persist_dir: str = "faiss_store", 
                 embedding_model: str = "all-MiniLM-L6-v2", 
                 chunk_size: int = 170, 
                 chunk_overlap: int = 0):
        super().__init__()
        self.pdf_path = pdf_path
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.index = None
        self.metadata = []
        if embedding_model == "openai":
            self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.model = SentenceTransformer(embedding_model)
        
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._load_pdf()
    
    def _load_pdf(self):
        """Load and index the PDF file."""
        if not os.path.exists(self.pdf_path):
            print(f"[WARNING] PDF file not found: {self.pdf_path}")
            return
        
        if self._try_load_index():
            print(f"[INFO] Loaded existing index from {self.persist_dir}")
            return

        print(f"[INFO] Loading PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        print(f"[INFO] Building vector store from PDF...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        
        self.metadata = [{"text": chunk.page_content} for chunk in chunks]
        
        embeddings_array = np.array(embeddings).astype('float32')
        dim = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings_array)
        
        self._save_index()
        print(f"[INFO] Indexed {len(chunks)} chunks from PDF")
    
    def _try_load_index(self) -> bool:
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(faiss_path)
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                return True
            except Exception as e:
                print(f"[WARNING] Failed to load index: {e}")
        return False
    
    def _save_index(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
    
    def search(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        
        query_emb = self.model.encode([query]).astype('float32')
        D, I = self.index.search(query_emb, top_k)
        
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata):
                result = {
                    "text": self.metadata[idx].get("text", ""),
                    "distance": float(dist),
                    "index": int(idx)
                }
                results.append(result)
        
        return results

    def populate(self, data: List[Dict[str, Any]]) -> bool:
        if not data:
            return False
        
        texts = [item.get('text', '') for item in data if 'text' in item]
        if not texts:
            return False
        
        embeddings = self.model.encode(texts).astype('float32')
        
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
        
        self.index.add(embeddings)
        self.metadata.extend([{"text": text} for text in texts])
        
        self._save_index()
        return True
    
    @staticmethod
    def initialize_and_populate(embedding_model: str = "all-MiniLM-L6-v2",
                                chunk_size: int = 170,
                                chunk_overlap: int = 0):
        print("[INFO] Initializing semantic database...")
        try:
            db = SemanticDatabase(
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            print("[INFO] Semantic database initialized and populated successfully.")
            return db
        except Exception as e:
            print(f"[ERROR] Failed to initialize semantic database: {e}")
            raise

# if __name__ == "__main__":
#     db = SemanticDatabase(pdf_path="src/snippets.pdf", chunk_size=200, chunk_overlap=0)
#     db.search("What should I do for client with 200,000 YTD spend and it is a RetaiL Company ?", top_k=5)