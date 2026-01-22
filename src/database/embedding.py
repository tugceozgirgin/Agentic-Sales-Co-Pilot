from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
import re
from langchain_core.documents import Document

# def load_pdf_document(pdf_path: str) -> List[Any]:
#     """Load a PDF file from the given path."""
#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()
#     print(f"[INFO] Loaded {len(documents)} pages from {pdf_path}")
#     return documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        if model_name == "openai":
            self.model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.model = SentenceTransformer(model_name)

        print(f"[INFO] Loaded embedding model: {model_name}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        try:
            response = self.model.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=512
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        full_text = "\n".join(doc.page_content for doc in documents)

        raw_snippets = re.split(r"(?:Snippet\s+\d+:)", full_text)
        raw_snippets = [s.strip() for s in raw_snippets if s.strip()]

        chunks = []
        for i, snippet in enumerate(raw_snippets, start=1):
            chunks.append(
                Document(
                    page_content=snippet,
                    metadata={
                        "snippet_id": i,
                        "type": "policy_rule",
                        "source": "sales_playbook"
                    }
                )
            )

        print(f"[INFO] Split playbook into {len(chunks)} snippet-level chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        
        if self.model_name == "openai":
            embeddings = [self._get_embedding(text) for text in texts]
            print(f"[INFO] Embeddings shape: {len(embeddings)}")
            return np.array(embeddings)
        else:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"[INFO] Embeddings shape: {embeddings.shape}")
            return embeddings


# if __name__ == "__main__":
#     docs = load_pdf_document("src/snippets.pdf")
#     emb_pipe = EmbeddingPipeline(model_name="openai")
#     chunks = emb_pipe.chunk_documents(docs)
#     embeddings = emb_pipe.embed_chunks(chunks)
#     print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)