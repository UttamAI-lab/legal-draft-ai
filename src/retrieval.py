"""
Retrieval Layer
---------------
Chunks processed documents, embeds them, stores in ChromaDB,
and retrieves relevant passages for a given drafting task.
"""

import hashlib
import chromadb
from chromadb.utils import embedding_functions
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class RetrievedChunk:
    """A retrieved piece of evidence from source documents."""
    text: str
    source: str
    page: int
    chunk_index: int
    relevance_score: float
    document_id: str


class RetrievalLayer:
    """
    Chunks, embeds, and retrieves from processed documents.
    
    Uses sentence-transformers for local embeddings (no API cost)
    and ChromaDB as the vector store.
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(exist_ok=True)
        
        # Local embedding model — no API key needed
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_document(self, processed_doc) -> int:
        """
        Chunk and embed a ProcessedDocument into the vector store.
        Returns number of chunks added.
        """
        chunks = self._chunk_text(
            processed_doc.cleaned_text,
            source=processed_doc.source_path,
            pages=processed_doc.pages
        )
        
        if not chunks:
            return 0
        
        # Prepare for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = self._make_id(chunk["text"], chunk["source"], chunk["chunk_index"])
            
            # Skip if already exists
            existing = self.collection.get(ids=[chunk_id])
            if existing["ids"]:
                continue
            
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "document_id": chunk["document_id"]
            })
        
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        
        return len(ids)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a drafting query.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count() or 1)
        )
        
        chunks = []
        if not results["documents"] or not results["documents"][0]:
            return chunks
        
        for i, (doc, meta, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            # ChromaDB cosine distance: 0=identical, 2=opposite
            relevance = 1 - (distance / 2)
            
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                page=meta.get("page", 0),
                chunk_index=meta.get("chunk_index", i),
                relevance_score=round(relevance, 3),
                document_id=meta.get("document_id", "")
            ))
        
        return sorted(chunks, key=lambda x: x.relevance_score, reverse=True)

    def _chunk_text(self, text: str, source: str, pages: list,
                    chunk_size: int = 500, overlap: int = 100) -> list[dict]:
        """
        Smart chunking: tries to split on paragraph boundaries,
        falls back to sentence-level, then character-level.
        """
        chunks = []
        document_id = self._make_id(source, "", 0)
        
        # Split by double newline (paragraph) first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = []
        current_len = 0
        chunk_index = 0
        
        for para in paragraphs:
            para_len = len(para)
            
            # If single paragraph is too long, split by sentence
            if para_len > chunk_size:
                sentences = self._split_sentences(para)
                for sent in sentences:
                    if current_len + len(sent) > chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        page = self._find_page_for_chunk(chunk_text, pages)
                        chunks.append({
                            "text": chunk_text,
                            "source": source,
                            "page": page,
                            "chunk_index": chunk_index,
                            "document_id": document_id
                        })
                        # Overlap: keep last part
                        overlap_text = ' '.join(current_chunk)[-overlap:]
                        current_chunk = [overlap_text, sent]
                        current_len = len(overlap_text) + len(sent)
                        chunk_index += 1
                    else:
                        current_chunk.append(sent)
                        current_len += len(sent)
            else:
                if current_len + para_len > chunk_size and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    page = self._find_page_for_chunk(chunk_text, pages)
                    chunks.append({
                        "text": chunk_text,
                        "source": source,
                        "page": page,
                        "chunk_index": chunk_index,
                        "document_id": document_id
                    })
                    current_chunk = [para]
                    current_len = para_len
                    chunk_index += 1
                else:
                    current_chunk.append(para)
                    current_len += para_len
        
        # Add remaining
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            page = self._find_page_for_chunk(chunk_text, pages)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page,
                "chunk_index": chunk_index,
                "document_id": document_id
            })
        
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter for legal text."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_page_for_chunk(self, chunk_text: str, pages: list) -> int:
        """Find which page a chunk likely came from."""
        for page_info in pages:
            if chunk_text[:100] in page_info.get("text", ""):
                return page_info.get("page", 1)
        return 1

    def _make_id(self, *parts) -> str:
        """Generate a stable hash ID."""
        combined = "|".join(str(p) for p in parts)
        return hashlib.md5(combined.encode()).hexdigest()

    def clear(self):
        """Clear all stored documents."""
        self.client.delete_collection("legal_documents")
        self.collection = self.client.get_or_create_collection(
            name="legal_documents",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )