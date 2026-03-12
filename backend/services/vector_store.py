"""
Vector Store Service
Manages ChromaDB for persistent vector storage and retrieval
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_assistant"


class VectorStore:
    """
    ChromaDB-backed vector store for document embeddings.
    Supports persistent storage, similarity search, and metadata filtering.
    """

    def __init__(self, persist_directory: str = "database/vector_db"):
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

    def _get_client(self):
        """Lazy initialization of ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info("ChromaDB client initialized at: %s", self.persist_directory)
            except ImportError:
                raise RuntimeError("ChromaDB not installed. Run: pip install chromadb")
        return self._client

    def _get_collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            count = self._collection.count()
            logger.info(
                "Collection '%s' ready (%d existing documents)",
                COLLECTION_NAME, count
            )
        return self._collection

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add document chunks with embeddings to the store.
        Uses upsert to handle duplicates gracefully.
        """
        collection = self._get_collection()

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            logger.info("Upserted %d documents into ChromaDB", len(ids))
        except Exception as e:
            logger.error("Failed to store documents: %s", str(e))
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search using cosine distance.

        Returns:
            List of dicts with text, metadata, distance, and id
        """
        collection = self._get_collection()

        if collection.count() == 0:
            logger.warning("Vector store is empty. No documents to search.")
            return []

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"]
            )

            # Parse results
            items = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    # Convert cosine distance to similarity score (0-1, higher is better)
                    similarity = 1.0 - distance

                    items.append({
                        "id": doc_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": similarity,
                        "distance": distance,
                        "embedding": results["embeddings"][0][i] if results.get("embeddings") else None
                    })

            logger.debug("Search returned %d results", len(items))
            return items

        except Exception as e:
            logger.error("Search failed: %s", str(e))
            raise

    def list_sources(self) -> List[str]:
        """Get list of all unique source documents."""
        collection = self._get_collection()

        if collection.count() == 0:
            return []

        try:
            # Get all metadatas
            results = collection.get(include=["metadatas"])
            sources = list({
                meta["source"]
                for meta in results.get("metadatas", [])
                if "source" in meta
            })
            return sorted(sources)
        except Exception as e:
            logger.error("Failed to list sources: %s", str(e))
            return []

    def get_document_count(self) -> int:
        """Get total number of stored chunks."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def delete_source(self, source: str) -> int:
        """Delete all chunks belonging to a specific source file. Returns count deleted."""
        collection = self._get_collection()
        try:
            results = collection.get(where={"source": source}, include=["metadatas"])
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                logger.info("Deleted %d chunks for source: %s", len(ids), source)
            return len(ids)
        except Exception as e:
            logger.error("Failed to delete source %s: %s", source, str(e))
            raise

    def clear(self) -> None:
        """Delete all documents from the collection."""
        try:
            client = self._get_client()
            client.delete_collection(COLLECTION_NAME)
            self._collection = None  # Reset so it gets recreated
            logger.info("Cleared all documents from vector store")
        except Exception as e:
            logger.error("Failed to clear vector store: %s", str(e))
            raise