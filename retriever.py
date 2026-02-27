"""Vector similarity search against MongoDB Atlas."""

from dataclasses import dataclass

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

import config

load_dotenv()


@dataclass
class SearchResult:
    chunk_id: str
    file_path: str
    module: str
    chunk_type: str
    name: str
    source: str
    docstring: str
    parent_class: str
    score: float


class Retriever:
    """Embeds a query and retrieves the most relevant code chunks from MongoDB."""

    def __init__(self):
        self._embedding_model: SentenceTransformer | None = None
        self._mongo_collection = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._embedding_model

    @property
    def collection(self):
        if self._mongo_collection is None:
            self._mongo_collection = MongoClient(config.MONGODB_URI)[config.DB_NAME][config.COLLECTION_NAME]
        return self._mongo_collection

    def search(self, question: str, top_k: int = config.DEFAULT_TOP_K) -> list[SearchResult]:
        """Embed the question and run Atlas $vectorSearch to find the top-k matching chunks."""
        query_vector = self.embedding_model.encode(question).tolist()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": config.VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 1, "file_path": 1, "module": 1, "chunk_type": 1,
                    "name": 1, "source": 1, "docstring": 1, "parent_class": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        return [
            SearchResult(
                chunk_id=doc["_id"],
                file_path=doc.get("file_path", ""),
                module=doc.get("module", ""),
                chunk_type=doc.get("chunk_type", ""),
                name=doc.get("name", ""),
                source=doc.get("source", ""),
                docstring=doc.get("docstring", ""),
                parent_class=doc.get("parent_class", ""),
                score=doc.get("score", 0.0),
            )
            for doc in self.collection.aggregate(pipeline)
        ]

    def format_results_as_context(self, results: list[SearchResult]) -> str:
        """Format retrieved chunks into a context block for the LLM prompt."""
        if not results:
            return "No relevant code found."

        parts = []
        for i, r in enumerate(results, 1):
            header = f"### [{i}] {r.name}  ({r.file_path} · {r.chunk_type})"
            if r.docstring:
                header += f"\n# {r.docstring[:200]}"
            parts.append(f"{header}\n```python\n{r.source}\n```")
        return "\n\n".join(parts)
