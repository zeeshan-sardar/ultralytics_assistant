"""
Builds the searchable knowledge base by cloning the Ultralytics repo,
parsing Python files into semantic chunks via AST, embedding each chunk,
and upserting everything into MongoDB Atlas.

Usage:
    uv run python indexer.py
"""

import ast
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from pymongo import MongoClient, ReplaceOne
from pymongo.operations import SearchIndexModel
from sentence_transformers import SentenceTransformer

import config

load_dotenv()


@dataclass
class CodeChunk:
    """A semantically complete unit of source code — a class, method, or function."""

    chunk_id: str
    file_path: str
    module: str
    chunk_type: str
    name: str
    source: str
    lineno_start: int
    lineno_end: int
    docstring: str = ""
    parent_class: str = ""
    decorators: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)

    def build_embedding_text(self) -> str:
        """
        Combine module path, name, docstring, and source into the text that gets embedded.

        Prepending natural language metadata (module, name, docstring) alongside
        the raw source bridges the vocabulary gap between user questions and code tokens,
        significantly improving retrieval quality.
        """
        parts = [
            f"Module: {self.module}",
            f"Type: {self.chunk_type}",
            f"Name: {self.name}",
        ]
        if self.docstring:
            parts.append(f"Description: {self.docstring}")
        parts.append(self.source)
        return "\n".join(parts)

    def to_mongodb_document(self) -> dict:
        return {
            "_id": self.chunk_id,
            "file_path": self.file_path,
            "module": self.module,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "source": self.source,
            "lineno_start": self.lineno_start,
            "lineno_end": self.lineno_end,
            "docstring": self.docstring,
            "parent_class": self.parent_class,
            "decorators": self.decorators,
            "embedding": self.embedding,
        }


def clone_or_update_repo() -> Path:
    """Clone the Ultralytics repo with --depth=1, or pull if it already exists."""
    repo_dir = Path(config.REPO_DIR)
    if repo_dir.exists():
        print("Repository already cloned. Pulling latest changes…")
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--quiet"], check=True)
    else:
        print(f"Cloning {config.REPO_URL}…")
        subprocess.run(
            ["git", "clone", "--depth=1", "--quiet", config.REPO_URL, str(repo_dir)],
            check=True,
        )
    return repo_dir


def file_path_to_module_name(file_path: Path, repo_root: Path) -> str:
    """Convert a file path to a dotted Python module name."""
    parts = list(file_path.relative_to(repo_root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def extract_docstring(node: ast.AST) -> str:
    try:
        return ast.get_docstring(node) or ""
    except Exception:
        return ""


def extract_decorator_names(node) -> list[str]:
    names = []
    for decorator in getattr(node, "decorator_list", []):
        if isinstance(decorator, ast.Name):
            names.append(decorator.id)
        elif isinstance(decorator, ast.Attribute):
            base = decorator.value.id if isinstance(decorator.value, ast.Name) else "?"
            names.append(f"{base}.{decorator.attr}")
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            names.append(decorator.func.id)
    return names


def get_source_lines(full_source: str, node) -> tuple[str, int, int]:
    lines = full_source.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno]), node.lineno, node.end_lineno


def split_if_too_long(source: str) -> list[tuple[str, int]]:
    """Split oversized functions into overlapping sub-chunks to preserve context at boundaries."""
    lines = source.splitlines()
    if len(lines) <= config.MAX_CHUNK_LINES:
        return [(source, 0)]

    sub_chunks = []
    step = config.MAX_CHUNK_LINES - config.CHUNK_OVERLAP_LINES
    for start in range(0, len(lines), step):
        window = lines[start : start + config.MAX_CHUNK_LINES]
        if len(window) >= config.MIN_CHUNK_LINES:
            sub_chunks.append(("\n".join(window), start))
    return sub_chunks or [(source, 0)]


def make_chunk_id(file_path: str, *identifiers) -> str:
    raw = ":".join([file_path, *[str(i) for i in identifiers]])
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def parse_file_into_chunks(file_path: Path, repo_root: Path) -> list[CodeChunk]:
    """Parse a single Python file into CodeChunks using AST traversal."""
    source_text = file_path.read_text(encoding="utf-8", errors="replace")
    module_name = file_path_to_module_name(file_path, repo_root)
    relative_path = str(file_path.relative_to(repo_root))

    try:
        syntax_tree = ast.parse(source_text)
    except SyntaxError:
        return []

    chunks: list[CodeChunk] = []

    mod_doc = extract_docstring(syntax_tree)
    if mod_doc:
        chunks.append(CodeChunk(
            chunk_id=make_chunk_id(relative_path, "__module__", 0),
            file_path=relative_path,
            module=module_name,
            chunk_type="module_docstring",
            name=module_name,
            source=f'"""{mod_doc}"""',
            lineno_start=1,
            lineno_end=1,
            docstring=mod_doc,
        ))

    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.ClassDef):
            class_source, cs, ce = get_source_lines(source_text, node)
            header = "\n".join(class_source.splitlines()[:30])
            chunks.append(CodeChunk(
                chunk_id=make_chunk_id(relative_path, node.name, node.lineno),
                file_path=relative_path,
                module=module_name,
                chunk_type="class",
                name=f"{module_name}.{node.name}",
                source=header,
                lineno_start=cs,
                lineno_end=min(ce, cs + 30),
                docstring=extract_docstring(node),
                decorators=extract_decorator_names(node),
            ))

            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                meth_source, ms, _ = get_source_lines(source_text, item)
                for idx, (part_source, offset) in enumerate(split_if_too_long(meth_source)):
                    suffix = f"_part{idx}" if idx else ""
                    chunks.append(CodeChunk(
                        chunk_id=make_chunk_id(relative_path, f"{node.name}.{item.name}{suffix}", ms + offset),
                        file_path=relative_path,
                        module=module_name,
                        chunk_type="method",
                        name=f"{module_name}.{node.name}.{item.name}",
                        source=part_source,
                        lineno_start=ms + offset,
                        lineno_end=ms + offset + part_source.count("\n"),
                        docstring=extract_docstring(item) if not idx else "",
                        parent_class=node.name,
                        decorators=extract_decorator_names(item) if not idx else [],
                    ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.col_offset != 0:
                continue
            fn_source, fs, _ = get_source_lines(source_text, node)
            for idx, (part_source, offset) in enumerate(split_if_too_long(fn_source)):
                suffix = f"_part{idx}" if idx else ""
                chunks.append(CodeChunk(
                    chunk_id=make_chunk_id(relative_path, f"{node.name}{suffix}", fs + offset),
                    file_path=relative_path,
                    module=module_name,
                    chunk_type="function",
                    name=f"{module_name}.{node.name}",
                    source=part_source,
                    lineno_start=fs + offset,
                    lineno_end=fs + offset + part_source.count("\n"),
                    docstring=extract_docstring(node) if not idx else "",
                    decorators=extract_decorator_names(node) if not idx else [],
                ))

    return chunks


def find_all_python_files(repo_root: Path) -> Iterator[Path]:
    for directory_name in config.INDEX_DIRS:
        directory = repo_root / directory_name
        if directory.exists():
            yield from directory.rglob("*.py")
        else:
            print(f"  Warning: directory not found — {directory}")


class ChunkEmbedder:
    """Wraps SentenceTransformer to embed code chunks in batches."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        print(f"Loading embedding model '{model_name}'…")
        self.model = SentenceTransformer(model_name)
        self.vector_dimensions: int = self.model.get_sentence_embedding_dimension()
        print(f"  Vector size: {self.vector_dimensions} dimensions")

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()


def get_mongo_collection():
    client = MongoClient(config.MONGODB_URI)
    return client[config.DB_NAME][config.COLLECTION_NAME]


def create_vector_search_index_if_missing(collection, vector_dimensions: int) -> None:
    """Create the Atlas knnVector search index if it does not already exist."""
    db = collection.database
    if collection.name not in db.list_collection_names():
        print(f"  Creating collection '{collection.name}'…")
        db.create_collection(collection.name)

    existing = [idx.get("name") for idx in collection.list_search_indexes()]
    if config.VECTOR_INDEX_NAME in existing:
        print(f"Vector search index '{config.VECTOR_INDEX_NAME}' already exists ✓")
        return

    print(f"Creating Atlas vector search index '{config.VECTOR_INDEX_NAME}'…")
    collection.create_search_index(SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    "embedding": {
                        "type": "knnVector",
                        "dimensions": vector_dimensions,
                        "similarity": "cosine",
                    }
                },
            }
        },
        name=config.VECTOR_INDEX_NAME,
    ))
    print("  ↳ Index will be ready to query in ~1-2 minutes.")


def upsert_all_chunks(collection, chunks: list[CodeChunk]) -> int:
    """Bulk upsert chunks using chunk_id as the stable key."""
    if not chunks:
        return 0
    operations = [
        ReplaceOne({"_id": chunk.chunk_id}, chunk.to_mongodb_document(), upsert=True)
        for chunk in chunks
    ]
    result = collection.bulk_write(operations)
    return result.upserted_count + result.modified_count


def run_indexing() -> None:
    missing_config = config.validate()
    if missing_config:
        print(f"\nERROR: Set these in your .env file: {', '.join(missing_config)}")
        sys.exit(1)

    repo_root = clone_or_update_repo()

    print(f"\nParsing Python files from: {', '.join(config.INDEX_DIRS)}")
    all_chunks: list[CodeChunk] = []
    file_count = 0
    for python_file in find_all_python_files(repo_root):
        all_chunks.extend(parse_file_into_chunks(python_file, repo_root))
        file_count += 1
    print(f"  {file_count} files → {len(all_chunks)} raw chunks")

    seen_ids: set[str] = set()
    unique_chunks = [c for c in all_chunks if not (c.chunk_id in seen_ids or seen_ids.add(c.chunk_id))]
    print(f"  After deduplication: {len(unique_chunks)} unique chunks")

    embedder = ChunkEmbedder()
    print(f"\nEmbedding {len(unique_chunks)} chunks…")
    vectors = embedder.embed_texts([c.build_embedding_text() for c in unique_chunks])
    for chunk, vector in zip(unique_chunks, vectors):
        chunk.embedding = vector

    print("\nConnecting to MongoDB Atlas…")
    collection = get_mongo_collection()
    create_vector_search_index_if_missing(collection, embedder.vector_dimensions)

    print("Writing to MongoDB…")
    num_written = upsert_all_chunks(collection, unique_chunks)
    print(f"  {num_written} documents written. Total: {collection.count_documents({})} documents")
    print("\n✅ Indexing complete! Wait ~2 minutes for the vector index to activate.")
    print("   Then run: uv run streamlit run app.py")


if __name__ == "__main__":
    run_indexing()
