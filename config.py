import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI: str = os.getenv("MONGODB_URI", "")
DB_NAME: str = os.getenv("DB_NAME", "ultralytics_assistant")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "code_chunks")
VECTOR_INDEX_NAME: str = "vector_index"

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"

REPO_URL: str = "https://github.com/ultralytics/ultralytics.git"
REPO_DIR: str = os.getenv("REPO_DIR", "./ultralytics_repo")
INDEX_DIRS: list[str] = [
    "ultralytics/models",
    "ultralytics/engine",
    "ultralytics/data",
]

MAX_CHUNK_LINES: int = 80
MIN_CHUNK_LINES: int = 5
CHUNK_OVERLAP_LINES: int = 10

DEFAULT_TOP_K: int = 6
LLM_TEMPERATURE: float = 0.2
LLM_MAX_TOKENS: int = 1024


def validate() -> list[str]:
    """Return names of required config keys that are not set."""
    missing = []
    if not MONGODB_URI:
        missing.append("MONGODB_URI")
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    return missing
