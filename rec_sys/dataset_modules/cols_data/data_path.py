from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parents[2]

# Directories
RAW_DATA_DIR = BASE_DIR / "raw_data"
PROCESSED_DATA_DIR = BASE_DIR / "dataset"
UNIQUE_USER_DIR = BASE_DIR / "unique_user_data"
GRAPH_DIR = BASE_DIR / "pt_graphs"

print(BASE_DIR)

# Files
RAW_MUS_FILE = RAW_DATA_DIR / "mus_instr.json"
RAW_METADATA_FILE = RAW_DATA_DIR / "meta_instr.json"
DATASET_FILE = PROCESSED_DATA_DIR / "dataset.parquet"
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.parquet"

# Constants
MAX_NAME_LEN = 10
MAX_DESC_LEN = 15

# ---- Sentence Transformer ----
MODEL_NAME = "all-MiniLM-L6-v2"

# ---- Config ----
WORDS_FIELDS: List[str] = ["product_name", "product_info", "comment"]
BATCH_EMB_SIZE: int = 64

USER_FIELD = "reviewerID"
