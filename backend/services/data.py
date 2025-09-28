from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.core.config import settings

REQUIRED = {"User-ID", "Book-Title", "Rating"}

def get_dataset_path() -> Path:  ## enquanto Dataset local
    return settings.DATA_PATH

@lru_cache(maxsize=1)
def load_user_item_matrix() -> pd.DataFrame:
    df = pd.read_csv(settings.DATA_PATH, dtype={"user_id": str})
    if "user_id" in df.columns:
        df = df.set_index("user_id")

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def invalidate_matrix_cache():
    load_user_item_matrix.cache_clear()
