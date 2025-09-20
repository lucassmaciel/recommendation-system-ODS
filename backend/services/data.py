from pathlib import Path
import pandas as pd
from backend.core.config import settings

REQUIRED = {"User-ID", "Book-Title", "Rating"}

def get_dataset_path() -> Path:  ## enquanto Dataset local
    return settings.DATA_PATH

def load_ratings_df() -> pd.DataFrame:
    path = get_dataset_path()
    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado em {path}")

    df = pd.read_csv(path)

    # remove colunas-índice salvas no CSV (Unnamed: 0)
    bad_idx = [c for c in df.columns if c.lower().startswith("unnamed")]
    if bad_idx:
        df = df.drop(columns=bad_idx)

    # renomeia para nomes seguros - tava quebrando aqui
    rename_map = {
        "Book-Title": "book",
        "User-ID": "user",
        "Rating": "rating",
    }
    df = df.rename(columns=rename_map)

    expected = {"user", "book", "rating"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Esperava colunas {sorted(expected)}, encontrei {list(df.columns)}"
        )

    df = df.dropna(subset=["user", "book", "rating"]).copy()
    df["user"] = df["user"].astype(str)
    df["book"] = df["book"].astype(str)
    df["rating"] = df["rating"].astype(float)
    return df[["user", "book", "rating"]]