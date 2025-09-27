from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
from pandas import DataFrame

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed-data"

def _require(p: Path):
    if not p.exists():
        st.error(f"Arquivo ausente: {p}")
        st.stop()

@st.cache_data
def load_data() -> tuple[DataFrame, DataFrame]:
    user_path = DATA_DIR / "final_user_df.csv"
    books_path = DATA_DIR / "books_info.csv"
    _require(user_path)
    _require(books_path)
    user_df = pd.read_csv(user_path)
    books_df = pd.read_csv(books_path)
    return user_df, books_df
