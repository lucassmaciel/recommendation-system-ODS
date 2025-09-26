from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from pandas import DataFrame
from utils import render_card_grid

BACKEND = "http://localhost:8000"


@st.cache_data(show_spinner=False)
def _catalog_from_ratings(books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói um catálogo 1-por-livro a partir do CSV de avaliações (long).
    Heurística:
      1) prioriza linhas com imagem
      2) depois ano mais recente
    Mantém apenas colunas usadas pelos cards.
    """
    df = books_df.copy()

    keep_cols = [c for c in ["book", "author", "year", "image"] if c in df.columns]

    has_img = df.get("image")
    df["_has_img"] = has_img.notna() & (has_img.astype(str).str.len() > 0) if has_img is not None else False

    year_col = "year" if "year" in df.columns else None
    sort_by = ["_has_img"]
    ascending = [False]
    if year_col:
        sort_by.append(year_col)
        ascending.append(False)

    df = df.sort_values(sort_by, ascending=ascending, kind="mergesort")

    # 1 linha por título
    df = df.drop_duplicates(subset=["book"], keep="first")

    df = df[keep_cols].copy()
    df.drop(columns=[c for c in ["_has_img"] if c in df.columns], inplace=True, errors="ignore")
    return df


def render(user_df: DataFrame, books_df: DataFrame):
    st.subheader("Recomendações")
    st.caption("Este app usa filtragem colaborativa user-based (backend FastAPI).")

    user_id = st.session_state.get("current_user") or st.selectbox(
        "Selecione o id do usuário:", user_df["user_id"].unique()
    )

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        top_n = st.slider("Top-N", min_value=1, max_value=20, value=8)
    with c2:
        metric = st.selectbox("Métrica", ["hybrid", "cosine", "pearson"], index=0)

    if st.button("Ver recomendações"):
        payload = {"user_id": str(user_id), "top_n": int(top_n), "like_threshold": 7}
        try:
            r = requests.post(
                f"{BACKEND}/v1/recomendar",
                params={"sim_metric": metric},
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            recs = r.json().get("recommendations", [])
            if not recs:
                st.warning("Sem recomendações para este usuário.")
                return

            rec_df = pd.DataFrame(recs).drop_duplicates(subset=["book"])
            st.dataframe(rec_df)

            books_catalog = _catalog_from_ratings(books_df)

            # Merge sem explosão
            view = rec_df.merge(books_catalog, on="book", how="left").drop_duplicates(subset=["book"])
            render_card_grid(view)
            st.success(f"{len(view)} recomendações exibidas para usuário {user_id}.")
        except requests.RequestException as e:
            st.error(f"Erro ao buscar recomendações: {e}")
