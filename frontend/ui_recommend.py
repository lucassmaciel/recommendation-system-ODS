from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from pandas import DataFrame
from utils import render_card_grid

BACKEND = "http://localhost:8000"  # ajuste se necessário

def render(user_df: DataFrame, books_df: DataFrame):
    st.subheader("Recomendações")
    st.caption("Este app usa filtragem colaborativa user-based (backend FastAPI).")

    user_id = st.session_state.current_user or st.selectbox(
        "Selecione o id do usuário:", user_df["user_id"].unique()
    )

    c1, c2, c3 = st.columns([1,1,2])
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

            rec_df = pd.DataFrame(recs)
            view = rec_df.merge(books_df, on="book", how="left")

            render_card_grid(view)

            st.success(f"{len(view)} recomendações exibidas para usuário {user_id}.")
        except requests.RequestException as e:
            st.error(f"Erro ao buscar recomendações: {e}")
