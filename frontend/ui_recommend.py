from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()


def render_recommendation_card(row: pd.Series) -> None:
    """Renderiza um card de livro recomendado."""
    with st.container():
        # imagem do livro
        if pd.notna(row.get("image")):
            st.image(row["image"], width="stretch")
        else:
            st.image(
                "https://via.placeholder.com/350x500?text=No+Image", width="stretch"
            )

        # título e autor
        st.markdown(f"**{row.get('book', 'Sem título')}**")
        if pd.notna(row.get("author")):
            st.markdown(f"*{row['author']}*")

        # score e book_id em duas colunas
        col1, col2 = st.columns(2)
        with col1:
            if pd.notna(row.get("year")):
                st.caption(f"Ano: {int(row['year'])}")
        with col2:
            if pd.notna(row.get("book_id")):
                st.caption(f"ID: {int(row["book_id"])}")


def get_recommendations(user_id: str, top_n: int, metric: str) -> DataFrame | None:
    """Busca recomendações no backend."""
    try:
        payload = {"user_id": str(user_id), "top_n": int(top_n), "like_threshold": 7}

        r: requests.Response = requests.post(
            f"{os.environ.get("BACKEND_URL")}/v1/recomendar",
            params={"sim_metric": metric},
            json=payload,
            timeout=30,
        )
        r.raise_for_status()

        return pd.DataFrame(r.json().get("recommendations", []))
    except requests.RequestException as e:
        st.error(f"Erro ao buscar recomendações: {e}")
        return None


def render(user_df: DataFrame, books_df: DataFrame) -> None:
    """Renderiza a página de recomendações."""
    st.subheader("✨ Recomendações")
    st.caption("Sistema de recomendação usando filtragem colaborativa user-based.")

    # seleção de usuário
    user_id = str(st.session_state.get("current_user", ""))
    if not user_id:
        user_id = str(
            st.selectbox(
                "Selecione um usuário:",
                options=sorted(user_df["user_id"].unique()),
                index=0,
            )
        )
    # parâmetros de recomendação
    col1, col2 = st.columns(2)
    with col1:
        top_n: int = st.slider("Número de recomendações", 1, 20, 8)
    with col2:
        metric: str = st.selectbox(
            "Métrica de similaridade", options=["hybrid", "cosine", "pearson"], index=0
        )
    if st.button("🔍 Ver recomendações", width="stretch"):
        with st.spinner("Buscando recomendações..."):
            # buscar recomendações
            if not user_id:
                st.warning("Selecione um usuário válido.")
                return
            rec_df: DataFrame | None = get_recommendations(user_id, top_n, metric)
            if rec_df is None or rec_df.empty:
                st.warning("Sem recomendações para este usuário.")
                return
            if rec_df is None or rec_df.empty:
                st.warning("Sem recomendações para este usuário.")
                return

            # preparar catálogo
            catalog: DataFrame = books_df.sort_values(
                ["book", "image"], ascending=[True, False]
            ).drop_duplicates(subset=["book"], keep="first")

            # merge com informações do catálogo
            view: DataFrame = rec_df.merge(catalog, on="book", how="left").drop_duplicates(
                subset=["book"]
            )

            # renderizar grid de recomendações - ideal 3 por linha
            cols = st.columns(3)
            for i, (_, row) in enumerate(view.iterrows()):
                with cols[i % 3]:
                    render_recommendation_card(row)

            st.success(f"✨ {len(view)} recomendações exibidas para usuário {user_id}")
