from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from pandas import DataFrame
from streamlit_star_rating import st_star_rating
from utils import paginate_catalog

_HAS_STARS = True

BACKEND = "http://localhost:8000"

# -------- modal de avaliação --------
@st.dialog("Avaliar livro")
def _rating_dialog(book: str):
    uid = st.session_state.get("current_user")
    if not uid:
        st.warning("⚠️ Selecione um usuário na barra lateral para avaliar.")
        if st.button("Fechar", type="secondary"):
            st.rerun()
        return

    st.markdown(f"**{book}**")
    st.markdown("---")

    if _HAS_STARS:
        nota = st_star_rating(
            label="Sua nota",
            maxValue=10,
            defaultValue=7,
            key="rate_stars_dialog",
            size=25
        )
    else:
        nota = st.slider("Sua nota", 1, 10, 7, key="rate_slider_dialog", help="1 = Ruim, 10 = Excelente")

    st.info(f"Nota selecionada: {nota}/10")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Salvar avaliação", type="primary", use_container_width=True):
            with st.spinner("Salvando..."):
                try:
                    payload = {"user_id": str(uid), "book": str(book), "rating": int(nota)}
                    r = requests.post(f"{BACKEND}/v1/rating", json=payload, timeout=20)
                    r.raise_for_status()
                    st.success("Avaliação salva com sucesso!")
                    st.cache_data.clear()
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Falha ao salvar avaliação: {e}")
    with c2:
        if st.button("Cancelar", use_container_width=True):
            st.rerun()

# -------- página do catálogo --------
def render(books_df: DataFrame):
    st.subheader("📚 Catálogo de Livros")

    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Buscar livro ou autor:", placeholder="Digite o nome do livro ou autor...")
    with col2:
        st.write("")  # espaçamento

    df = books_df.copy()
    if search:
        mask = df["book"].astype(str).str.contains(search, case=False, na=False) | \
               df["author"].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]

        if len(df) == 0:
            st.warning("Nenhum livro encontrado com os critérios de busca.")
            st.stop()

    # dedupe por book para não repetir capas
    df = df.sort_values(["book", "image"], ascending=[True, False]).drop_duplicates("book")

    # Mostrar quantos livros foram encontrados
    if search:
        st.info(f"{len(df)} livro(s) encontrado(s)")

    page = paginate_catalog(df, page_size=12, key="cat_page")

    num_books = len(page)
    if num_books > 0:
        num_cols = 3
        cols = st.columns(num_cols)

        for i, row in enumerate(page.itertuples()):
            col_idx = i % num_cols
            with cols[col_idx].container(border=True, height=600):
                    # correção de warning que tava dando
                    img_url = row.image if hasattr(row, "image") and pd.notna(row.image) else ""
                    st.image(str(img_url))
                    st.markdown("---")

                    st.markdown(f"**{row.book}**")
                    st.caption(f"👤 {row.author}")

                    if pd.notna(getattr(row, "year", None)):
                        st.caption(f"📅 {int(row.year)}")

                    action_cols = st.columns([1, 1])
                    with action_cols[1]:
                        if st.button("⭐ Avaliar", key=f"rate_{row.Index}", type="secondary"):
                            _rating_dialog(row.book)
