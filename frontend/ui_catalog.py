from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
from pandas import DataFrame
from streamlit_star_rating import st_star_rating
from utils import paginate

_USE_STAR_RATING = True

BACKEND_URL = "http://localhost:8000"

# -------- modal de avaliação --------
@st.dialog("Avaliar livro")
def show_rating_dialog(book_title: str):
    """Exibe um modal para o usuário avaliar um livro."""
    user_id = st.session_state.get("current_user")
    if not user_id:
        st.warning("⚠️ Selecione um usuário na barra lateral para poder avaliar.")
        if st.button("Fechar"):
            st.rerun()
        return

    st.markdown(f"**{book_title}**")
    st.markdown("---")

    if _USE_STAR_RATING:
        rating: int = st_star_rating(label="Sua nota", maxValue=10, defaultValue=7, key="rate_stars", size=25)
    else:
        rating: int = st.slider("Sua nota", 1, 10, 7, key="rate_slider", help="1 = Ruim, 10 = Excelente")

    if rating:
        st.info(f"Nota selecionada: {rating}/10")

    col1, col2 = st.columns(2)
    if col1.button("Salvar avaliação", type="primary", use_container_width=True):
        payload = {"user_id": str(user_id), "book": str(book_title), "rating": int(rating)}
        try:
            with st.spinner("Salvando..."):
                response = requests.post(f"{BACKEND_URL}/v1/rating", json=payload, timeout=10)
                response.raise_for_status()
            st.success("Avaliação salva com sucesso!")
            st.cache_data.clear() # Limpa o cache para recarregar as recomendações
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Falha ao salvar avaliação: {e}")

    if col2.button("Cancelar", use_container_width=True):
        st.rerun()

def render_book_card(row: pd.Series) -> None:
    """Renderiza um card de livro do catálogo."""
    with st.container():
        # imagem do livro
        st.image(row["image"], width="stretch")

        # título e autor
        st.markdown(f"**{row.get('book', 'Sem título')}**")
        if pd.notna(row.get("author")):
            st.markdown(f"*{row.author}*")

        # ano, book_id e botão de avaliar em três colunas
        col1, col2 , col3= st.columns([1, 1, 1])
        with col1:
            if pd.notna(row.get("year")):
                st.caption(f"📅 {int(row.year)}")
        with col2:
            bid = row.get("books_id", row.get("book_id"))
            if pd.notna(bid):
                try:
                    bid_str = str(int(bid))
                except Exception:
                    bid_str = str(bid)
                st.caption(f"ID: {bid_str}")
        with col3:
            key_suffix = row.get("books_id", row.name)
            if st.button("⭐ Avaliar", key=f"rate_{key_suffix}", type="secondary", width="stretch"):
                show_rating_dialog(row.book)

# -------- página do catálogo --------
def render(books_df: DataFrame) -> None:
    """Renderiza a página do catálogo."""
    st.subheader("📚 Catálogo de Livros")

    # Barra de busca
    search: str = st.text_input(
        "🔍 Buscar livro ou autor:",
        placeholder="Digite o nome do livro ou autor..."
    )

    # Filtrar dataframe
    df: DataFrame = books_df.copy()
    if search:
        mask = (
            df["book"].astype(str).str.contains(search, case=False, na=False) |
            df["author"].astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]

        if df.empty:
            st.warning("Nenhum livro encontrado.")
            return

    # remover duplicatas
    df = (df.sort_values(["book", "image"], ascending=[True, False])
            .drop_duplicates(subset=["book"]))

    # paginação
    page: DataFrame = paginate(df, page_size=9, key="catalog_page")

    cols = st.columns(3)
    for idx, (_, row) in enumerate(page.iterrows()):
        with cols[idx % 3]:
            render_book_card(row)

    st.divider()
