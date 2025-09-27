from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas import DataFrame
from utils import paginate


def render_book_card(row: pd.Series) -> None:
  """Renderiza o card dos livros avaliados."""
  with st.container():
        if pd.notna(row.get("image")):
            st.image(row["image"], width="stretch")
        else:
            st.image("https://via.placeholder.com/350x500?text=No+Image", width="stretch")

        st.markdown(f"**{row.get('book', 'Sem título')}**")

        if pd.notna(row.get("author")):
            st.markdown(f"*{row['author']}*")

        # rating e ano em duas colunas e com tamanhos diferentes
        col1, col2 = st.columns(2)
        with col1:
            rating = row.get("rating")
            if pd.notna(rating):
                st.metric("Nota", f"{int(rating)}")
        with col2:
            year = row.get("year")
            if pd.notna(year):
                st.caption(f"Ano: {int(year)}")

def render(books_df: DataFrame) -> None:
    """Renderiza a página de avaliações geral."""
    st.subheader("Minhas avaliações")

    # verifica o usuário atual
    uid = st.session_state.current_user
    if not uid:
        st.info("Selecione um usuário na barra lateral.")
        return

    # valida colunas necessárias
    required_columns: set[str] = {"user_id", "book", "rating"}
    if not required_columns.issubset(books_df.columns):
        st.warning("books_info.csv não tem as colunas necessárias (user_id, book, rating).")
        return

    # pega os livros que o usuário avaliou do df
    hist = books_df[books_df["user_id"].astype(str) == str(uid)].copy()

    if hist.empty:
        st.caption("Você ainda não tem avaliações no dataset.")
        return

    # ordena por rating e título
    hist: DataFrame = hist.sort_values(["rating", "book"], ascending=[False, True])

    # pagina exibindo 8 livros
    page: DataFrame = paginate(hist, page_size=6, key="my_hist_page")

    cols = st.columns(3)
    for idx, (_, row) in enumerate(page.iterrows()):
        with cols[idx % 3]:
            render_book_card(row)

    st.divider()
