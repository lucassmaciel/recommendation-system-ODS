from __future__ import annotations
import pandas as pd
import streamlit as st
from pandas import DataFrame
from utils import paginate_catalog

def render(books_df: DataFrame):
    st.subheader("Catálogo de Livros")
    search = st.text_input("Buscar livro ou autor:")
    df = books_df.copy()
    if search:
        mask = df["book"].astype(str).str.contains(search, case=False, na=False) | \
               df["author"].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]
    # dedupe por book para não repetir capas
    df = df.sort_values(["book","image"], ascending=[True, False]).drop_duplicates("book")
    page = paginate_catalog(df, page_size=12, key="cat_page")
    cols = st.columns(3)
    for i, row in page.iterrows():
        with cols[i % 3], st.container(border=True):
            st.image(row.get("image",""), width="stretch")
            st.markdown(f"**{row['book']}**")
            st.caption(f"Autor: {row['author']}")
            if pd.notna(row.get("year")):
                st.caption(f"Ano: {int(row['year'])}")
