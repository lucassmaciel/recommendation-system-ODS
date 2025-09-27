from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas import DataFrame


def paginate(df: DataFrame, page_size: int = 16, key: str = "page") -> DataFrame:
    total: int = len(df)
    if total == 0:
        return df

    if key not in st.session_state:
        st.session_state[key] = 1

    page = st.session_state[key]
    n_pages: int = max(1, (total + page_size - 1) // page_size)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Anterior", disabled=(page <= 1), key=f"{key}_prev"):
            st.session_state[key] = page - 1
            st.rerun()

    with col2:
        st.markdown(f"<div style='text-align: center'>Página {page}/{n_pages}</div>", unsafe_allow_html=True)

    with col3:
        if st.button("Próxima →", disabled=(page >= n_pages), key=f"{key}_next"):
            st.session_state[key] = page + 1
            st.rerun()
    st.caption(f"Total: {total} itens")

    # Retorna slice da página atual
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx]

def paginate_catalog(df: DataFrame, page_size: int = 12, key: str = "cat_page") -> DataFrame:
    total: int = len(df)
    if total == 0:
        return df
    page = st.session_state.get(key, 1)
    n_pages = max(1, (total + page_size - 1) // page_size)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("← Anterior", disabled=(page <= 1), key=f"{key}_prev"):
            page -= 1
    with c3:
        if st.button("Próxima →", disabled=(page >= n_pages), key=f"{key}_next"):
            page += 1

    st.session_state[key] = page
    st.caption(f"Página {page}/{n_pages} — {total} itens")

    i0 = (page - 1) * page_size
    return df.iloc[i0:i0 + page_size]

def render_card_grid(df: pd.DataFrame):

    if df.empty:
        st.warning("Nenhum item para exibir")
        return

    cols= st.columns(4)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 4] and st.container():
            st.image(row["image"], use_container_width=True)

            st.markdown(f"**{row.get('book', 'Sem título')}**")

            if pd.notna(row.get("author")):
                st.markdown(f"*{row['author']}*")

            col1, col2 = st.columns(2)
            with col1:
                if pd.notna(row.get("score")):
                    st.caption(f"Score: {row['score']:.2f}")
            with col2:
                if pd.notna(row.get("year")):
                    st.caption(f"Ano: {int(row['year'])}")
                    st.write("")

            st.markdown("---")
