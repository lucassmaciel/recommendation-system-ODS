from __future__ import annotations
import pandas as pd
import streamlit as st
from pandas import DataFrame

def paginate(df: DataFrame, page_size: int = 12, key: str = "page") -> DataFrame:
    total = len(df)
    if total == 0: return df
    page = st.session_state.get(key, 1)
    n_pages = max(1, (total + page_size - 1)//page_size)
    c1, c2, c3 = st.columns([1,2,1])
    with c1:
        if st.button("← Anterior", disabled=(page<=1), key=f"{key}_prev"):
            page -= 1
    with c3:
        if st.button("Próxima →", disabled=(page>=n_pages), key=f"{key}_next"):
            page += 1
    st.session_state[key] = page
    st.caption(f"Página {page}/{n_pages} — {total} itens")
    i0 = (page-1)*page_size
    return df.iloc[i0:i0+page_size]

def paginate_catalog(df: DataFrame, page_size: int = 12, key: str = "page") -> DataFrame:
    total = len(df)
    if total == 0: return df
    page = st.session_state.get(key, 1)
    n_pages = max(1, (total + page_size - 1)//page_size)
    c1, c2, c3 = st.columns([1,2,1])
    with c3:
        if st.button("Pesquisar", disabled=(page>=n_pages), key=f"{key}_next"):
            page += 1
    st.session_state[key] = page
    st.caption(f"Página {page}/{n_pages} — {total} itens")
    i0 = (page-1)*page_size
    return df.iloc[i0:i0+page_size]