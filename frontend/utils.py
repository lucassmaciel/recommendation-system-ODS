from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas import DataFrame


def paginate(df: DataFrame, page_size: int = 12, key: str = "page") -> DataFrame:
    total = len(df)
    if total == 0:
        return df
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

def paginate_catalog(df: DataFrame, page_size: int = 12, key: str = "cat_page") -> DataFrame:
    total = len(df)
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

def ensure_card_css():
    st.html("""
    <style>
      .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
      @media (max-width: 1200px) { .grid { grid-template-columns: repeat(3, 1fr); } }
      @media (max-width: 880px)  { .grid { grid-template-columns: repeat(2, 1fr); } }
      @media (max-width: 640px)  { .grid { grid-template-columns: 1fr; } }

      .card {
        display:flex; flex-direction:column;
        border:1px solid #e5e7eb; border-radius:12px; padding:12px;
        height: 420px; background:#fff;
      }
      .card img { width:100%; height:230px; object-fit:cover; border-radius:8px; }
      .card .title { font-weight:700; margin:8px 0 4px; }
      .card .meta  { color:#6b7280; font-size:0.9rem; }
      .spacer { margin-top:auto; }
    </style>
    """)

def render_card_grid(df: pd.DataFrame):
    ensure_card_css()
    html = ['<div class="grid">']
    for _, row in df.iterrows():
        img = (row.get("image") or "").replace('"', "&quot;")
        title = str(row.get("book", ""))
        author = row.get("author") if pd.notna(row.get("author")) else "—"
        year = row.get("year")
        year_txt = f"{int(year)}" if pd.notna(year) else "—"
        score = row.get("score")
        score_txt = f"{score:.4f}" if score is not None and pd.notna(score) else "—"

        html.append(f"""
        <div class="card">
          <img src="{img}">
          <div class="title">{title}</div>
          <div class="meta">Autor: {author}</div>
          <div class="spacer"></div>
          <div class="meta">Ano: {year_txt} · Score: {score_txt}</div>
        </div>
        """)
    html.append("</div>")
    st.html("".join(html))
