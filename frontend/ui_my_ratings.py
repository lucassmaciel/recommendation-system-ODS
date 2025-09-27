from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas import DataFrame
from utils import paginate


def render(books_df: DataFrame):
    st.subheader("Minhas avaliações")

    uid = st.session_state.current_user
    if not uid:
        st.info("Selecione um usuário na barra lateral.")
        return

    # --- HISTÓRICO (do CSV books_info.csv) ---
    if not {"user_id", "book", "rating"}.issubset(books_df.columns):
        st.warning("books_info.csv não tem as colunas necessárias (user_id, book, rating).")
        hist = pd.DataFrame()
    else:
        hist = books_df[books_df["user_id"].astype(str) == str(uid)].copy()
        hist = hist.sort_values(["rating", "book"], ascending=[False, True])

    st.markdown("### Histórico")
    if hist.empty:
        st.caption("Você ainda não tem avaliações no dataset.")
    else:
        page = paginate(hist, page_size=12, key="my_hist_page")

        st.html("""
        <style>
          .grid { display:grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
          .card {
            display:flex; flex-direction:column;
            border:1px solid #e5e7eb; border-radius:12px; padding:12px;
            height: 420px; /* força todos a mesma altura */
            background:#fff;
          }
          .card img {
            width:100%; height:230px; object-fit:cover; border-radius:8px;
          }
          .card .title { font-weight:700; margin:8px 0 4px; }
          .card .meta  { color:#6b7280; font-size:0.9rem; }
          .spacer { margin-top:auto; }
        </style>
        """)

        html = ['<div class="grid">']
        for _, row in page.iterrows():
            img = (row.get("image") or "").replace('"', "&quot;")
            title = str(row.get("book", ""))
            rating = int(row["rating"]) if pd.notna(row["rating"]) else "—"
            author = row.get("author", "—")
            year = int(row["year"]) if pd.notna(row.get("year")) else "—"

            html.append(f"""
            <div class="card">
              <img src="{img}">
              <div class="spacer"></div>
              <div class="meta">Título: {title}</div>
              <div class="meta">Nota: {rating}</div>
              <div class="meta">Autor: {author}</div>
              <div class="meta">Ano: {year}</div>
            </div>
            """)

        html.append("</div>")
        st.html("".join(html))

    st.divider()
