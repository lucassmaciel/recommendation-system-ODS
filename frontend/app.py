from __future__ import annotations

import state
import streamlit as st
from data import load_data
from ui_catalog import render as catalog
from ui_my_ratings import render as myratings
from ui_recommend import render as recommend

st.set_page_config(page_title="Sistema de Recomendação", layout="wide")

# CSS global
st.html("""
<style>
  img { height: 300px !important; object-fit: cover !important; }
</style>
""")


def sidebar(user_ids):
    with st.sidebar:
        st.header("Conta")
        uid = st.selectbox("Selecionar usuário", user_ids)
        if st.button("Usar esta conta"):
            st.session_state.current_user = str(uid)
        st.caption(f"Logado como: **{st.session_state.current_user or '—'}**")

def main():
    state.init()
    user_df, books_df = load_data()
    sidebar(user_df["user_id"].unique())

    st.markdown("# :rainbow[Sistema de recomendação de livros]")
    st.caption("Esse sistema utiliza filtragem colaborativa e correlação híbrida para calcular a distância entre os itens.")

    tab_rec, tab_cat, tab_my = st.tabs(["✨ Recomendações", "📚 Catálogo", "⭐ Minhas avaliações"])
    with tab_rec:
        recommend(user_df, books_df)
    with tab_cat:
        catalog(books_df)
    with tab_my:
        myratings(books_df)

if __name__ == "__main__":
    main()
