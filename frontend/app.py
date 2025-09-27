from __future__ import annotations

import requests
import state
import streamlit as st
from data import load_data
from ui_catalog import render as catalog
from ui_my_ratings import render as myratings
from ui_recommend import render as recommend

st.set_page_config(page_title="Sistema de Recomendação", layout="wide")

# Endpoint do backend
BACKEND = "http://localhost:8000"

# CSS global
st.html("""
<style>
  img { height: 300px !important; object-fit: cover !important; }
</style>
""")


def sidebar(user_ids):
    with st.sidebar:
        st.header("Conta")

        current = st.session_state.get("current_user")
        st.caption(f"Logado como: **{current or '—'}**")

        st.divider()

        options = [str(x) for x in user_ids]
        default_idx = options.index(str(current)) if current and str(current) in options else 0 if options else 0
        uid = st.selectbox("Selecionar usuário existente", options=options, index=default_idx if options else 0)
        if st.button("Entrar", use_container_width=True):
            st.session_state.current_user = str(uid)
            st.toast(f"Logado como {uid}")

        st.divider()

        new_id = st.text_input("Cadastrar novo ID", value="", placeholder="ex.: 9999")
        if st.button("Cadastrar", use_container_width=True):
            if not new_id.strip():
                st.warning("Informe um ID.")
            else:
                try:
                    r = requests.post(f"{BACKEND}/v1/users/signup",
                                      json={"user_id": new_id.strip()},
                                      timeout=20)
                    r.raise_for_status()
                    payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    if not payload.get("created", True):
                        st.warning(f"O ID **{uid}** já existe. Escolha outro.")
                    else:
                        try:
                            load_data.clear()
                        except Exception:
                            st.cache_data.clear()
                        st.success(f"Usuário {new_id.strip()} cadastrado.")
                        st.rerun()
                except requests.RequestException as e:
                    st.error(f"Falha ao cadastrar: {e}")


def main():
    state.init()
    user_df, books_df = load_data()

    # Para que apareça sempre em ordem crescente
    user_ids = (
        user_df["user_id"]
        .dropna()
        .astype(int)
        .sort_values()
        .astype(str)
        .unique()
    )

    sidebar(user_ids)

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
