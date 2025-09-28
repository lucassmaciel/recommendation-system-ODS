from __future__ import annotations

import os
from typing import TYPE_CHECKING

import requests
import state
import streamlit as st
from data import load_data
from dotenv import load_dotenv
from pandas.core.frame import DataFrame
from ui_catalog import render as catalog
from ui_my_ratings import render as myratings
from ui_recommend import render as recommend

if TYPE_CHECKING:
    from pandas import DataFrame

load_dotenv()

st.set_page_config(page_title="Sistema de Recomendação", layout="wide", page_icon="📚")
st.html("<style>img { height: 300px !important; object-fit: cover !important; }</style>") # limita o tamanho da imagem dos livros

# --- SIDEBAR E AUTENTICAÇÃO ---
def sidebar(users_map: DataFrame):
    """Renderiza a barra lateral para login e cadastro."""
    with st.sidebar:
        st.header("👤 Conta")

        # sistema de mensagens flash para feedbacks
        flash = st.session_state.pop("flash", None)
        if flash:
            st.success(flash["msg"]) if flash["type"] == "success" else st.warning(flash["msg"])

        # exibe o nome do usuário logado, não o ID
        current_user_id = st.session_state.get("current_user")
        if current_user_id and not users_map.empty:
            current_username = users_map.loc[users_map["user_id"] == int(current_user_id), "username"].iloc[0]
            st.caption(f"Logado como: **{current_username}**")
        else:
            st.caption("Logado como: **—**")
        st.divider()

        # --- LOGIN ---
        usernames = users_map["username"].tolist()
        try:
            # encontra o índice do username atual para ser o padrão do selectbox
            default_idx: int = usernames.index(current_username) if current_user_id else 0
        except (ValueError, NameError):
            default_idx = 0

        selected_username = st.selectbox("Selecionar usuário", options=usernames, index=default_idx)

        if st.button("Entrar", use_container_width=True):
            # mapeia o username selecionado de volta para o seu user_id
            user_id_to_login = users_map.loc[users_map["username"] == selected_username, "user_id"].iloc[0]
            st.session_state.current_user = str(user_id_to_login)
            st.toast(f"Logado como {selected_username}")
            st.rerun()
        st.divider()

        # --- CADASTRO ---
        new_username: str = st.text_input("Cadastrar novo usuário", placeholder="ex.: Joao Silva")
        if st.button("Cadastrar", use_container_width=True):
            username: str = new_username.strip()
            if not username:
                st.warning("Informe um nome de usuário.")
            else:
                try:
                    # envia o 'username' para o backend
                    payload: dict[str, str] = {"username": username}
                    r: requests.Response = requests.post(f"{os.environ.get("BACKEND_URL")}/v1/users/signup", json=payload, timeout=20)
                    r.raise_for_status()
                    response_data = r.json()

                    if not response_data.get("created", True):
                        st.warning(f"O nome de usuário **{username}** já existe.")
                    else:
                        # pega o novo user_id retornado pelo backend
                        new_user_id = response_data.get("user_id")
                        st.session_state["current_user"] = new_user_id
                        st.session_state["flash"] = {"type": "success", "msg": f"Usuário **{username}** cadastrado!"}
                        load_data.clear() # limpa o cache para recarregar a lista de usuários
                        st.rerun()
                except requests.RequestException as e:
                    st.error(f"Falha ao cadastrar: {e}")


def main():
    """Orquestra a execução do aplicativo."""
    state.init()
    user_df, books_df = load_data()

    # cria um DataFrame de mapeamento: user_id <-> username
    # isso simplifica as buscas na interface
    users_map: DataFrame = user_df[["user_id", "username"]].drop_duplicates().sort_values("username")

    sidebar(users_map)

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
