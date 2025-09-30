import os

import numpy as np
import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from data import load_data
from dotenv import load_dotenv

load_dotenv()


def search_books_in_csv(query: str) -> str:
    _, df = load_data()
    # normalização, garante que todas as colunas serão lidas como string
    search_cols: list[str] = ["book", "author", "year", "publisher", "user_id", "rating"]

    for col in search_cols:
        df[col] = df[col].astype(str)

    mask = np.logical_or.reduce([
        df[col].str.contains(query, case=False, na=False) for col in search_cols
    ])

    results_df = df[mask]

    if results_df.empty:
        return "Nenhum livro encontrado com essa busca."

    return results_df.to_json(orient="records")


@st.cache_resource
def setup_agent():
    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("groq", {}).get(
        "api_key"
    )

    agent = Agent(
        name="book_agent",
        role="Você é um bibliotecário especialista em livros. Use suas ferramentas para encontrar informações e dar recomendações. Responda sempre em português.",
        model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
        tools=[search_books_in_csv],
        markdown=True,
    )
    return agent


def render():
    st.subheader("🤖 Converse com o Agente Bibliotecário")

    book_agent = setup_agent()
    if not book_agent:
        st.error(
            "Chave da API do Groq não configurada. Verifique seu arquivo .env ou os segredos do Streamlit."
        )
        return

    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_input(
            "Peça uma recomendação ou informação...",
            placeholder="Ex: Me fale sobre os livros de J.R.R. Tolkien",
            label_visibility="collapsed",
        )
        submit_button = st.form_submit_button("Enviar Pergunta")

    if submit_button and prompt:
        # adiciona a mensagem do usuário à memória do chat
        st.session_state.agent_messages.append({"role": "user", "content": prompt})

        # gera a resposta do agente
        with st.spinner("O agente está pensando..."):
            run_output = book_agent.run(prompt)
            response_text = run_output.content

        # adiciona a resposta do agente a memória
        st.session_state.agent_messages.append(
            {"role": "assistant", "content": response_text}
        )

    st.divider()

    # inicializa a memória do chat se ainda não existir
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    # exibe da mais nova para a mais antiga
    for message in reversed(st.session_state.agent_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
