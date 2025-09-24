from pathlib import Path

import pandas as pd
import streamlit as st
from pandas import DataFrame

st.html("""
<style>
    /* Aplica a todos os elementos de imagem no app */
    img {
        /* Define uma altura fixa para todas as imagens */
        height: 300px !important;
        /* Garante que a imagem cubra o espaço sem distorcer */
        object-fit: cover !important;
    }
</style>
""")

st.set_page_config(
    page_title="Sistema de Recomendação",

    layout="wide"
)

@st.cache_data
def load_data() -> tuple[DataFrame, DataFrame]:
    user_path = Path("processed-data/final_user_df.csv")
    user_df: DataFrame = pd.read_csv(user_path)
    books_path = Path("processed-data/books_info.csv")
    books_df: DataFrame = pd.read_csv(books_path)
    return user_df, books_df

user_df, books_df = load_data()

def save_rating_to_csv(df: DataFrame, path: Path):
    df.to_csv(path, index=False)

def show_catalog():
    st.title("Catálogo de Livros")

    cols = st.columns(3)

    for i, (_, row) in enumerate(books_df.iterrows()):
        with cols[i % 3], st.container(border=True):
            img_col, text_col = st.columns([1, 1])

            with img_col:
                st.image(row["image"], use_container_width=True)

            with text_col:
                st.markdown(f"**{row['book']}**")
                st.caption(f"Autor: {row['author']}")
                st.caption(f"Ano: {int(row['year'])}")

def recommend_app():
    st.write("""
         # :rainbow[Sistema de recomendação de livros]
         *Esse sistema utiliza filtragem colaborativa e correlação de Pearlson para calcular a distância entre os itens*.
         """)

    user_id = st.selectbox("Selecione o id do usuário:", user_df["user_id"].unique())
    book_to_rate = st.selectbox("Selecione um livro para avaliar:", books_df["book"].unique())
    rating: int = st.slider("Avaliação (1-10):", 1, 10)

    new_rating_df = pd.DataFrame([{
        "user_id": user_id,
        "book": book_to_rate,
        "rating": rating
    }])

    # updated_user_df = pd.concat([user_df, new_rating_df], ignore_index=True)
    # user_path = Path("processed-data/final_user_df.csv")
    # save_rating_to_csv(updated_user_df, user_path)

    # st.success(f"Avaliação de '{book_to_rate}' por Usuário {user_id} adicionada!")
    # st.cache_data.clear()
    # st.rerun()

if __name__ == "__main__":
    recommend_app()
    show_catalog()
