from math import sqrt
from pathlib import Path

import pandas as pd
import streamlit as st
from pandas import DataFrame

user_path = Path("processed-data/user_based_df.csv")
user_df: DataFrame = pd.read_csv(user_path)
books_path = Path("processed-data/item_based_df.csv")
books_df: DataFrame = pd.read_csv(books_path)

def save_new_user(user_id, book, rating):
    new_user_data ={
        "user_id": user_id,
        f"{book}": str(book),
        "rating": int(rating),
    }

def pearson(rating1, rating2):
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for key in rating1:
        if key in rating2:
            n += 1
            x = rating1[key]
            y = rating2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += pow(x, 2)
            sum_y2 += pow(y, 2)
    denominator = sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n)
    if denominator == 0:
        return 0
    return (sum_xy - (sum_x * sum_y) / n) / denominator

def recommend_app():
    st.write("""
         # :rainbow[Sistema de recomendação de livros]
         *Esse sistema utiliza filtragem colaborativa e correlação de Pearlson para calcular a distância entre os itens*.
         """)
    users_ratings: DataFrame = user_df
    books: DataFrame = books_df

    if not users_ratings.empty:
        user_id = st.selectbox("Selecione o id do usuário:", users_ratings["user_id"].unique())
        book_to_rate = st.selectbox("Selecione um livro para avaliar:", books["book"].unique())
        rating: int = st.slider("Avaliação (1-10):", 1, 10)

        if st.button("Avaliar Livro"):
            new_rating = pd.DataFrame({"user_id": user_id, book_to_rate: book_to_rate}, user_id[book_to_rate: rating])
            users_ratings = pd.concat([users_ratings, new_rating], ignore_index=True)
            save_new_user(user_id, book_to_rate, rating)
            st.write(f"Avaliação para {book_to_rate} adicionada com sucesso!")

if __name__ == "__main__":
    recommend_app()
