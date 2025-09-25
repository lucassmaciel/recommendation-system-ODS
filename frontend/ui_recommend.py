from __future__ import annotations
import pandas as pd
import streamlit as st
from pandas import DataFrame
from streamlit_star_rating import st_star_rating

def render(user_df: DataFrame, books_df: DataFrame):
    st.subheader("Recomendações")
    st.caption("Este app usa filtragem colaborativa user-based.")
    user_id = st.selectbox("Selecione o id do usuário:", user_df["user_id"].unique())
    book_to_rate = st.selectbox("Selecione um livro para avaliar:", books_df["book"].unique())
    rating: int = st_star_rating(label="Avalie", maxValue=10, defaultValue=5, key="star_rating")
    if st.button("Confirmar"):
        st.success(f"Avaliação confirmada: nota {rating} para **{book_to_rate}** (usuário {user_id})")