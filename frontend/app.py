import streamlit as st
import pandas as pd
from math import sqrt
from collections import Counter

st.set_page_config(page_title="kNN (Votação Majoritária) • Recomendação", layout="wide")

# ==============================
# Utilidades
# ==============================
def load_ratings(csv_path: str = "dataset.csv") -> pd.DataFrame:
    """Carrega o CSV com colunas Username, Game, Rating."""
    try:
        df = pd.read_csv(csv_path)
        expected = {"Username", "Game", "Rating"}
        if not expected.issubset(df.columns):
            st.error("O CSV deve conter as colunas: Username, Game, Rating")
            return pd.DataFrame(columns=["Username", "Game", "Rating"])
        return df
    except FileNotFoundError:
        st.warning("dataset.csv não encontrado no diretório atual. Envie um arquivo CSV abaixo.")
        return pd.DataFrame(columns=["Username", "Game", "Rating"])


def cosine_on_overlap(a: dict, b: dict) -> float:
    """Similaridade do cosseno considerando apenas itens em comum; 0.0 se não houver interseção."""
    common = set(a).intersection(b)
    if not common:
        return 0.0
    num = sum(a[i] * b[i] for i in common)
    den_a = sqrt(sum(a[i] * a[i] for i in common))
    den_b = sqrt(sum(b[i] * b[i] for i in common))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def to_user_ratings(df: pd.DataFrame) -> dict:
    """Converte para {user: {game: rating}}"""
    users = {}
    for row in df.itertuples(index=False):
        users.setdefault(row.Username, {})[row.Game] = float(row.Rating)
    return users


def neighbors_who_rated(users_dict: dict, username: str, game: str) -> list:
    """Retorna [(sim, vizinho, rating_no_game), ...] para quem avaliou o game."""
    target = users_dict.get(username, {})
    triples = []
    for other, ratings in users_dict.items():
        if other == username:
            continue
        if game in ratings:
            sim = cosine_on_overlap(target, ratings)
            if sim > 0.0:
                triples.append((sim, other, ratings[game]))
    triples.sort(reverse=True, key=lambda x: x[0])
    return triples


def majority_vote(votes: list[int], tie_break: str = "like") -> int | None:
    """Retorna 1 (curtir), 0 (não curtir) ou None (sem votos). tie_break: 'like' ou 'dislike'."""
    if not votes:
        return None
    c = Counter(votes)
    if c[1] == c[0]:
        return 1 if tie_break == "like" else 0
    return 1 if c[1] > c[0] else 0


# ==============================
# Carregamento de dados
# ==============================
st.title("🔍 kNN com Votação Majoritária (Filtragem Colaborativa)")
st.caption("Escolha um usuário e um jogo não avaliado por ele; o app usa os K vizinhos mais próximos que avaliaram esse jogo para votar se o usuário vai **curtir** (1) ou **não curtir** (0).")

# Fonte de dados: dataset.csv local ou upload
left, right = st.columns([1, 1])
with left:
    st.subheader("📁 Dataset")
    df_local = load_ratings("dataset.csv")

with right:
    uploaded = st.file_uploader("Ou envie um CSV (Username, Game, Rating)", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        df_uploaded = pd.read_csv(uploaded)
        expected = {"Username", "Game", "Rating"}
        if expected.issubset(df_uploaded.columns):
            df = df_uploaded.copy()
            st.success("Usando o CSV enviado.")
        else:
            st.error("O CSV enviado não possui as colunas exigidas. Usando dataset local (se existir).")
            df = df_local.copy()
    else:
        df = df_local.copy()

if df.empty:
    st.info("Carregue um CSV válido para continuar.")
    st.stop()

# Mostrar amostra
with st.expander("👀 Prévia do dataset"):
    st.dataframe(df.head(20))

users_dict = to_user_ratings(df)

# ==============================
# Painel de parâmetros
# ==============================
st.subheader("⚙️ Parâmetros")
col1, col2, col3, col4 = st.columns([1.2, 1.2, 1, 1])

with col1:
    username = st.selectbox("Usuário alvo", sorted(users_dict.keys()))

with col2:
    # jogos não avaliados pelo usuário
    rated_games = set(users_dict.get(username, {}).keys())
    all_games = set(df["Game"].dropna().unique().tolist())
    candidate_games = sorted(list(all_games - rated_games))
    if not candidate_games:
        st.warning("Este usuário já avaliou todos os jogos do dataset. Escolha outro usuário.")
        st.stop()
    game = st.selectbox("Jogo alvo (não avaliado pelo usuário)", candidate_games)

with col3:
    k = st.slider("K vizinhos", min_value=1, max_value=10, value=3, step=1)

with col4:
    thr = st.slider("Limiar 'curtir' (≥)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)

col5, col6 = st.columns([1,1])
with col5:
    tie_break = st.radio("Desempate", options=["like", "dislike"], index=0, horizontal=True, help="Critério em caso de empate na votação")
with col6:
    run_btn = st.button("Executar previsão", type="primary")

# ==============================
# Execução
# ==============================
if run_btn:
    triples = neighbors_who_rated(users_dict, username, game)  # [(sim, vizinho, rating), ...]
    if not triples:
        st.error("Sem vizinhos válidos (ninguém com similaridade > 0 avaliou esse jogo).")
        st.stop()

    top = triples[:k]

    # Monta dataframe para visualização
    viz_df = pd.DataFrame([
        {"Vizinho": nb, "Similaridade": sim, "Rating no jogo": r, "Voto (1=curtiu)": 1 if r >= thr else 0}
        for (sim, nb, r) in top
    ])
    st.subheader("👥 Vizinhos selecionados")
    st.dataframe(viz_df, use_container_width=True)

    votes = viz_df["Voto (1=curtiu)"].astype(int).tolist()
    result = majority_vote(votes, tie_break=tie_break)

    st.subheader("🗳️ Resultado da votação")
    like_count = sum(votes)
    dislike_count = len(votes) - like_count
    st.write(f"**Votos (curtir=1 / não curtir=0):** {votes}")
    st.write(f"**Contagem:** curtir={like_count} • não curtir={dislike_count}")
    if result is None:
        st.warning("Sem votos.")
    else:
        st.success(f"**Previsão para ({username}, {game}) com k={len(top)}:** {'curtir (1)' if result==1 else 'não curtir (0)'}")

    with st.expander("🔎 Detalhes da Similaridade"):
        st.caption("A similaridade do cosseno é calculada apenas nos jogos em comum entre o usuário-alvo e cada vizinho.")
        st.code(
            """
def cosine_on_overlap(a: dict, b: dict) -> float:
    common = set(a).intersection(b)
    if not common:
        return 0.0
    num = sum(a[i] * b[i] for i in common)
    den_a = sqrt(sum(a[i] * a[i] for i in common))
    den_b = sqrt(sum(b[i] * b[i] for i in common))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)
            """
        )

# Rodapé
st.caption("© kNN (majority vote) — exemplo educacional")