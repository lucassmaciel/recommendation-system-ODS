from models.schemas import RecoRequest
from services import recommender_user
from backend.core.config import settings
import pandas as pd
import numpy as np

def ndcg_at_k(recommended, relevant, k=50):
    """Calcula NDCG@k para um usuário"""
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_user(df, recommender_func, user_id, sim_metric="cosine", k=50, like_threshold=7, test_fraction=0.3):
    """
    Avalia métricas para um único usuário de forma confiável.
    """
    user_ratings = df.loc[user_id]
    rated_books = user_ratings[user_ratings >= like_threshold].index.tolist()
    if len(rated_books) < 2:
        return 0.0, 0.0, 0.0  # não dá para avaliar

    n_test = max(1, int(len(rated_books) * test_fraction))
    test_books = rated_books[-n_test:]
    df_train = df.copy()
    df_train.loc[user_id, test_books] = 0

    req = RecoRequest(
        user_id=str(user_id),
        data_path=None,
        top_n=k,
        like_threshold=like_threshold
    )
    res = recommender_func(req, df=df_train, sim_metric=sim_metric)
    recommended_books = [r.book for r in res.recommendations]

    relevant = set(test_books)
    hits = len(set(recommended_books) & relevant)
    precision = hits / k
    recall = hits / len(relevant)
    ndcg = ndcg_at_k(recommended_books, relevant, k)

    return precision, recall, ndcg


def evaluate_recommender_stable_general(df, recommender_func, sim_metric="cosine", k=50, like_threshold=7, test_fraction=0.3):
    """
    Avalia de forma estável e confiável um recommender.
    - Usa uma fração dos itens de cada usuário como teste (holdout)
    - Mantém treino e teste separados
    - Calcula Precision@k, Recall@k e NDCG@k
    """
    precisions, recalls, ndcgs = [], [], []

    for user_id in df.index:
        user_ratings = df.loc[user_id]
        rated_books = user_ratings[user_ratings >= like_threshold].index.tolist()

        if len(rated_books) < 50:
            continue

        # Define items de teste de forma determinística
        n_test = max(1, int(len(rated_books) * test_fraction))
        test_books = rated_books[-n_test:]  # últimos n_test itens
        train_books = [b for b in rated_books if b not in test_books]

        # Cria df de treino sem os itens de teste
        df_train = df.copy()
        df_train.loc[user_id, test_books] = 0

        req = RecoRequest(
            user_id=str(user_id),
            data_path=str(settings.DATA_PATH),  # passa caminho válido
            top_n=k,
            like_threshold=like_threshold
        )

        # Recomendações com treino
        res = recommender_func(req, df=df_train, sim_metric=sim_metric)
        recommended_books = [r.book for r in res.recommendations]

        relevant = set(test_books)
        hits = len(set(recommended_books) & relevant)

        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
        ndcgs.append(ndcg_at_k(recommended_books, relevant, k))

    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

def main():
    df = pd.read_csv(settings.DATA_PATH, index_col=0)

    # Usuário específico para exibir recomendações
    user_id = "261829"  # usuário fixo para teste
    print(f"Usuário selecionado para teste: {user_id}")

    req = RecoRequest(
        user_id=user_id,
        data_path=str(settings.DATA_PATH),
        top_n=5,
        like_threshold=7
    )

    # Recomendação para cada métrica
    for metric in ["cosine", "pearson", "hybrid"]:
        res = recommender_user.recommend_user_based(req, df=df, sim_metric=metric)
        print(f"\n=== {metric.capitalize()} ===")
        for r in res.recommendations:
            print(r.book, r.score)

    # Avaliação de métricas de forma estável e confiável
    print("\n=== Avaliação de Acurácia (usuários com >=50 avaliações, holdout 50%) ===")
    for metric in ["cosine", "pearson", "hybrid"]:
        p, r, n = evaluate_recommender_stable_general(df, recommender_user.recommend_user_based, sim_metric=metric, k=50)
        print(f"{metric.capitalize()} -> Precision@50: {p:.4f}, Recall@50: {r:.4f}, NDCG@50: {n:.4f}")


if __name__ == "__main__":
    main()
