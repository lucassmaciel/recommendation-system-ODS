from backend.models.schemas import RecoRequest
from backend.services import recommender_user
from backend.core.config import settings
import pandas as pd
import numpy as np

def ndcg_at_k(recommended, relevant, k=20):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0



def evaluate_recommender_stable_general(df, recommender_func, sim_metric="cosine", k=20, like_threshold=7, test_fraction=0.3):
    accuracies, recalls, ndcgs = [], [], []

    for user_id in df.index:
        user_ratings = df.loc[user_id]
        rated_books = user_ratings[user_ratings >= like_threshold].index.tolist()

        if len(rated_books) < 20:
            continue

        n_test = max(1, int(len(rated_books) * test_fraction))
        test_books = rated_books[-n_test:]  # últimos n_test itens
        train_books = [b for b in rated_books if b not in test_books]

        df_train = df.copy()
        df_train.loc[user_id, test_books] = 0

        req = RecoRequest(
            user_id=str(user_id),
            data_path=str(settings.DATA_PATH),
            top_n=k,
            like_threshold=like_threshold
        )

        res = recommender_func(req, df=df_train, sim_metric=sim_metric)
        recommended_books = [r.book for r in res.recommendations]

        relevant = set(test_books)
        hits = len(set(recommended_books) & relevant)

        accuracies.append(hits / k)
        recalls.append(hits / len(relevant))
        ndcgs.append(ndcg_at_k(recommended_books, relevant, k))

    return np.mean(accuracies), np.mean(recalls), np.mean(ndcgs)


def main():
    df = pd.read_csv(settings.DATA_PATH, index_col=0)

    user_id = "23872"  # usuário fixo para teste
    print(f"Usuário selecionado para teste: {user_id}")

    req = RecoRequest(
        user_id=user_id,
        data_path=str(settings.DATA_PATH),
        top_n=5,
        like_threshold=7
    )

    # Testar recomendação normal
    for metric in ["cosine", "pearson"]:
        res = recommender_user.recommend_user_based_weighted_100(req, df=df, sim_metric=metric)
        print(f"\n=== {metric.capitalize()} ===")
        for r in res.recommendations:
            print(r.book, r.score)

    print("\n=== Avaliação de Acurácia (usuários com >=20 avaliações, holdout 20%) ===")
    for metric in ["cosine", "pearson"]:
        p, r, n = evaluate_recommender_stable_general(
            df,
            recommender_user.recommend_user_based_weighted_100,  # NOVO ALGORITMO
            sim_metric=metric,
            k=20
        )
        print(f"{metric.capitalize()} -> Accuracy@20: {p:.4f}, Recall@20: {r:.4f}, NDCG@20: {n:.4f}")

    for metric in ["cosine", "pearson"]:
        p, r, n = evaluate_recommender_stable_general(
            df,
            recommender_user.recommend_user_based_weighted_optimized,
            sim_metric=metric,
            k=20
        )
        print(f"{metric.capitalize()} -> Accuracy@20: {p:.4f}, Recall@20: {r:.4f}, NDCG@20: {n:.4f}")


if __name__ == "__main__":
    main()
