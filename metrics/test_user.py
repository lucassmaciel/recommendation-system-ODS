import pandas as pd
import numpy as np
from backend.models.schemas import RecoRequest
from backend.services import recommender_user
from backend.core.config import settings

def _ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, b in enumerate(recommended[:k]):
        if b in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_accuracy(df, recommender_func, 
                      sim_metric="cosine", top_n=20, like_threshold=5, 
                      test_fraction=0.2, k_neighbors=150, min_overlap=1, 
                      shrink_cos=2, shrink_pear=1, verbose=False):
    """
    Avaliação da acurácia (Precision@K), Recall@K e NDCG@K usando user-based + item-based.
    """

    all_hits = 0
    all_recs = 0
    evaluated_users = 0

    all_recall = 0.0
    all_ndcg_10 = 0.0
    all_ndcg_20 = 0.0

    np.random.seed(42)  # Reprodutibilidade

    for user_id in df.index:
        user_ratings = df.loc[user_id]
        liked_books = user_ratings[user_ratings >= like_threshold].index.tolist()

        # --- FILTRO: ignora usuários com poucos ratings ---
        if len(user_ratings[user_ratings > 0]) < 3:
            continue
        if len(liked_books) < 2:
            continue

        n_test = max(1, int(len(liked_books) * test_fraction))
        test_books = list(np.random.choice(liked_books, n_test, replace=False))

        # DF de treino: remove os itens de teste
        df_train = df.copy()
        df_train.loc[user_id, test_books] = 0.0
        
        # Preparar requisição para a função de recomendação
        req = RecoRequest(
            user_id=str(user_id),
            data_path="",  # o df já está sendo passado
            top_n=top_n,
            like_threshold=like_threshold
        )

        # Obter recomendações
        res = recommender_func(
            req, 
            df=df_train, 
            sim_metric=sim_metric,
            k_neighbors=k_neighbors, 
            min_overlap=min_overlap,
            shrink_cos=shrink_cos,
            shrink_pear=shrink_pear
        )

        recommended_books = [r.book for r in res.recommendations]

        # Hits comparando com a parte de teste
        hits = len(set(recommended_books) & set(test_books))
        num_recs = len(recommended_books)
        if num_recs == 0:
            continue

        # Atualiza métricas
        all_hits += hits
        all_recs += num_recs
        evaluated_users += 1

        all_recall += hits / len(test_books)
        all_ndcg_10 += _ndcg_at_k(recommended_books, set(test_books), 10)
        all_ndcg_20 += _ndcg_at_k(recommended_books, set(test_books), 20)

        if verbose:
            precision = hits / num_recs
            recall = hits / len(test_books)
            print(f"Usuário {user_id}: Precision={precision:.4f}, Recall={recall:.4f}, Hits={hits}, Recs={num_recs}")

    mean_precision = all_hits / all_recs if all_recs > 0 else 0.0
    mean_recall = all_recall / evaluated_users if evaluated_users > 0 else 0.0
    mean_ndcg_10 = all_ndcg_10 / evaluated_users if evaluated_users > 0 else 0.0
    mean_ndcg_20 = all_ndcg_20 / evaluated_users if evaluated_users > 0 else 0.0

    print(f"\n--- Estatísticas Finais ---")
    print(f"Usuários Avaliados: {evaluated_users}")
    print(f"Média Precision@{top_n}: {mean_precision:.4f}")
    print(f"Média Recall@{top_n}: {mean_recall:.4f}")
    print(f"NDCG@10: {mean_ndcg_10:.4f}, NDCG@20: {mean_ndcg_20:.4f}")
    print(f"Total de Acertos: {all_hits}, Total de Recomendações: {all_recs}")

    return mean_precision, mean_recall, mean_ndcg_10, mean_ndcg_20


def main():
    df = pd.read_csv(settings.DATA_PATH, index_col=0)
    df.index = df.index.astype(str)

    TOP_N = 50
    LIKE_THRESHOLD = 5
    TEST_FRACTION = 0.2

    print("\n=== Avaliação de Acurácia Otimizada ===")
    for metric in ["cosine", "pearson"]:
        precision, recall, ndcg10, ndcg20 = evaluate_accuracy(
            df,
            recommender_user.recommend_user_based_better,
            sim_metric=metric,
            top_n=TOP_N,
            like_threshold=LIKE_THRESHOLD,
            test_fraction=TEST_FRACTION,
            k_neighbors=200,
            min_overlap=1,
            shrink_cos=1.5,
            shrink_pear=0.5,
            verbose=False
        )
        print(f"{metric.capitalize()} -> Precision@{TOP_N}: {precision:.4f}, Recall@{TOP_N}: {recall:.4f}, NDCG@10: {ndcg10:.4f}, NDCG@20: {ndcg20:.4f}")


if __name__ == "__main__":
    main()
