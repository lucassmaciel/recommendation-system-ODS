import pandas as pd
import numpy as np
from backend.models.schemas import RecoRequest
from backend.services import recommender_user
from backend.core.config import settings

def evaluate_accuracy(df, recommender_func, 
                      sim_metric="cosine", top_n=20, like_threshold=5, 
                      test_fraction=0.2, k_neighbors=150, min_overlap=1, 
                      shrink_cos=2, shrink_pear=1, verbose=False):
    """
    Avaliação da acurácia (Precision@K) usando user-based + item-based.
    """
    all_hits = 0
    all_recs = 0
    evaluated_users = 0

    np.random.seed(42)  # Reprodutibilidade

    for user_id in df.index:
        user_ratings = df.loc[user_id]
        liked_books = user_ratings[user_ratings >= like_threshold].index.tolist()

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

        all_hits += hits
        all_recs += num_recs
        evaluated_users += 1

        if verbose:
            print(f"Usuário {user_id}: Hits={hits}, Recomendações={num_recs}, Accuracy={hits / num_recs:.4f}")

    mean_accuracy = all_hits / all_recs if all_recs > 0 else 0.0

    print(f"\n--- Estatísticas Finais ---")
    print(f"Usuários Avaliados: {evaluated_users}")
    print(f"Média Precision@{top_n}: {mean_accuracy:.4f}")
    print(f"Total de Acertos: {all_hits}, Total de Recomendações: {all_recs}")

    return mean_accuracy


def main():
    df = pd.read_csv(settings.DATA_PATH, index_col=0)
    df.index = df.index.astype(str)

    TOP_N = 20
    LIKE_THRESHOLD = 5
    TEST_FRACTION = 0.2

    print("\n=== Avaliação de Acurácia Otimizada ===")
    for metric in ["cosine", "pearson"]:
        accuracy = evaluate_accuracy(
            df,
            recommender_user.recommend_user_based_better,
            sim_metric=metric,
            top_n=TOP_N,
            like_threshold=LIKE_THRESHOLD,
            test_fraction=TEST_FRACTION,
            k_neighbors=150,
            min_overlap=1,
            shrink_cos=2,
            shrink_pear=1,
            verbose=False
        )
        print(f"{metric.capitalize()} -> Precision@{TOP_N}: {accuracy:.4f}")

if __name__ == "__main__":
    main()
