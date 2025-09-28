import pandas as pd
import numpy as np
from backend.models.schemas import RecoRequest
from backend.services import recommender_user
from backend.core.config import settings


def evaluate_accuracy_single_user(df, recommender_func, user_id, 
                                  sim_metric="cosine", top_n=20, 
                                  like_threshold=5, test_fraction=0.2):
    """
    Calcula a acurácia de recomendação para UM usuário específico.
    """
    # 1. Pegar avaliações do usuário
    user_ratings = df.loc[user_id]
    liked_books = user_ratings[user_ratings >= like_threshold].index.tolist()

    if len(liked_books) < 2:
        print(f"Usuário {user_id} não tem itens suficientes para avaliar.")
        return

    # 2. Dividir em treino (parte 1) e teste (parte 2 - gabarito)
    n_test = max(1, int(len(liked_books) * test_fraction))
    test_books = list(np.random.choice(liked_books, n_test, replace=False))

    df_train = df.copy()
    df_train.loc[user_id, test_books] = 0.0  # zera os itens de teste

    # 3. Gerar recomendações usando apenas a parte 1
    req = RecoRequest(
        user_id=str(user_id),
        data_path="",
        top_n=top_n,
        like_threshold=like_threshold
    )

    res = recommender_func(req, df=df_train, sim_metric=sim_metric)
    recommended_books = [r.book for r in res.recommendations]

    # 4. Comparar com o gabarito
    hits = len(set(recommended_books) & set(test_books))
    total_recs = len(recommended_books)

    # 5. Calcular acurácia
    accuracy = hits / total_recs if total_recs > 0 else 0.0

    # 6. Mostrar relatório
    print(f"\n=== Avaliação Usuário {user_id} ({sim_metric}) ===")
    print(f"Itens de teste (gabarito): {test_books}")
    print(f"Recomendações: {recommended_books}")
    print(f"Acertos: {hits} de {total_recs}")
    print(f"Acurácia: {accuracy:.2%}")

    return accuracy

def main():
    df = pd.read_csv(settings.DATA_PATH, index_col=0)
    df.index = df.index.astype(str)
    user_id = input("Digite o ID do usuário para avaliação: ")

    USER_ID = user_id
    TOP_N = 20

    for metric in ["cosine", "pearson"]:
        evaluate_accuracy_single_user(
            df,
            recommender_user.recommend_user_based_better,
            user_id=USER_ID,
            sim_metric=metric,
            top_n=TOP_N,
            like_threshold=5,
            test_fraction=0.5
        )


if __name__ == "__main__":
    main()
