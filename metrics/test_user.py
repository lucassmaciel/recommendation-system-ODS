import numpy as np
import pandas as pd

from backend.core.config import settings
from backend.models.schemas import RecoRequest
from backend.services import recommender_user


def evaluate_accuracy_single_user(  # noqa: PLR0913
    df,
    recommender_func,
    user_id,
    sim_metric,
    top_n,
    like_threshold,
    test_fraction,
):
    """Calcula a acurácia de recomendação para UM usuário específico."""
    # pegar avaliações do usuário
    user_ratings = df.loc[user_id]
    liked_books = user_ratings[user_ratings >= like_threshold].index.tolist()

    min_books = 2
    if len(liked_books) < min_books:
        print(
            f"Usuário {user_id} não tem itens suficientes para avaliar (com nota >= {like_threshold})."
        )
        return None  # retorna None se não for possível avaliar

    # dividir em treino (parte 1) e teste (parte 2 - gabarito)
    n_test: int = max(1, int(len(liked_books) * test_fraction))
    test_books = list(np.random.choice(liked_books, n_test, replace=False))  # noqa: NPY002

    df_train = df.copy()
    df_train.loc[user_id, test_books] = 0.0  # zera os itens de teste

    # gerar recomendações usando apenas a parte 1
    req = RecoRequest(
        user_id=str(user_id), k_neighbors=20, top_n=top_n, like_threshold=like_threshold
    )

    res = recommender_func(req, df=df_train, sim_metric=sim_metric)
    recommended_books = [r.book for r in res.recommendations]

    # comparar com o gabarito
    hits: int = len(set(recommended_books) & set(test_books))
    total_recs: int = len(recommended_books)

    # calcular acurácia
    accuracy = hits / total_recs if total_recs > 0 else 0.0

    # mostrar relatório
    print(f"\n=== Avaliação Usuário {user_id} ({sim_metric}) ===\n")
    print(f"Itens de teste (gabarito): {test_books}\n")
    print(f"Recomendações: {recommended_books}\n")
    print(f"Acertos: {hits} de {total_recs}")
    print(f"Acurácia: {accuracy:.2%}")

    return accuracy


def main():
    df = pd.read_csv(settings.DATA_PATH, index_col=0)

    df.index = df.index.astype(str)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df: pd.DataFrame = df.fillna(0.0)

    user_id: str = input("Digite o ID do usuário para avaliação: ").strip()

    if user_id not in df.index:
        print(f"ERRO: Usuário com ID '{user_id}' não encontrado no DataFrame.")
        return

    for metric in ["cosine", "pearson"]:
        evaluate_accuracy_single_user(
            df,
            recommender_user.recommend_user_based_better,
            user_id=user_id,
            sim_metric=metric,
            top_n=20,
            like_threshold=7,
            test_fraction=0.5,
        )


if __name__ == "__main__":
    main()
