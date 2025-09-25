import numpy as np
import pandas as pd
from models.schemas import RecoItem, RecoResponse

def cosine_similarity(vec1, vec2):
    mask = (vec1 > 0) & (vec2 > 0)
    if mask.sum() == 0:
        return 0.0
    v1, v2 = vec1[mask], vec2[mask]
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom if denom > 0 else 0.0

def pearson_similarity(vec1, vec2):
    mask = (vec1 > 0) & (vec2 > 0)
    if mask.sum() < 2:  # exige pelo menos 2 avaliações em comum
        return 0.0
    v1, v2 = vec1[mask], vec2[mask]
    v1, v2 = v1 - v1.mean(), v2 - v2.mean()
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.dot(v1, v2) / denom if denom > 0 else 0.0

def recommend_user_based(req, sim_metric="cosine", alpha=0.9, df=None):
    if df is None:
        df = pd.read_csv(req.data_path, index_col=0)

    # Preenchimento de valores faltantes (NaN) por 0
    df = df.fillna(0)

    user_id = int(req.user_id)
    if user_id not in df.index:
        return RecoResponse(user_id=req.user_id, recommendations=[])

    target_ratings = df.loc[user_id]
    liked_books = target_ratings[target_ratings >= req.like_threshold].index.tolist()
    if not liked_books:
        return RecoResponse(user_id=req.user_id, recommendations=[])

    # calcula similaridade do usuário alvo com todos os outros usuários
    sims = {}
    for other_id in df.index:
        if other_id == user_id:
            continue
        other_ratings = df.loc[other_id]

        if sim_metric == "cosine":
            sim = cosine_similarity(target_ratings.values, other_ratings.values)
        elif sim_metric == "pearson":
            sim = pearson_similarity(target_ratings.values, other_ratings.values)
        elif sim_metric == "hybrid":
            sim_cos = cosine_similarity(target_ratings.values, other_ratings.values)
            sim_pear = pearson_similarity(target_ratings.values, other_ratings.values)

            # Ajuste dinâmico: se poucas avaliações em comum, confiar mais no Cosine
            mask = (target_ratings.values > 0) & (other_ratings.values > 0)
            if mask.sum() < 3:
                alpha_adj = 0.9  # prioriza Cosine quando poucos itens em comum
            else:
                alpha_adj = alpha
            sim = alpha_adj * sim_cos + (1 - alpha_adj) * sim_pear
        else:
            raise ValueError(f"Similarity {sim_metric} não suportada")

        if sim > 0:
            sims[other_id] = sim

    if not sims:
        return RecoResponse(user_id=req.user_id, recommendations=[])

    scores = {}
    sim_sums = {}
    for other_id, sim in sims.items():
        for book, rating in df.loc[other_id].items():
            if rating >= req.like_threshold and book not in liked_books:
                scores[book] = scores.get(book, 0) + sim * rating
                sim_sums[book] = sim_sums.get(book, 0) + sim

    # Normalização dos scores finais para evitar vieses de usuários com mais avaliações
    recommendations_list = []
    for book in scores:
        score = scores[book] / (sim_sums[book] + 1e-9)
        recommendations_list.append((book, score))

    # Top N
    top_books = sorted(recommendations_list, key=lambda x: x[1], reverse=True)[:req.top_n]
    recommendations = [RecoItem(book=book, score=score, reason=sim_metric) for book, score in top_books]

    return RecoResponse(user_id=req.user_id, recommendations=recommendations)
