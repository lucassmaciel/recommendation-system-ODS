import numpy as np
import pandas as pd
from ..models.schemas import RecoItem, RecoResponse

def ndcg_score(recommended_books, relevant_books, k):
    dcg = 0.0
    for i, book in enumerate(recommended_books[:k]):
        if book in relevant_books:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_books), k)))
    return dcg / idcg if idcg > 0 else 0.0

def cosine_similarity(vec1, vec2, shrinkage_cos=10):
    mask = (vec1 > 0) & (vec2 > 0)
    if mask.sum() == 0:
        return 0.0
    v1, v2 = vec1[mask], vec2[mask]

    v1 = v1 - v1.mean()
    v2 = v2 - v2.mean()

    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    base_sim = num / denom if denom > 0 else 0.0

    return base_sim * (mask.sum() / (mask.sum() + shrinkage_cos))

def pearson_similarity(vec1, vec2, shrinkage_pearson=5):
    mask = (vec1 > 0) & (vec2 > 0)
    if mask.sum() < 2:
        return 0.0
    v1, v2 = vec1[mask], vec2[mask]

    v1, v2 = v1 - v1.mean(), v2 - v2.mean()
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    base_sim = np.dot(v1, v2) / denom if denom > 0 else 0.0
    return base_sim * (mask.sum() / (mask.sum() + shrinkage_pearson))

def recommend_user_based_weighted_100(req, sim_metric="cosine",
                                       shrinkage_cos=10,
                                       shrinkage_pearson=5,
                                       top_n=None,
                                       coverage_shrinkage=5,
                                       df=None):
    if df is None:
        df = pd.read_csv(req.data_path, index_col=0)

    df = df.fillna(0)
    user_id = int(req.user_id)
    if user_id not in df.index:
        return RecoResponse(user_id=req.user_id, recommendations=[])

    target_ratings = df.loc[user_id]
    liked_books = target_ratings[target_ratings >= req.like_threshold].index.tolist()
    if not liked_books:
        return RecoResponse(user_id=req.user_id, recommendations=[])

    sims = {}
    for other_id in df.index:
        if other_id == user_id:
            continue
        other_ratings = df.loc[other_id]

        if sim_metric == "cosine":
            sim = cosine_similarity(target_ratings.values, other_ratings.values, shrinkage_cos)
        elif sim_metric == "pearson":
            sim = pearson_similarity(target_ratings.values, other_ratings.values, shrinkage_pearson)
        else:
            raise ValueError(f"Similarity {sim_metric} não suportada")

        sims[other_id] = sim

    if not sims:
        return RecoResponse(user_id=req.user_id, recommendations=[])
    if top_n is not None:
        sims = dict(sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n])

    scores = {}
    sim_sums = {}
    for other_id, sim in sims.items():
        coverage = (df.loc[user_id] > 0) & (df.loc[other_id] > 0)
        coverage_score = coverage.sum() / (coverage.sum() + coverage_shrinkage)

        for book, rating in df.loc[other_id].items():
            if rating >= req.like_threshold and book not in liked_books:
                scores[book] = scores.get(book, 0) + sim * rating * coverage_score
                sim_sums[book] = sim_sums.get(book, 0) + sim * coverage_score


    if scores:
        min_score, max_score = min(scores.values()), max(scores.values())
        for book in scores:
            scores[book] = (scores[book] - min_score) / (max_score - min_score + 1e-9)

    recommendations_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_books = [book for book, _ in recommendations_list]
    ndcg = ndcg_score(recommended_books, liked_books, req.top_n)

    final_recommendations = [
        RecoItem(book=book, score=scores[book], reason=f"{sim_metric} (NDCG:{ndcg:.4f})")
        for book, _ in recommendations_list[:req.top_n]
    ]

    return RecoResponse(user_id=req.user_id, recommendations=final_recommendations)

def recommend_user_based_weighted_optimized(req, sim_metric="cosine",
                                                 shrinkage_cos=5,
                                                 shrinkage_pearson=2,
                                                 coverage_shrinkage=2,
                                                 top_n=100,
                                                 df=None):
    if df is None:
        df = pd.read_csv(req.data_path, index_col=0)

    df = df.fillna(0)
    user_id = int(req.user_id)
    if user_id not in df.index:
        return RecoResponse(user_id=req.user_id, recommendations=[])

    target_ratings = df.loc[user_id]
    liked_books = target_ratings[target_ratings >= req.like_threshold].index.tolist()
    if not liked_books:
        return RecoResponse(user_id=req.user_id, recommendations=[])

    sims = {}
    for other_id in df.index:
        if other_id == user_id:
            continue
        other_ratings = df.loc[other_id]

        if sim_metric == "cosine":
            sim = cosine_similarity(target_ratings.values, other_ratings.values, shrinkage_cos)
        elif sim_metric == "pearson":
            sim = pearson_similarity(target_ratings.values, other_ratings.values, shrinkage_pearson)
        elif sim_metric == "hybrid":
            sim_cos = cosine_similarity(target_ratings.values, other_ratings.values, shrinkage_cos)
            sim_pear = pearson_similarity(target_ratings.values, other_ratings.values, shrinkage_pearson)
            mask = (target_ratings.values > 0) & (other_ratings.values > 0)
            alpha_adj = 0.9 if mask.sum() < 3 else 0.5
            sim = alpha_adj * sim_cos + (1 - alpha_adj) * sim_pear
        else:
            raise ValueError(f"Similarity {sim_metric} não suportada")

        sims[other_id] = sim

    if not sims:
        return RecoResponse(user_id=req.user_id, recommendations=[])

    sims = dict(sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n])

    scores = {}
    sim_sums = {}
    for other_id, sim in sims.items():
        coverage = (df.loc[user_id] > 0) & (df.loc[other_id] > 0)
        coverage_score = coverage.sum() / (coverage.sum() + coverage_shrinkage)

        for book, rating in df.loc[other_id].items():
            if rating >= req.like_threshold and book not in liked_books:
                scores[book] = scores.get(book, 0) + sim * rating * coverage_score
                sim_sums[book] = sim_sums.get(book, 0) + sim * coverage_score

    if scores:
        min_score, max_score = min(scores.values()), max(scores.values())
        for book in scores:
            scores[book] = (scores[book] - min_score) / (max_score - min_score + 1e-9)

    recommendations_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    recommended_books = [book for book, _ in recommendations_list]
    ndcg = ndcg_score(recommended_books, liked_books, req.top_n)

    final_recommendations = [
        RecoItem(book=book, score=scores[book], reason=f"{sim_metric} (NDCG:{ndcg:.4f})")
        for book, _ in recommendations_list[:req.top_n]
    ]

    return RecoResponse(user_id=req.user_id, recommendations=final_recommendations)
