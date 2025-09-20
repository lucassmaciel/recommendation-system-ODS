from math import sqrt
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd

from backend.models.schemas import RecoRequest, RecoResponse, RecoItem
from backend.services.data import load_ratings_df

# ---------- utils ----------

def _build_dicts(df: pd.DataFrame) -> tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    user_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)
    item_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)

    for row in df.itertuples(index=False):
        u = row.user
        b = row.book
        r = float(row.rating)
        user_ratings[u][b] = r
        item_ratings[b][u] = r
    return dict(user_ratings), dict(item_ratings)

def _cosine_items(
        b1: str, b2: str, item_ratings: Dict[str, Dict[str, float]], min_overlap: int
) -> tuple[float, int]:
    """similaridade do cosseno entre itens (apenas usuários em comum)."""
    u1 = item_ratings.get(b1, {})
    u2 = item_ratings.get(b2, {})
    if not u1 or not u2:
        return 0.0, 0
    common = set(u1).intersection(u2)
    n = len(common)
    if n < min_overlap:
        return 0.0, n
    num = sum(u1[u] * u2[u] for u in common)
    den1 = sqrt(sum(u1[u] * u1[u] for u in common))
    den2 = sqrt(sum(u2[u] * u2[u] for u in common))
    if den1 == 0 or den2 == 0:
        return 0.0, n
    return num / (den1 * den2), n

# ---------- recommenders ----------

def recommend_item_based(req: RecoRequest) -> RecoResponse:
    """
    Para cada candidato (livro não avaliado pelo usuário),
    olha os k itens mais similares entre os que o usuário já avaliou
    e calcula um score ponderado pela similaridade.
    """
    df = load_ratings_df()
    user_ratings, item_ratings = _build_dicts(df)

    u = req.user_id
    if u not in user_ratings:
        return RecoResponse(user_id=u, recommendations=[])

    # livros avaliados pelo usuário alvo e universo de candidatos
    rated_by_u = user_ratings[u]                      # dict[book] = rating
    all_books = set(item_ratings.keys())
    candidates = sorted(all_books - set(rated_by_u.keys()))

    recs: List[RecoItem] = []
    min_overlap = 3  # pode expor via config se quiser

    for cand in candidates:
        sims: List[Tuple[float, str, int]] = []  # (sim, book_j, overlap)
        for b_j, r_uj in rated_by_u.items():
            sim, overlap = _cosine_items(cand, b_j, item_ratings, min_overlap=min_overlap)
            if sim > 0:
                sims.append((sim, b_j, overlap))

        if not sims:
            continue

        # pega top-k vizinhos por similaridade
        sims.sort(key=lambda t: t[0], reverse=True)
        top = sims[: req.k_neighbors]

        # score = média ponderada pelas similaridades das notas do usuário nos vizinhos
        num = 0.0
        den = 0.0
        likes = 0
        total = 0
        for sim, b_j, _ov in top:
            r_uj = rated_by_u[b_j]
            num += sim * r_uj
            den += abs(sim)
            if r_uj >= req.like_threshold:
                likes += 1
            total += 1
        score = round(num / den, 4) if den else 0.0

        reason = f"topK={len(top)}, overlap≥{min_overlap}, likes={likes}/{total}"
        recs.append(RecoItem(book=cand, score=score, reason=reason))

    recs.sort(key=lambda it: (it.score or 0.0), reverse=True)
    return RecoResponse(user_id=u, recommendations=recs[: req.top_n])

def similar_items_item_based(book_id: str, top_n: int = 5) -> List[RecoItem]:
    """Retorna itens similares a um livro usando cosine item-item."""
    df = load_ratings_df()
    _, item_ratings = _build_dicts(df)

    if book_id not in item_ratings:
        return []

    min_overlap = 3
    sims: List[Tuple[float, str, int]] = []
    for other in item_ratings.keys():
        if other == book_id:
            continue
        sim, ov = _cosine_items(book_id, other, item_ratings, min_overlap)
        if sim > 0:
            sims.append((sim, other, ov))

    sims.sort(key=lambda t: t[0], reverse=True)
    out = [RecoItem(book=other, score=round(sim, 4), reason=f"overlap={ov}") for sim, other, ov in sims[:top_n]]
    return out
