import numpy as np
import pandas as pd
from backend.models.schemas import RecoItem, RecoResponse

# --- sims com shrinkage (user-user) ---
def _cosine_shrunk(a: np.ndarray, b: np.ndarray, mask: np.ndarray, shrink=10) -> float:
    if mask.sum() == 0: return 0.0
    av, bv = a[mask], b[mask]
    av, bv = av - av.mean(), bv - bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    base = float(np.dot(av, bv) / denom) if denom > 0 else 0.0
    return base * (mask.sum() / (mask.sum() + shrink))

def _pearson_shrunk(a: np.ndarray, b: np.ndarray, mask: np.ndarray, shrink=5) -> float:
    if mask.sum() < 2: return 0.0
    av, bv = a[mask], b[mask]
    av, bv = av - av.mean(), bv - bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    base = float(np.dot(av, bv) / denom) if denom > 0 else 0.0
    return base * (mask.sum() / (mask.sum() + shrink))

# --- item-item cosine com shrinkage ---
def _item_cosine_shrunk(col_a: np.ndarray, col_b: np.ndarray, shrink=10) -> float:
    mask = (col_a > 0) & (col_b > 0)
    if mask.sum() == 0: return 0.0
    a, b = col_a[mask].astype(float), col_b[mask].astype(float)
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    base = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    return base * (mask.sum() / (mask.sum() + shrink))

def _ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, b in enumerate(recommended[:k]):
        if b in relevant:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def recommend_user_based_better(req, df=None,
                                sim_metric="hybrid",
                                k_neighbors=20,
                                min_overlap=2,
                                shrink_cos=5,
                                shrink_pear=2,
                                coverage_shrink=2):
    # --- carregar matriz wide ---
    if df is None:
        df = pd.read_csv(req.data_path, dtype={"user_id": str}).set_index("user_id")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    uid = str(req.user_id)
    if uid not in df.index:
        return RecoResponse(user_id=req.user_id, recommendations=[])

    urow = df.loc[uid].astype(float)
    rated_cnt = int((urow > 0).sum())
    liked_books = [b for b, v in urow.items() if v >= req.like_threshold]
    target = urow.values
    cols = np.array(df.columns)
    seen_mask_u = (target > 0)
    seen_books = set(cols[seen_mask_u])

    adaptive_min_overlap = 1 if rated_cnt <= 2 else min_overlap
    local_metric = sim_metric
    if local_metric == "pearson" and rated_cnt < 3:
        local_metric = "cosine"

    # --- USER-BASED SIMILARITY ---
    sims = []
    for other_id, row in df.iterrows():
        if other_id == uid:
            continue
        other = row.values.astype(float)
        mask = (target > 0) & (other > 0)
        if int(mask.sum()) < adaptive_min_overlap:
            continue

        if local_metric == "cosine":
            s = _cosine_shrunk(target, other, mask, shrink=shrink_cos)
        elif local_metric == "pearson":
            s = _pearson_shrunk(target, other, mask, shrink=shrink_pear)
        elif local_metric == "hybrid":
            s_cos = _cosine_shrunk(target, other, mask, shrink=shrink_cos)
            s_pear = _pearson_shrunk(target, other, mask, shrink=shrink_pear)
            alpha = 0.9 if int(mask.sum()) < 3 else 0.5
            s = alpha * s_cos + (1 - alpha) * s_pear
        else:
            raise ValueError("Similarity não suportada")
        if s != 0.0:
            sims.append((other_id, float(s)))

    # --- ITEM-BASED SCORE MAIS AGRESSIVO ---
    scores_item = {}
    if liked_books:
        liked_set = set(liked_books)
        candidates = [c for c in df.columns if c not in seen_books]
        col_cache = {b: df[b].values.astype(float) for b in set(candidates).union(liked_set)}
        # Popularidade: contagem de ratings positivos
        pop = (df > 0).sum(axis=0).to_dict()
        for c in candidates:
            sc = 0.0
            for lb in liked_set:
                sim = _item_cosine_shrunk(col_cache[c], col_cache[lb], shrink=10)
                sc += sim * float(urow[lb])
            if sc != 0.0:
                # Aumenta peso item-based e pondera por popularidade
                scores_item[c] = sc * 0.7 + 0.3 * pop.get(c, 0)      

    # --- USER-BASED SCORES ---
    scores_user, sim_sums = {}, {}
    neighbor_ids = [nid for nid, _ in sorted(sims, key=lambda x: x[1], reverse=True)[:k_neighbors]]
    neighbor_sims = dict(sims)
    user_means = df.replace(0, np.nan).mean(axis=1).fillna(0.0)
    mean_u = float(user_means.loc[uid])
    coverage_shrink = 1.0
    for nid in neighbor_ids:
        s = neighbor_sims[nid]
        v = df.loc[nid].astype(float)
        mean_v = float(v.replace(0, np.nan).mean() or 0.0)
        overlap = ((df.loc[uid] > 0) & (v > 0)).sum()
        coverage = overlap / (overlap + coverage_shrink)
        cand_mask = (v > 0) & (~seen_mask_u)
        for book, rating in v[cand_mask].items():
            scores_user[book] = scores_user.get(book, 0.0) + s * (rating - mean_v) * coverage
            sim_sums[book] = sim_sums.get(book, 0.0) + abs(s) * coverage

    # --- COMBINA USER + ITEM SCORES ---
    combined_scores = {}
    for b in set(list(scores_user.keys()) + list(scores_item.keys())):
        combined_scores[b] = mean_u
        if b in scores_user:
            combined_scores[b] += scores_user[b] / (sim_sums.get(b, 1e-9))
        if b in scores_item:
            combined_scores[b] += 2.0 * scores_item[b] 

    # --- ranking final ---
    combined_sorted = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:req.top_n]
    rec_books = [b for b, _ in combined_sorted]
    ndcg = _ndcg_at_k(rec_books, set(liked_books), req.top_n)

    return RecoResponse(
        user_id=req.user_id,
        recommendations=[RecoItem(book=b, score=s, reason=f"{local_metric}+item (NDCG:{ndcg:.4f})")
                         for b, s in combined_sorted]
    )
