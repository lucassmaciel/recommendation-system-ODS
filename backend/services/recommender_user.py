import numpy as np
import pandas as pd
from backend.models.schemas import RecoItem, RecoResponse

def _cosine_on_overlap(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0: return 0.0
    av, bv = a[mask], b[mask]
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(av, bv) / (na * nb))

def _pearson_on_overlap(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    if n < 2: return 0.0
    av, bv = a[mask], b[mask]
    av, bv = av - av.mean(), bv - bv.mean()
    sa, sb = np.linalg.norm(av), np.linalg.norm(bv)
    if sa == 0 or sb == 0: return 0.0
    return float(np.dot(av, bv) / (sa * sb))

def recommend_user_based(req, sim_metric="cosine", alpha=0.9, df=None, k_neighbors=20, min_overlap=2):
    # --- carregar matriz wide ---
    if df is None:
        df = pd.read_csv(req.data_path, dtype={"user_id": str}).set_index("user_id")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    uid = str(req.user_id)
    if uid not in df.index:
        return RecoResponse(user_id=req.user_id, recommendations=[])

    target = df.loc[uid].astype(float).values
    cols = np.array(df.columns)
    seen_mask_u = (target > 0)
    seen_books = set(cols[seen_mask_u])

    # médias por usuário
    user_means = (df.replace(0, np.nan)).mean(axis=1).fillna(0.0)
    mean_u = float(user_means.loc[uid])

    # --- similaridades ---
    sims = []
    for other_id, row in df.iterrows():
        if other_id == uid: continue
        other = row.values.astype(float)
        mask = (target > 0) & (other > 0)
        if mask.sum() < min_overlap:
            continue

        if sim_metric == "cosine":
            s = _cosine_on_overlap(target, other, mask)
        elif sim_metric == "pearson":
            s = _pearson_on_overlap(target, other, mask)
        elif sim_metric == "hybrid":
            s_cos = _cosine_on_overlap(target, other, mask)
            s_pear = _pearson_on_overlap(target, other, mask)
            s = (0.9 if mask.sum() < 5 else alpha) * s_cos + (0.1 if mask.sum() < 5 else (1 - alpha)) * s_pear
        else:
            raise ValueError("Similarity não suportada")

        if s != 0.0:
            sims.append((other_id, float(s)))

    if not sims:
        popularity = (df > 0).sum(axis=0).sort_values(ascending=False)
        recs = [b for b in popularity.index if b not in seen_books][:req.top_n]
        return RecoResponse(user_id=req.user_id,
                            recommendations=[RecoItem(book=b, score=None, reason="popularity") for b in recs])

    sims_sorted = sorted(sims, key=lambda kv: kv[1], reverse=True)[:k_neighbors]
    neighbor_ids = [nid for nid, _ in sims_sorted]
    neighbor_sims = dict(sims_sorted)

    # --- scoring por desvio da média do vizinho ---
    scores, weights = {}, {}
    for nid in neighbor_ids:
        s = neighbor_sims[nid]
        v = df.loc[nid].astype(float)
        mean_v = float(user_means.loc[nid])
        cand_mask = (v > 0) & (~seen_mask_u)
        if not cand_mask.any():
            continue
        contrib = v[cand_mask] - mean_v
        for book, delta in zip(cols[cand_mask], contrib.values):
            scores[book] = scores.get(book, 0.0) + s * float(delta)
            weights[book] = weights.get(book, 0.0) + abs(s)

    if not scores:
        agg, wsum = {}, {}
        for nid in neighbor_ids:
            s = neighbor_sims[nid]
            v = df.loc[nid].astype(float)
            for book, rating in v.items():
                if rating > 0 and book not in seen_books:
                    agg[book] = agg.get(book, 0.0) + s * rating
                    wsum[book] = wsum.get(book, 0.0) + abs(s)
        if not agg:
            popularity = (df > 0).sum(axis=0).sort_values(ascending=False)
            recs = [b for b in popularity.index if b not in seen_books][:req.top_n]
            return RecoResponse(user_id=req.user_id,
                                recommendations=[RecoItem(book=b, score=None, reason="popularity") for b in recs])
        scored = [(b, agg[b] / (wsum[b] + 1e-9)) for b in agg]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:req.top_n]
        return RecoResponse(user_id=req.user_id,
                            recommendations=[RecoItem(book=b, score=s, reason=sim_metric) for b, s in top])

    combined = []
    for b in scores:
        pred = mean_u + scores[b] / (weights[b] + 1e-9)
        combined.append((b, float(pred)))
    combined.sort(key=lambda x: x[1], reverse=True)
    top = combined[:req.top_n]

    return RecoResponse(user_id=req.user_id,
                        recommendations=[RecoItem(book=b, score=s, reason=sim_metric) for b, s in top])
