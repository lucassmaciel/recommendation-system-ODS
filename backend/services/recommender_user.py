import numpy as np
import pandas as pd
from backend.models.schemas import RecoItem, RecoResponse

DEBUG = False
def _log(*a, **k):
    if DEBUG: print(*a, **k)

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

# --- item-item cosine com shrinkage (coluna vs coluna) ---
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
                                shrink_cos=10,
                                shrink_pear=5,
                                coverage_shrink=2):
    # --- carregar matriz wide ---
    if df is None:
        df = pd.read_csv(req.data_path, dtype={"user_id": str}).set_index("user_id")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = df.apply(lambda row: (row - row[row > 0].mean()) / (row[row > 0].std() or 1), axis=1)

    uid = str(req.user_id)
    _log("[rec] uid:", uid, "idx dtype:", df.index.dtype, "shape:", df.shape)
    _log("[rec] uid in index?", uid in df.index)
    if uid not in df.index:
        _log("[rec] EARLY RETURN: user not in index")
        return RecoResponse(user_id=req.user_id, recommendations=[])

    # perfil do usuário
    urow = df.loc[uid].astype(float)
    rated_cnt = int((urow > 0).sum())
    liked_books = [b for b, v in urow.items() if v >= req.like_threshold]
    _log("[rec] rated_count_user:", rated_cnt,
         "like_threshold:", req.like_threshold,
         "liked_books(sample):", liked_books[:8])

    target = urow.values
    cols = np.array(df.columns)
    seen_mask_u = (target > 0)
    seen_books = set(cols[seen_mask_u])

    # ---- métrica/overlap para perfis rasos ----
    adaptive_min_overlap = 1 if rated_cnt <= 2 else min_overlap
    local_metric = sim_metric
    if local_metric == "pearson" and rated_cnt < 3:
        local_metric = "cosine"  # Pearson precisa de ≥2 coavaliações

    # --- tentar com parâmetros adaptativos ---
    sims = []
    total_scanned = 0
    for other_id, row in df.iterrows():
        if other_id == uid:
            continue
        total_scanned += 1
        other = row.values.astype(float)
        mask = (target > 0) & (other > 0)
        ov = int(mask.sum())
        if ov < adaptive_min_overlap:
            continue

        if local_metric == "cosine":
            s = _cosine_shrunk(target, other, mask, shrink=shrink_cos)
        elif local_metric == "pearson":
            s = _pearson_shrunk(target, other, mask, shrink=shrink_pear)
        elif local_metric == "hybrid":
            s_cos = _cosine_shrunk(target, other, mask, shrink=shrink_cos)
            s_pear = _pearson_shrunk(target, other, mask, shrink=shrink_pear)
            alpha = 0.9 if ov < 3 else 0.5
            s = alpha * s_cos + (1 - alpha) * s_pear
        else:
            raise ValueError("Similarity não suportada")

        if s != 0.0:
            sims.append((other_id, float(s)))

    _log("[rec] scanned_users:", total_scanned, "sims_total_before_cut(A):", len(sims),
         "metric:", local_metric, "min_overlap:", adaptive_min_overlap)

    # --- tentativa (cosine + min_overlap=1) antes da popularidade ---
    if not sims:
        _log("[rec] RETRY neighbors: cosine + min_overlap=1 (looser)")
        for other_id, row in df.iterrows():
            if other_id == uid:
                continue
            other = row.values.astype(float)
            mask = (target > 0) & (other > 0)
            ov = int(mask.sum())
            if ov < 1:
                continue
            s = _cosine_shrunk(target, other, mask, shrink=max(2, shrink_cos // 2))
            if s != 0.0:
                sims.append((other_id, float(s)))
        _log("[rec] sims_total_before_cut(B):", len(sims))

    # Se ainda não tem vizinhos, tenta item-based se houver ao menos 1 "liked"
    if not sims:
        if liked_books:
            _log("[rec] FALLBACK item-based: using liked_books to rank candidates")
            liked_set = set(liked_books)
            candidates = [c for c in df.columns if c not in seen_books]
            scores_ib = {}

            col_cache = {b: df[b].values.astype(float) for b in set(candidates).union(liked_set)}
            for c in candidates:
                sc = 0.0
                for lb in liked_set:
                    sim = _item_cosine_shrunk(col_cache[c], col_cache[lb], shrink=10)
                    sc += sim * float(urow[lb])
                if sc != 0.0:
                    scores_ib[c] = sc
            if scores_ib:
                ranked = sorted(scores_ib.items(), key=lambda x: x[1], reverse=True)[:req.top_n]
                _log("[rec] item-based_top:", [b for b, _ in ranked[:10]])
                ndcg = _ndcg_at_k([b for b, _ in ranked], set(liked_books), req.top_n)
                return RecoResponse(
                    user_id=req.user_id,
                    recommendations=[RecoItem(book=b, score=float(s), reason=f"item-based (NDCG:{ndcg:.4f})")
                                     for b, s in ranked]
                )
            _log("[rec] item-based empty -> fallback popularity")

        # Popularidade
        _log("[rec] FALLBACK popularity: no neighbors even after retry")
        popularity = (df > 0).sum(axis=0).sort_values(ascending=False)
        recs = [b for b in popularity.index if b not in seen_books][:req.top_n]
        _log("[rec] popularity_top(sample):", recs[:8])
        return RecoResponse(
            user_id=req.user_id,
            recommendations=[RecoItem(book=b, score=None, reason="popularity") for b in recs]
        )

    # --- vizinhos e scoring ---
    sims_sorted = sorted(sims, key=lambda kv: kv[1], reverse=True)[:k_neighbors]
    neighbor_ids = [nid for nid, _ in sims_sorted]
    neighbor_sims = dict(sims_sorted)

    _log("[rec] top_neighbors:", [(nid, round(neighbor_sims[nid], 4)) for nid in neighbor_ids[:10]])
    for nid in neighbor_ids[:10]:
        ov = int(((df.loc[uid] > 0) & (df.loc[nid] > 0)).sum())
        _log(f"[rec] nbr={nid} overlap={ov} sim={neighbor_sims[nid]:.4f}")

    # baseline deviation + normalização por soma de sims + cobertura
    user_means = (df.replace(0, np.nan)).mean(axis=1).fillna(0.0)
    mean_u = float(user_means.loc[uid])

    scores = {}
    sim_sums = {}
    for nid in neighbor_ids:
        s = neighbor_sims[nid]
        v = df.loc[nid].astype(float)
        mean_v = float((v.replace(0, np.nan)).mean() or 0.0)

        overlap = ((df.loc[uid] > 0) & (v > 0)).sum()
        coverage = overlap / (overlap + coverage_shrink)

        cand_mask = (v > 0) & (~seen_mask_u)
        if not cand_mask.any():
            continue

        for book, rating in v[cand_mask].items():
            delta = float(rating - mean_v)
            scores[book] = scores.get(book, 0.0) + s * delta * coverage
            sim_sums[book] = sim_sums.get(book, 0.0) + abs(s) * coverage

    if not scores:
        _log("[rec] scores empty -> SECOND FALLBACK (weighted ratings)")
        agg, wsum = {}, {}
        for nid in neighbor_ids:
            s = neighbor_sims[nid]
            v = df.loc[nid].astype(float)
            for book, rating in v.items():
                if rating > 0 and book not in seen_books:
                    agg[book] = agg.get(book, 0.0) + s * rating
                    wsum[book] = wsum.get(book, 0.0) + abs(s)
        if not agg:
            _log("[rec] SECOND FALLBACK also empty -> popularity")
            popularity = (df > 0).sum(axis=0).sort_values(ascending=False)
            recs = [b for b in popularity.index if b not in seen_books][:req.top_n]
            _log("[rec] popularity_top(sample):", recs[:8])
            return RecoResponse(user_id=req.user_id,
                                recommendations=[RecoItem(book=b, score=None, reason="popularity") for b in recs])
        scored = [(b, agg[b] / (wsum[b] + 1e-9)) for b in agg]
        scored.sort(key=lambda x: x[1], reverse=True)
        _log("[rec] top_raw_scores_fallback:", [(b, round(s,6)) for b, s in scored[:10]])
        top = scored[:req.top_n]
        _log("[rec] final_top_fallback:", [b for b, _ in top])
        return RecoResponse(user_id=req.user_id,
                            recommendations=[RecoItem(book=b, score=s, reason="weighted_ratings") for b, s in top])

    # previsão final
    top_raw = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    _log("[rec] top_raw_scores:", [(b, round(s,6)) for b, s in top_raw])
    for b, _ in top_raw[:3]:
        contrib = []
        for nid in neighbor_ids:
            r = float(df.loc[nid, b])
            if r > 0:
                contrib.append((nid, round(neighbor_sims[nid], 4), r))
        contrib.sort(key=lambda x: x[1], reverse=True)
        _log(f"[rec] contrib[{b}]:", contrib[:5])

    combined = []
    for b in scores:
        pred = mean_u + scores[b] / (sim_sums[b] + 1e-9)
        combined.append((b, float(pred)))
    combined.sort(key=lambda x: x[1], reverse=True)

    rec_books = [b for b, _ in combined[:req.top_n]]
    ndcg = _ndcg_at_k(rec_books, set(liked_books), req.top_n)
    _log("[rec] final_top:", rec_books)
    _log("[rec] ndcg@k:", round(ndcg, 6))

    return RecoResponse(
        user_id=req.user_id,
        recommendations=[RecoItem(book=b, score=s, reason=f"{local_metric} (NDCG:{ndcg:.4f})")
                         for b, s in combined[:req.top_n]]
    )
