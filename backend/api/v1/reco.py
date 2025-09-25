from fastapi import APIRouter, HTTPException, Query
from backend.models.schemas import RecoRequest, RecoResponse
from backend.services.data import get_dataset_path, load_user_item_matrix
from backend.services.recommender_user import recommend_user_based
import pandas as pd
import numpy as np

router = APIRouter(tags=["recommender"])

@router.get("/dataset-path")
def dataset_path():
    return {"path": str(get_dataset_path())}

@router.get("/stats")
def stats():
    """
    Retorna stats do dataset.
    Fins de sanity do servidor.
    """
    path = dataset_path()["path"]
    df = pd.read_csv(path)

    if "user_id" in df.columns:
        df = df.set_index("user_id")

    M, N = df.shape  # M usuários, N livros
    vals = df.to_numpy(dtype=float)

    mask_rated = vals > 0
    nnz = int(mask_rated.sum())
    sparsity = 1.0 - (nnz / (M * N))

    # considera apenas avaliações > 0
    rated_vals = vals[mask_rated]
    rating_min_nonzero = float(rated_vals.min()) if nnz else 0.0
    rating_max = float(rated_vals.max()) if nnz else 0.0
    rating_mean = float(rated_vals.mean()) if nnz else 0.0

    avg_ratings_per_user = float((mask_rated.sum(axis=1).mean())) if M else 0.0
    avg_ratings_per_item = float((mask_rated.sum(axis=0).mean())) if N else 0.0

    return {
        "rows": int(M),
        "users": int(M),
        "books": int(N),
        "ratings_count": nnz,
        "sparsity": round(sparsity, 6),
        "rating_min_nonzero": rating_min_nonzero,
        "rating_max": rating_max,
        "rating_mean": round(rating_mean, 4),
        "avg_ratings_per_user": round(avg_ratings_per_user, 2),
        "avg_ratings_per_item": round(avg_ratings_per_item, 2),
    }

@router.post("/recomendar", response_model=RecoResponse)
def recomendar(
        req: RecoRequest,
        sim_metric: str = Query("hybrid", pattern="^(cosine|pearson|hybrid)$"),
):
    """
    Recomendação user-based. O corpo controla user_id, top_n, like_threshold.
    A métrica vem como query param (?sim_metric=cosine|pearson|hybrid).
    """
    try:
        df = load_user_item_matrix()
        return recommend_user_based(req, df=df, sim_metric=sim_metric)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e