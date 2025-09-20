from fastapi import APIRouter, HTTPException
from backend.models.schemas import RecoRequest, RecoResponse, SimilarItemsResponse
from backend.services.data import get_dataset_path, load_ratings_df
from backend.services.recommender import recommend_item_based, similar_items_item_based

router = APIRouter(tags=["recommender"])

@router.get("/dataset-path")
def dataset_path():
    return {"path": str(get_dataset_path())}

@router.get("/stats") ## uso para df em nuvem -- confirmar com Lucas
def stats():
    df = load_ratings_df()
    return {
        "rows": int(len(df)),
        "users": int(df["User-ID"].nunique()),
        "books": int(df["Book-Title"].nunique()),
        "rating_min": float(df["Rating"].min()),
        "rating_max": float(df["Rating"].max()),
    }

@router.post("/recomendar", response_model=RecoResponse)
def recomendar(req: RecoRequest):
    try:
        return recommend_item_based(req)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/similar/{book_id}", response_model=SimilarItemsResponse)
def similares(book_id: str, top_n: int = 5):
    try:
        items = similar_items_item_based(book_id, top_n=top_n)
        return SimilarItemsResponse(book=book_id, similar=items)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
