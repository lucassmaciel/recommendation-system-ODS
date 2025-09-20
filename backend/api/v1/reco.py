from fastapi import APIRouter, HTTPException
from backend.models.schemas import RecoRequest, RecoResponse, SimilarItemsResponse
from backend.services.data import get_dataset_path
from backend.services.recommender import recommend_stub, similar_items_stub

router = APIRouter(tags=["recommender"])

@router.get("/dataset-path")
def dataset_path():
    """
    Rota utilitária para debug: mostra onde esperamos o CSV cru.
    """
    return {"path": str(get_dataset_path())}

@router.post("/recomendar", response_model=RecoResponse)
def recomendar(req: RecoRequest):
    """
    POST /v1/recomendar
    Por enquanto devolve estrutura vazia (sem KNN).
    """
    # validações simples se path existe de fato
    path = get_dataset_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"ratings.csv não encontrado em {path}")
    return recommend_stub(req)

@router.get("/similar/{book_id}", response_model=SimilarItemsResponse)
def similares(book_id: str, top_n: int = 5):
    """
    GET /v1/similar/{book_id}
    Placeholder de 'itens similares'.
    """
    return similar_items_stub(book_id, top_n=top_n)
