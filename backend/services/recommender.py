from backend.models.schemas import RecoRequest, RecoResponse, RecoItem, SimilarItemsResponse

def recommend_stub(req: RecoRequest) -> RecoResponse:
    """
    Placeholder de recomendação.
    Futuro: implementar KNN user-based/item-based.
    """
    # retornando vazio por enquanto
    return RecoResponse(user_id=req.user_id, recommendations=[])

def similar_items_stub(book_id: str, top_n: int = 5) -> SimilarItemsResponse:
    """
    Placeholder de itens similares.
    """
    return SimilarItemsResponse(book=book_id, similar=[])
