from pydantic import BaseModel, Field
from typing import List

class RecoRequest(BaseModel):
    user_id: str = Field(..., description="Username alvo")
    k_neighbors: int = Field(3, ge=1, le=50)
    top_n: int = Field(5, ge=1, le=50)
    like_threshold: float = Field(3.5, ge=0, le=5)

class RecoItem(BaseModel):
    book: str
    score: float | None = None
    reason: str | None = None

class RecoResponse(BaseModel):
    user_id: str
    recommendations: List[RecoItem]

class SimilarItemsResponse(BaseModel):
    book: str
    similar: List[RecoItem]
