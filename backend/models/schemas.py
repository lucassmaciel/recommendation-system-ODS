from pydantic import BaseModel, Field
from typing import List

class RecoRequest(BaseModel):
    user_id: str = Field(..., description="Username alvo")
    k_neighbors: int = Field(20, ge=1, le=100)
    top_n: int = Field(5, ge=1, le=50)
    like_threshold: float = Field(7.0, ge=0, le=10)

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
