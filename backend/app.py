from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.v1.reco import router as reco_router

app = FastAPI(title="Book Recommender API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  ## to-do: isolar aplicação na prod
    allow_methods=["*"],  ## to-do: isolar aplicação na prod
    allow_headers=["*"],  ## to-do: isolar aplicação na prod
)

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

# v1 routes
app.include_router(reco_router, prefix="/v1")
