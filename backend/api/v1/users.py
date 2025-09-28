from pathlib import Path
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from filelock import FileLock

from backend.core.config import settings
from backend.services.data import load_user_item_matrix
from backend.models.schemas import UserSignup

router = APIRouter(tags=["users"])

def get_matrix_path() -> Path:
    return Path(settings.DATA_PATH)  # processed-data/final_user_df.csv

def _atomic_write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(path)

@router.post("/users/signup")
def users_signup(payload: UserSignup):
    """
    Cadastra usuário no final_user_df.csv (formato wide):
      - Garante coluna 'user_id'
      - Se o user_id não existir, adiciona nova linha com todas as colunas (livros) = 0
      - Limpa cache da matriz para refletir no /v1/recomendar
    """
    uid = payload.user_id.strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id vazio")

    mat_path = get_matrix_path()
    if not mat_path.parent.exists():
        raise HTTPException(status_code=500, detail=f"Diretório não existe: {mat_path.parent}")

    lock = FileLock(f"{mat_path}.lock")
    with lock:
        if mat_path.exists():
            mat = pd.read_csv(mat_path)
        else:
            mat = pd.DataFrame(columns=["user_id"])

        if "user_id" not in mat.columns:
            mat.insert(0, "user_id", [])
        mat["user_id"] = mat["user_id"].astype(str)

        # se já existir, só retorna ok
        if uid in set(mat["user_id"]):
            try:
                load_user_item_matrix.cache_clear()
            except Exception:
                pass
            return {"ok": True, "user_id": uid, "created": False}

        # monta a nova linha: todas as colunas (livros) = 0
        cols = list(mat.columns)
        if cols and cols[0] == "user_id":
            book_cols = cols[1:]
        else:
            book_cols = []

        new_row = {"user_id": uid}
        for c in book_cols:
            new_row[c] = 0

        mat = pd.concat([mat, pd.DataFrame([new_row])], ignore_index=True)

        # normaliza numéricos nas colunas de livros
        if book_cols:
            mat[book_cols] = mat[book_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        _atomic_write_csv(mat, mat_path, index=False)

    # limpa cache da matriz
    try:
        load_user_item_matrix.cache_clear()
    except Exception:
        pass

    return {"ok": True, "user_id": uid, "created": True}
