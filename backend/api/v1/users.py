import contextlib
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from filelock import FileLock

from backend.core.config import settings
from backend.models.schemas import UserSignup
from backend.services.data import load_user_item_matrix

router = APIRouter(tags=["users"])

def get_matrix_path() -> Path:
    return Path(settings.DATA_PATH)  # processed-data/final_user_df.csv

def _atomic_write_csv(df: pd.DataFrame, path: Path, index) -> None:
    tmp: Path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(path)

@router.post("/users/signup")
def users_signup(payload: UserSignup):
    """
    Cadastra usuário no final_user_df.csv (formato wide):
      - Garante coluna 'user_id'
      - Se o user_id não existir, adiciona nova linha com todas as colunas (livros) = 0
      - Limpa cache da matriz para refletir no /v1/recomendar.
    """
    uid: str = payload.username.strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id vazio")

    mat_path: Path = get_matrix_path()
    if not mat_path.parent.exists():
        raise HTTPException(status_code=500, detail=f"Diretório não existe: {mat_path.parent}")

    lock = FileLock(f"{mat_path}.lock")
    with lock:
        if mat_path.exists():
            mat: pd.DataFrame = pd.read_csv(mat_path)
        else:
            # se o arquivo não existe, crie com as duas colunas
            mat = pd.DataFrame(columns=["user_id", "username"])

        # garante que as colunas existam e tenham os tipos corretos
        if "user_id" not in mat.columns:
            mat.insert(0, "user_id", [])
        if "username" not in mat.columns:
            mat.insert(1, "username", [])

        mat["user_id"] = pd.to_numeric(mat["user_id"]).astype("Int64")
        mat["username"] = mat["username"].astype(str)

        # verifica se o username já existe
        if uid in set(mat["username"]):
            # encontra o ID existente para retornar
            existing_id = mat.loc[mat["username"] == uid, "user_id"].iloc[0]
            return {"ok": True, "user_id": int(existing_id), "created": False}

        # gera um novo user_id numérico (max + 1)
        new_user_id = (mat["user_id"].max() + 1) if not mat.empty else 1

        # monta a nova linha com ID e username
        new_row = {"user_id": new_user_id, "username": uid}
        book_cols = [c for c in mat.columns if c not in ["user_id", "username"]]
        for c in book_cols:
            new_row[c] = 0.0

        mat = pd.concat([mat, pd.DataFrame([new_row])], ignore_index=True)
        _atomic_write_csv(mat, mat_path, index=False)

    # limpa cache da matriz
    with contextlib.suppress(Exception):
        load_user_item_matrix.cache_clear()

    return {"ok": True, "user_id": int(new_user_id), "created": True}
