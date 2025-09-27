from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from filelock import FileLock
from backend.core.config import settings
from backend.models.schemas import RatingIn
from backend.services.data import load_user_item_matrix

router = APIRouter(tags=["ratings"])

# --- helper para achar o books_info.csv ---
def get_books_info_path() -> Path:
    """
    Retorna o caminho do books_info.csv (formato long).
    Se não houver BOOKS_INFO_PATH no settings, faz fallback para
    o diretório do DATA_PATH: <DATA_PATH>.parent / 'books_info.csv'.
    """
    if hasattr(settings, "BOOKS_INFO_PATH") and settings.BOOKS_INFO_PATH:
        return Path(settings.BOOKS_INFO_PATH)
    # fallback
    return Path(settings.DATA_PATH).parent / "books_info.csv"

def get_matrix_path() -> Path:
    """Caminho do final_user_df.csv (formato wide)."""
    return Path(settings.DATA_PATH)

def _atomic_write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Escrita atômica para CSV long (index=False)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(path)

@router.post("/rating")
def add_or_update_rating(r: RatingIn):
    """
    Upsert:
      1) books_info.csv (long): remove (user_id, book) anterior e insere nova linha com metadados + rating
      2) final_user_df.csv (wide): garante linha do usuário e coluna do livro e atualiza a célula
    """
    books_path = get_books_info_path()   # long
    mat_path   = get_matrix_path()       # wide (final_user_df.csv)

    if not books_path.exists():
        raise HTTPException(status_code=500, detail=f"books_info.csv not found at: {books_path}")
    if not mat_path.parent.exists():
        raise HTTPException(status_code=500, detail=f"Directory for final_user_df.csv not found: {mat_path.parent}")

    # Locks em ordem estável (mesma ordem sempre) p/ evitar interlock
    lock_paths = sorted([books_path, mat_path], key=lambda p: str(p))
    locks = [FileLock(f"{p}.lock") for p in lock_paths]

    try:
        # adquire todos os locks
        for L in locks:
            L.acquire(timeout=10)

        # ---------- 1) Upsert no books_info (long) ----------
        bi = pd.read_csv(books_path)

        required = {"user_id", "book", "rating", "author", "year", "publisher", "image"}
        missing = required - set(bi.columns)
        if missing:
            raise HTTPException(status_code=500, detail=f"books_info.csv missing columns: {sorted(missing)}")

        # normaliza tipos básicos
        bi["user_id"] = bi["user_id"].astype(str)
        bi["book"] = bi["book"].astype(str)

        # valida se o título existe no catálogo (em qualquer usuário)
        catalog_row = bi[bi["book"] == r.book]
        if catalog_row.empty:
            raise HTTPException(status_code=400, detail=f"Book not found in catalog: {r.book}")

        # metadados canônicos da 1ª ocorrência
        meta = catalog_row.iloc[0][["author", "year", "publisher", "image"]].to_dict()

        # remove avaliação anterior do mesmo (user_id, book)
        mask_same = (bi["user_id"] == str(r.user_id)) & (bi["book"] == str(r.book))
        bi = bi.loc[~mask_same]

        # insere a nova linha
        new_row = {
            "user_id": str(r.user_id),
            "book": str(r.book),
            "rating": int(r.rating),
            "author": meta.get("author"),
            "year": meta.get("year"),
            "publisher": meta.get("publisher"),
            "image": meta.get("image"),
        }
        bi = pd.concat([bi, pd.DataFrame([new_row])], ignore_index=True)

        _atomic_write_csv(bi, books_path, index=False)

        # ---------- 2) Upsert no final_user_df (wide) ----------
        if mat_path.exists():
            mat = pd.read_csv(mat_path)
        else:
            # cria estrutura mínima caso ainda não exista
            mat = pd.DataFrame(columns=["user_id"])

        if "user_id" not in mat.columns:
            mat.insert(0, "user_id", [])

        # garante tipos e chaves
        mat["user_id"] = mat["user_id"].astype(str)

        # cria linha do usuário se não existir
        if str(r.user_id) not in set(mat["user_id"]):
            mat = pd.concat([mat, pd.DataFrame([{"user_id": str(r.user_id)}])], ignore_index=True)

        # cria coluna do livro se não existir
        if str(r.book) not in mat.columns:
            mat[str(r.book)] = 0

        # atualiza a célula
        mat.loc[mat["user_id"] == str(r.user_id), str(r.book)] = int(r.rating)

        # normaliza numéricos (mantendo user_id como string)
        num_cols = [c for c in mat.columns if c != "user_id"]
        mat[num_cols] = mat[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        _atomic_write_csv(mat, mat_path, index=False)

    finally:
        # libera em ordem inversa
        for L in reversed(locks):
            try:
                L.release()
            except Exception:
                pass

    # limpa cache do dataset em memória para próximas recomendações
    try:
        load_user_item_matrix.cache_clear()
    except Exception:
        pass

    return {"ok": True}