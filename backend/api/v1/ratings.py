from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from filelock import FileLock
from backend.core.config import settings
from backend.models.schemas import RatingIn

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

def _atomic_write_csv_long(df: pd.DataFrame, path: Path) -> None:
    """Escrita atômica para CSV long (index=False)."""
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

@router.post("/rating")
def add_or_update_rating(r: RatingIn):
    """
    Upsert no books_info.csv (formato long):
      - valida se o livro existe no catálogo (pelo 'book')
      - remove linha anterior do mesmo (user_id, book)
      - adiciona nova linha com metadados do catálogo + rating novo
    """
    path = get_books_info_path()
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"books_info.csv not found at: {path}")

    lock = FileLock(f"{path}.lock")
    with lock:
        bi = pd.read_csv(path)

        required = {"user_id", "book", "rating", "author", "year", "publisher", "image"}
        missing = required - set(bi.columns)
        if missing:
            raise HTTPException(status_code=500, detail=f"books_info.csv missing columns: {sorted(missing)}")

        # normaliza tipos básicos
        bi["user_id"] = bi["user_id"].astype(str)
        bi["book"] = bi["book"].astype(str)

        # valida se book existe no catálogo (em qualquer usuário)
        catalog_row = bi[bi["book"] == r.book]
        if catalog_row.empty:
            # Se quiser permitir livros novos, precisaria de uma fonte de metadados.
            raise HTTPException(status_code=400, detail=f"Book not found in catalog: {r.book}")

        # pega metadados do primeiro match
        meta = catalog_row.iloc[0][["author", "year", "publisher", "image"]].to_dict()

        # remove avaliação anterior do mesmo (user_id, book), se existir
        mask_same = (bi["user_id"] == r.user_id) & (bi["book"] == r.book)
        bi = bi.loc[~mask_same]

        # adiciona nova linha
        new_row = {
            "user_id": r.user_id,
            "book": r.book,
            "rating": int(r.rating),
            "author": meta.get("author"),
            "year": meta.get("year"),
            "publisher": meta.get("publisher"),
            "image": meta.get("image"),
        }
        bi = pd.concat([bi, pd.DataFrame([new_row])], ignore_index=True)

        # salva (index=False no long)
        _atomic_write_csv_long(bi, path)

    return {"ok": True}
