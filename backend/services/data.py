from backend.core.config import settings
from pathlib import Path

def get_dataset_path() -> Path:
    """
    Retorna o caminho do CSV Ratings cru.
    Futuro: validar colunas, cachear leitura, trocar fonte, etc.
    """
    return settings.DATA_PATH
