from pathlib import Path


class Settings:
    ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = ROOT / "processed-data" / "df_filtrado.csv"

settings = Settings()
