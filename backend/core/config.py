from pathlib import Path

class Settings:
    ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = ROOT / "unprocessed-data" / "ratings.csv"  # no momento csv cru

settings = Settings()