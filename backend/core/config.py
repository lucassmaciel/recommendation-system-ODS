from pathlib import Path

class Settings:
    ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = ROOT / "processed-data" / "final_df.csv"  # no momento csv cru

settings = Settings()