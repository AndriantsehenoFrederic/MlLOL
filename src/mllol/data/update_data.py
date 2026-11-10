import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
CLEAN_FILE_NAME = "cleaned_dataset.json"
FILE_PATH = Path(__file__).resolve().parent
FILE_NAME = "dataset.json"

df = pd.read_json(FILE_PATH / FILE_NAME)
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
df = df.loc[df["datetime_utc"]>= pd.Timestamp("2026-01-01", tz="UTC")].copy() # on garde que les match de 2026
print(f"Nombres de matchs: {len(df)}")
print(f"Voici les colonnes avant d'ếtre nettoyé: {df.columns}")

# On va enlever des colonnes

df = df.drop(columns=["match_id", "tournament", "game_number", "gamelength", "riot_game_id", "source_url"])
print(f"Voici les colonnes qui restent: {df.columns}")

# Sauvegarde du dataset nettoyé
json_clean_df = df.to_json(FILE_PATH/CLEAN_FILE_NAME, force_ascii=False)
