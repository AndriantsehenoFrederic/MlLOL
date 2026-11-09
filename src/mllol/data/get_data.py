import pandas as pd
import json
from pathlib import Path

FILE_NAME = "dataset.json"
PATH = Path(__file__).resolve().parent # savoir dans quel dossier on est
FILE_PATH = PATH / FILE_NAME

df = pd.read_parquet("hf://datasets/gptilt/lol-esports-matches/games/games.parquet")
json_df = df.to_json()
with open (FILE_PATH,"w", encoding="utf-8") as w:
    json.dump(json_df,w,ensure_ascii=False, indent=4)