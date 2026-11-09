import pandas as pd
import json
from pathlib import Path

FILE_NAME = "dataset.json"
PATH = Path(__file__).resolve().parent # savoir dans quel dossier on est
FILE_PATH = PATH / FILE_NAME

df = pd.read_parquet("hf://datasets/gptilt/lol-esports-matches/games/games.parquet")
json_df = df.to_json(FILE_PATH, force_ascii=False)
