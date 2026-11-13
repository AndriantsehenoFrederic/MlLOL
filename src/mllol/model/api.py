"""API locale pour sélectionner des champions et utiliser le modèle entraîné."""

import csv
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import DEFAULT_OUTPUT, ROLES, encode_composition


ARTIFACTS = Path(os.environ.get("MLLOL_ARTIFACTS", DEFAULT_OUTPUT))
ASSET_FILE = Path(__file__).with_name("champion_assets.json")

app = FastAPI(title="MlLOL Composition API", version="0.1.0")


class PredictionRequest(BaseModel):
    blue: list[str] = Field(min_length=5, max_length=5)
    red: list[str] = Field(min_length=5, max_length=5)
    team: Literal["blue", "red"] = "blue"


def _vocabulary() -> dict[str, int]:
    path = ARTIFACTS / "vocabulary.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Vocabulaire absent. Lance d'abord l'entraînement.")
    names = json.loads(path.read_text(encoding="utf-8"))["champions"]
    return {name: index for index, name in enumerate(names)}


@lru_cache(maxsize=1)
def _model():
    path = ARTIFACTS / "composition_model.keras"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Modèle absent. Lance d'abord l'entraînement.")
    os.environ["KERAS_BACKEND"] = "jax"
    import keras

    return keras.models.load_model(path, compile=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": (ARTIFACTS / "composition_model.keras").is_file()}


@app.get("/champions")
def champions() -> dict:
    vocabulary = _vocabulary()
    assets = json.loads(ASSET_FILE.read_text(encoding="utf-8"))
    version = assets["version"]
    image_keys = assets["image_keys"]

    match_counts: dict[str, int] = {}
    stats_path = ARTIFACTS / "champion_stats_train.csv"
    if stats_path.is_file():
        with stats_path.open(newline="", encoding="utf-8") as file:
            match_counts = {row["champion"]: int(row["matches"]) for row in csv.DictReader(file)}

    return {
        "model_ready": (ARTIFACTS / "composition_model.keras").is_file(),
        "roles": list(ROLES),
        "champions": [
            {
                "name": name,
                "portrait_url": (
                    f"https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{image_keys[name]}.png"
                    if name in image_keys else None
                ),
                "matches": match_counts.get(name, 0),
            }
            for name in vocabulary
        ],
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    vocabulary = _vocabulary()
    canonical = {name.casefold(): name for name in vocabulary}

    def normalize(picks: list[str]) -> list[str]:
        normalized = []
        for raw_name in picks:
            name = canonical.get(raw_name.strip().casefold())
            if name is None:
                raise HTTPException(status_code=422, detail=f"Champion absent de l'entraînement : {raw_name}")
            normalized.append(name)
        return normalized

    blue = normalize(request.blue)
    red = normalize(request.red)
    all_picks = blue + red
    if len(set(all_picks)) != 10:
        raise HTTPException(status_code=422, detail="Chaque champion ne peut être sélectionné qu'une fois.")

    features, _ = encode_composition(blue, red, vocabulary)
    probability_blue = float(_model().predict(features[None, :], verbose=0)[0, 0])
    return {
        "blue_probability": probability_blue,
        "red_probability": 1.0 - probability_blue,
        "selected_team": request.team,
        "selected_probability": probability_blue if request.team == "blue" else 1.0 - probability_blue,
    }
