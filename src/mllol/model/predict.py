"""Prédit la chance de victoire d'une composition complète déjà choisie.

Exemple : uv run --no-sync python -m mllol.model.predict \
  --blue 'Aatrox,Lee Sin,Ahri,Jinx,Thresh' \
  --red 'Gnar,Vi,Orianna,Ezreal,Nautilus' --team blue
"""

import argparse
import json
import os
from pathlib import Path

from .data import DEFAULT_OUTPUT, ROLES, encode_composition


def _parse_picks(value: str) -> list[str]:
    picks = [champion.strip() for champion in value.split(",")]
    if len(picks) != len(ROLES) or not all(picks):
        raise argparse.ArgumentTypeError(
            "Indique cinq champions séparés par des virgules, dans l'ordre top,jungle,mid,bot,support."
        )
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blue", required=True, type=_parse_picks, help="Cinq picks bleus séparés par des virgules")
    parser.add_argument("--red", required=True, type=_parse_picks, help="Cinq picks rouges séparés par des virgules")
    parser.add_argument("--team", choices=("blue", "red"), required=True, help="Équipe dont on veut la probabilité")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    vocabulary_path = args.artifacts / "vocabulary.json"
    model_path = args.artifacts / "composition_model.keras"
    if not vocabulary_path.is_file() or not model_path.is_file():
        parser.error("Modèle ou vocabulaire absent : lance d'abord l'entraînement.")

    vocabulary_data = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary = {name: index for index, name in enumerate(vocabulary_data["champions"])}
    canonical_names = {name.casefold(): name for name in vocabulary}

    def normalize(picks: list[str]) -> list[str]:
        unknown = [name for name in picks if name.casefold() not in canonical_names]
        if unknown:
            parser.error(
                f"Champions absents des matchs d'entraînement : {', '.join(unknown)}. "
                "La probabilité serait peu fiable pour ces picks."
            )
        return [canonical_names[name.casefold()] for name in picks]

    blue = normalize(args.blue)
    red = normalize(args.red)
    if len(set(blue)) != 5 or len(set(red)) != 5:
        parser.error("Un champion ne peut pas apparaître deux fois dans la même équipe.")
    if set(blue) & set(red):
        parser.error("Un champion ne peut pas apparaître dans les deux équipes d'une même partie.")

    features, _ = encode_composition(blue, red, vocabulary)
    os.environ["KERAS_BACKEND"] = "jax"
    import keras

    model = keras.models.load_model(model_path, compile=False)
    probability_blue = float(model.predict(features[None, :], verbose=0)[0, 0])
    probability_team = probability_blue if args.team == "blue" else 1 - probability_blue
    print(f"Probabilité estimée de victoire ({args.team}) : {probability_team:.1%}")


if __name__ == "__main__":
    main()
