"""Entraînement Keras et comparaison avec des statistiques de référence.

Exécution : uv run --no-sync python -m mllol.model.train
Statistiques seules : uv run --no-sync python -m mllol.model.train --stats-only
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_DATA, DEFAULT_OUTPUT, GameSplits, ROLES, encode_games, load_games, make_vocabulary, split_by_day
from .statistics import ChampionStatsBaseline, calibration_table, score_probabilities


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _describe_split(games: pd.DataFrame) -> dict:
    return {
        "matches": int(len(games)),
        "first_match_utc": games["datetime_utc"].min().isoformat(),
        "last_match_utc": games["datetime_utc"].max().isoformat(),
        "blue_win_rate": float(games["blue_win"].mean()),
    }


def _prepare_statistics(
    splits: GameSplits, output: Path, alpha: float, evaluate_test: bool
) -> tuple[ChampionStatsBaseline, dict]:
    baseline = ChampionStatsBaseline.fit(splits.train, alpha=alpha)
    role_stats = baseline.table()
    role_stats.to_csv(output / "champion_role_stats_train.csv", index=False)

    champion_stats = (
        role_stats.groupby("champion", as_index=False)
        .agg(matches=("matches", "sum"), wins=("wins", "sum"))
        .assign(win_rate=lambda frame: frame["wins"] / frame["matches"])
        .sort_values(["matches", "champion"], ascending=[False, True])
    )
    champion_stats.to_csv(output / "champion_stats_train.csv", index=False)

    results = {}
    periods = [("validation", splits.validation)]
    if evaluate_test:
        periods.append(("test", splits.test))
    for name, games in periods:
        y = games["blue_win"].to_numpy(dtype=np.int8)
        results[name] = {
            "blue_side_rate": score_probabilities(y, np.full(len(games), baseline.blue_rate)),
            "champion_role_stats": score_probabilities(y, baseline.predict_proba(games)),
        }

    print("\nTop 10 des champions joués dans l'entraînement :")
    print(champion_stats[["champion", "matches", "win_rate"]].head(10).to_string(index=False))
    if evaluate_test:
        print("\nScores statistiques sur les matchs de test :")
        for name, scores in results["test"].items():
            print(f"  {name}: log_loss={scores['log_loss']:.4f}, Brier={scores['brier_score']:.4f}, accuracy={scores['accuracy_at_0_5']:.3f}")
    return baseline, results


def _build_model(input_width: int, seed: int):
    # Import tardif : --stats-only fonctionne sans installation de Keras/JAX.
    os.environ["KERAS_BACKEND"] = "jax"
    import keras

    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.Input(shape=(input_width,)),
            keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)),
            keras.layers.Dropout(0.30),
            keras.layers.Dense(8, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return keras, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Fichier cleaned_dataset.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Dossier des résultats")
    parser.add_argument("--stats-only", action="store_true", help="Calcule les statistiques, sans importer Keras ni entraîner")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=50.0, help="Lissage du taux de victoire champion × rôle")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs et --batch-size doivent être positifs.")

    games = load_games(args.data)
    splits = split_by_day(games)
    vocabulary = make_vocabulary(splits.train)
    args.output.mkdir(parents=True, exist_ok=True)

    split_info = {
        "data_path": str(args.data.resolve()),
        "train": _describe_split(splits.train),
        "validation": _describe_split(splits.validation),
        "test": _describe_split(splits.test),
        "champions_in_train": len(vocabulary),
        "roles": list(ROLES),
        "target": "probability_blue_team_wins",
        "features": "five blue picks then five red picks, each in top/jungle/mid/bot/support order",
        "excluded_features": ["team_blue", "team_red", "bans_blue", "bans_red", "patch", "vod"],
    }
    _save_json(args.output / "dataset_split.json", split_info)
    _save_json(args.output / "vocabulary.json", {"champions": list(vocabulary), "roles": list(ROLES)})

    print(f"Parties : {len(games)} ; entraînement : {len(splits.train)} ; validation : {len(splits.validation)} ; test : {len(splits.test)}")
    print(f"Champions observés dans l'entraînement : {len(vocabulary)}")
    baseline, results = _prepare_statistics(splits, args.output, args.alpha, args.stats_only)
    if args.stats_only:
        _save_json(args.output / "metrics.json", {"statistics_only": True, "scores": results})
        print(f"\nStatistiques sauvegardées dans {args.output.resolve()} ; aucun modèle entraîné.")
        return

    x_train, unknown_train = encode_games(splits.train, vocabulary)
    x_validation, unknown_validation = encode_games(splits.validation, vocabulary)
    x_test, unknown_test = encode_games(splits.test, vocabulary)
    split_info["unknown_champion_picks"] = {
        "train": unknown_train,
        "validation": unknown_validation,
        "test": unknown_test,
    }
    _save_json(args.output / "dataset_split.json", split_info)
    if unknown_validation or unknown_test:
        print(f"Attention : {unknown_validation} picks inconnus en validation et {unknown_test} en test ; leurs blocs sont à zéro.")

    y_train = splits.train["blue_win"].to_numpy(dtype=np.float32)
    y_validation = splits.validation["blue_win"].to_numpy(dtype=np.float32)
    y_test = splits.test["blue_win"].to_numpy(dtype=np.float32)

    keras, model = _build_model(x_train.shape[1], args.seed)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, min_delta=1e-3, restore_best_weights=True
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    pd.DataFrame(history.history).to_csv(args.output / "history.csv", index=False)
    model.save(args.output / "composition_model.keras")

    validation_probabilities = model.predict(x_validation, verbose=0).reshape(-1)
    test_probabilities = model.predict(x_test, verbose=0).reshape(-1)
    results["test"] = {
        "blue_side_rate": score_probabilities(y_test, np.full(len(splits.test), baseline.blue_rate)),
        "champion_role_stats": score_probabilities(y_test, baseline.predict_proba(splits.test)),
    }
    results["validation"]["keras_model"] = score_probabilities(y_validation, validation_probabilities)
    results["test"]["keras_model"] = score_probabilities(y_test, test_probabilities)
    _save_json(
        args.output / "metrics.json",
        {
            "statistics_only": False,
            "selection_metric": "validation log_loss (EarlyStopping on val_loss)",
            "test_evaluated_after_training": True,
            "scores": results,
        },
    )

    predictions = pd.DataFrame(
        {
            "game_id": splits.test["game_id"].to_numpy(),
            "datetime_utc": splits.test["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
            "patch": splits.test["patch"].to_numpy(),
            "blue_win": y_test.astype(int),
            "p_blue_side_rate": np.full(len(splits.test), baseline.blue_rate),
            "p_champion_role_stats": baseline.predict_proba(splits.test),
            "p_keras_model": test_probabilities,
        }
    )
    predictions.to_csv(args.output / "test_predictions.csv", index=False)
    calibration_table(y_test, test_probabilities).to_csv(
        args.output / "test_calibration.csv", index=False
    )

    print("\nComparaison sur les matchs de test :")
    for name, scores in results["test"].items():
        print(f"  {name}: log_loss={scores['log_loss']:.4f}, Brier={scores['brier_score']:.4f}, accuracy={scores['accuracy_at_0_5']:.3f}")
    print(f"\nModèle et rapports sauvegardés dans {args.output.resolve()}")


if __name__ == "__main__":
    main()
