"""Repères statistiques simples, ajustés sur l'entraînement uniquement."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import ROLES


def _logit(probability: float) -> float:
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return float(np.log(probability / (1 - probability)))


def _sigmoid(value: float) -> float:
    return float(1 / (1 + np.exp(-value)))


@dataclass
class ChampionStatsBaseline:
    """Estime les chances bleues avec les taux de victoire champion × rôle.

    alpha ajoute des matchs virtuels à 50 % afin de limiter l'effet des champions
    très peu joués. La moyenne des cinq écarts garde les probabilités prudentes.
    """

    blue_rate: float
    rates: dict[tuple[str, str], float]
    counts: dict[tuple[str, str], tuple[int, int]]
    alpha: float

    @classmethod
    def fit(cls, train: pd.DataFrame, alpha: float = 50.0) -> "ChampionStatsBaseline":
        if alpha <= 0:
            raise ValueError("alpha doit être strictement positif.")
        counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for blue, red, blue_win in zip(
            train["picks_blue"], train["picks_red"], train["blue_win"]
        ):
            for role, champion in zip(ROLES, blue):
                counts[(champion, role)][0] += 1
                counts[(champion, role)][1] += int(blue_win)
            for role, champion in zip(ROLES, red):
                counts[(champion, role)][0] += 1
                counts[(champion, role)][1] += 1 - int(blue_win)

        rates = {
            key: (wins + alpha * 0.5) / (played + alpha)
            for key, (played, wins) in counts.items()
        }
        blue_rate = (int(train["blue_win"].sum()) + 1) / (len(train) + 2)
        return cls(
            blue_rate=blue_rate,
            rates=rates,
            counts={key: (values[0], values[1]) for key, values in counts.items()},
            alpha=alpha,
        )

    def predict_proba(self, games: pd.DataFrame) -> np.ndarray:
        predictions = np.empty(len(games), dtype=np.float64)
        for index, (blue, red) in enumerate(zip(games["picks_blue"], games["picks_red"])):
            role_effects = [
                _logit(self.rates.get((blue_champion, role), 0.5))
                - _logit(self.rates.get((red_champion, role), 0.5))
                for role, blue_champion, red_champion in zip(ROLES, blue, red)
            ]
            predictions[index] = _sigmoid(_logit(self.blue_rate) + np.mean(role_effects))
        return predictions

    def table(self) -> pd.DataFrame:
        rows = [
            {
                "champion": champion,
                "role": role,
                "matches": played,
                "wins": wins,
                "raw_win_rate": wins / played,
                "smoothed_win_rate": self.rates[(champion, role)],
            }
            for (champion, role), (played, wins) in self.counts.items()
        ]
        return pd.DataFrame(rows).sort_values(
            ["matches", "champion", "role"], ascending=[False, True, True]
        )


def score_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Mesures communes aux repères et au modèle ; les petites valeurs sont meilleures."""
    true = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(predicted, 1e-7, 1 - 1e-7)
    return {
        "log_loss": float(-np.mean(true * np.log(clipped) + (1 - true) * np.log(1 - clipped))),
        "brier_score": float(np.mean((true - predicted) ** 2)),
        "accuracy_at_0_5": float(np.mean((predicted >= 0.5) == true)),
    }


def calibration_table(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Compare probabilité annoncée et fréquence réellement observée par tranche."""
    bins = np.minimum((np.asarray(probabilities) * 10).astype(int), 9)
    frame = pd.DataFrame({"bin": bins, "probability": probabilities, "outcome": y_true})
    return (
        frame.groupby("bin", as_index=False)
        .agg(matches=("outcome", "size"), mean_probability=("probability", "mean"), win_rate=("outcome", "mean"))
        .assign(probability_range=lambda data: data["bin"].map(lambda index: f"{index / 10:.1f}–{(index + 1) / 10:.1f}"))
    )
