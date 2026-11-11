"""Lecture du dataset et transformation des compositions en variables du modèle."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROLES = ("top", "jungle", "mid", "bot", "support")
REQUIRED_COLUMNS = {
    "game_id", "datetime_utc", "patch", "team_blue", "team_red",
    "winner", "picks_blue", "picks_red",
}
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "cleaned_dataset.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class GameSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_games(path: Path) -> pd.DataFrame:
    """Charge les parties et échoue explicitement si une ligne est inutilisable."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {path}")

    games = pd.read_json(path)
    missing = REQUIRED_COLUMNS.difference(games.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if games.empty:
        raise ValueError("Le dataset est vide.")

    games = games.copy()
    games["datetime_utc"] = pd.to_datetime(
        games["datetime_utc"], utc=True, errors="coerce"
    )
    if games["datetime_utc"].isna().any():
        raise ValueError("Certaines dates sont invalides.")
    if games["game_id"].isna().any() or games["game_id"].duplicated().any():
        raise ValueError("game_id doit être présent et unique pour chaque partie.")

    for side in ("blue", "red"):
        column = f"picks_{side}"
        valid = games[column].map(
            lambda picks: isinstance(picks, list)
            and len(picks) == len(ROLES)
            and all(isinstance(c, str) and c.strip() for c in picks)
            and len(set(picks)) == len(ROLES)
        )
        if not valid.all():
            raise ValueError(f"Compositions invalides dans {column} : {(~valid).sum()}")

    valid_winners = games["winner"].eq(games["team_blue"]) | games["winner"].eq(
        games["team_red"]
    )
    if not valid_winners.all():
        raise ValueError(f"Vainqueur inconnu pour {(~valid_winners).sum()} parties.")
    if games["team_blue"].eq(games["team_red"]).any():
        raise ValueError("Une partie contient la même équipe des deux côtés.")

    games["blue_win"] = games["winner"].eq(games["team_blue"]).astype("int8")
    return games.sort_values(["datetime_utc", "game_id"]).reset_index(drop=True)


def split_by_day(games: pd.DataFrame) -> GameSplits:
    """Découpe 70/15/15 dans l'ordre du temps, sans partager une journée."""
    day_counts = games.groupby(games["datetime_utc"].dt.floor("D"), sort=True).size()
    if len(day_counts) < 3:
        raise ValueError("Il faut au moins trois journées de matchs pour le découpage.")

    cumulative = day_counts.cumsum().to_numpy()
    train_day_index = int(np.searchsorted(cumulative, len(games) * 0.70))
    validation_day_index = int(np.searchsorted(cumulative, len(games) * 0.85))
    if train_day_index >= validation_day_index or validation_day_index >= len(day_counts) - 1:
        raise ValueError("Impossible de créer trois périodes chronologiques non vides.")

    days = games["datetime_utc"].dt.floor("D")
    train_end = day_counts.index[train_day_index]
    validation_end = day_counts.index[validation_day_index]
    return GameSplits(
        train=games.loc[days <= train_end].copy(),
        validation=games.loc[(days > train_end) & (days <= validation_end)].copy(),
        test=games.loc[days > validation_end].copy(),
    )


def make_vocabulary(train: pd.DataFrame) -> dict[str, int]:
    """Apprend les noms disponibles uniquement sur les parties d'entraînement."""
    names = {
        champion
        for column in ("picks_blue", "picks_red")
        for picks in train[column]
        for champion in picks
    }
    return {name: index for index, name in enumerate(sorted(names))}


def encode_composition(
    blue: list[str], red: list[str], vocabulary: dict[str, int]
) -> tuple[np.ndarray, int]:
    """Produit dix blocs binaires : cinq rôles bleus, puis cinq rôles rouges.

    Un champion inconnu laisse son bloc à zéro. Le nombre d'inconnus est renvoyé
    pour rendre cette limite visible dans l'évaluation.
    """
    if len(blue) != len(ROLES) or len(red) != len(ROLES):
        raise ValueError("Il faut exactement cinq champions par équipe.")

    width = len(vocabulary)
    encoded = np.zeros(2 * len(ROLES) * width, dtype=np.float32)
    unknown = 0
    for side_index, picks in enumerate((blue, red)):
        for role_index, champion in enumerate(picks):
            champion_index = vocabulary.get(champion)
            if champion_index is None:
                unknown += 1
                continue
            encoded[(side_index * len(ROLES) + role_index) * width + champion_index] = 1.0
    return encoded, unknown


def encode_games(games: pd.DataFrame, vocabulary: dict[str, int]) -> tuple[np.ndarray, int]:
    encoded = np.zeros((len(games), 2 * len(ROLES) * len(vocabulary)), dtype=np.float32)
    unknown = 0
    for row_index, (blue, red) in enumerate(zip(games["picks_blue"], games["picks_red"])):
        encoded[row_index], count = encode_composition(blue, red, vocabulary)
        unknown += count
    return encoded, unknown
