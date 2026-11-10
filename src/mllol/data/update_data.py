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
json_clean_df = df.to_json(FILE_PATH/CLEAN_FILE_NAME, force_ascii=False, date_format="iso")

# Réunir les picks bleus et rouges dans une seule série
picks = pd.concat(
    [df["picks_blue"].explode(), df["picks_red"].explode()],
    ignore_index=True
).dropna()

# Compter les parties jouées par champion
parties_par_champion = picks.value_counts().sort_values()

# Graphique horizontal : hauteur adaptée au nombre de champions
fig, ax = plt.subplots(
    figsize=(12, max(8, len(parties_par_champion) * 0.23))
)

parties_par_champion.plot.barh(ax=ax, color="steelblue")
ax.bar_label(
    ax.containers[0],
    fmt="%d",
    padding=3,
    fontsize=7
)

# Laisser de la place à droite pour les nombres
ax.set_xlim(0, parties_par_champion.max() * 1.12)
ax.set_title("Nombre de parties jouées par champion en 2026")
ax.set_xlabel("Nombre de parties")
ax.set_ylabel("Champion")

plt.tight_layout()
image_path = FILE_PATH / "champions_2026.png"
fig.savefig(image_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Graphique enregistré ici : {image_path}")
print(f"----------------------------------------------Top10 champions le plus joués----------------------------------------------------------------------------\n{parties_par_champion.nlargest(10).to_string()}")