# Rift Oracle — interface Next.js de MlLOL

Interface locale pour choisir les cinq champions de chaque équipe, rôle par rôle,
puis afficher la probabilité de victoire calculée par le modèle Keras déjà entraîné.

## Démarrage

Le fichier `src/mllol/model/artifacts/composition_model.keras` et le vocabulaire
doivent exister. Si besoin, entraîne d'abord le modèle avec la commande indiquée
dans [le guide ML](../src/mllol/model/README.md).

Depuis la racine `MlLOL`, installe les dépendances Python :

```bash
uv sync --extra ml --extra web --inexact
```

Dans un **premier terminal**, démarre l'API Python :

```bash
uv run --no-sync uvicorn mllol.model.api:app --host 127.0.0.1 --port 8000
```

Dans un **second terminal**, démarre Next.js :

```bash
cd web
npm install
npm run dev
```

Ouvre [http://localhost:3000](http://localhost:3000). Clique sur un rôle, puis
sur le portrait d'un champion. La sélection passe automatiquement au prochain
rôle vide. Un champion choisi ne peut pas être sélectionné une deuxième fois.
Choisis ensuite l'équipe dont tu veux connaître les chances et lance l'analyse.

## Fonctionnement

- `src/mllol/model/api.py` charge le vocabulaire du modèle et sert les routes `/champions`, `/predict` et `/health`.
- Next.js appelle cette API côté serveur via `web/src/app/api/`. Le navigateur utilise uniquement les routes `/api/*` de Next.js.
- Les portraits proviennent de [Riot Data Dragon](https://developer.riotgames.com/docs/lol#data-dragon), avec les identifiants officiels enregistrés dans `champion_assets.json`.
- Seuls les champions présents dans l'entraînement sont proposés. Après un nouvel entraînement, redémarre l'API Python pour recharger le modèle.
- La prédiction suppose que les dix picks sont connus et concerne les matchs professionnels similaires au dataset 2026.

L'adresse de l'API Python est `http://127.0.0.1:8000` par défaut. Pour la
changer, définis `MLLOL_API_URL` dans `web/.env.local`, par exemple :

```text
MLLOL_API_URL=http://127.0.0.1:8001
```

## Vérifications

Depuis `web/` :

```bash
npm run typecheck
npm run build
```

Les visuels et noms des champions appartiennent à Riot Games. Cette interface
est un projet personnel non affilié à Riot Games.
