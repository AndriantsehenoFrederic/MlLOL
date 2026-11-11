# Modèle de victoire selon les compositions LoL

Le modèle reçoit **les cinq champions bleus et les cinq champions rouges**, dans
l'ordre `top, jungle, mid, bot, support`. Il estime la probabilité que l'équipe
**bleue** gagne. La commande de prédiction peut ensuite afficher la probabilité
de l'équipe bleue ou rouge, selon celle du champion choisi.

Le fichier utilisé par défaut est `src/mllol/data/cleaned_dataset.json`. Il
contient des **matchs professionnels de 2026** : les résultats décrivent donc
ce contexte compétitif et ne sont pas automatiquement valables en solo queue.

## 1. Préparer l'environnement

À la racine du projet :

```bash
uv sync --extra ml --inexact
```

L'option `ml` installe Keras et JAX ; `--inexact` conserve aussi les outils de
notebook déjà présents dans ton environnement virtuel. Le projet utilise Python 3.14 ; le backend
JAX est fixé **avant** l'import de Keras dans les scripts. L'entraînement
fonctionne sur CPU ; un GPU n'est pas nécessaire pour commencer.

## 2. Calculer seulement les statistiques

```bash
uv run --no-sync python -m mllol.model.train --stats-only
```

Cette commande **n'entraîne aucun modèle** et n'importe pas Keras. Elle :

1. vérifie que chaque partie a une date, un identifiant, dix picks et un vainqueur cohérents ;
2. trie les parties dans le temps et crée trois périodes de 70 %, 15 % et 15 %, sans couper une journée de matchs ;
3. calcule les taux de victoire champion × rôle sur **l'entraînement seulement** ;
4. compare ces statistiques sur les périodes de validation et de test.

Deux repères sont mesurés :

- **Taux côté bleu** : donne à toutes les compositions le même taux de victoire bleu observé à l'entraînement.
- **Statistiques champion × rôle** : combine les taux de victoire des cinq alliés et cinq ennemis, avec un lissage de 50 matchs virtuels à 50 % pour éviter qu'un champion très rare paraisse certain de gagner ou perdre.

Les statistiques sont un point de comparaison, pas une preuve de causalité : un
taux élevé peut aussi refléter la force des équipes qui sélectionnent ce
champion. Le script affiche le top 10 des champions les plus joués dans
l'entraînement et enregistre les résultats dans `src/mllol/model/artifacts/`.

## 3. Entraîner le modèle

```bash
uv run --no-sync python -m mllol.model.train
```

Options utiles :

```bash
uv run --no-sync python -m mllol.model.train --epochs 80 --batch-size 128 --seed 42
```

Le modèle est un petit réseau Keras : dix blocs binaires indiquent le champion
de chaque rôle et de chaque côté ; deux couches cachées permettent d'apprendre
des interactions entre champions. La régularisation et le dropout limitent le
surapprentissage. L'entraînement s'arrête si la perte de validation ne
s'améliore plus pendant dix époques et conserve les meilleurs poids.

`team_blue`, `team_red`, les bans, le patch et la VOD ne sont **pas** des
variables d'entrée : la question posée concerne les compositions. Les noms
d'équipes servent seulement à créer la cible gagnant/perdant. Le patch reste
dans le rapport des prédictions pour pouvoir examiner une éventuelle dérive.

## 4. Lire les résultats

Fichiers produits dans `src/mllol/model/artifacts/` :

| Fichier | Contenu |
| --- | --- |
| `dataset_split.json` | Nombre et dates des matchs par période ; picks inconnus en validation/test |
| `champion_stats_train.csv` | Fréquence et taux brut par champion sur l'entraînement |
| `champion_role_stats_train.csv` | Fréquence, taux brut et taux lissé par champion et rôle |
| `metrics.json` | Scores des repères et du modèle sur validation/test |
| `history.csv` | Perte et précision à chaque époque |
| `test_predictions.csv` | Trois probabilités et résultat réel pour chaque match de test |
| `test_calibration.csv` | Probabilité moyenne annoncée et fréquence réelle de victoire par tranche |
| `composition_model.keras` | Modèle entraîné |
| `vocabulary.json` | Noms et ordre des champions utilisés par le modèle |

Dans `metrics.json`, compare d'abord **`log_loss`** et **`brier_score`** sur
`test` : plus petits = meilleures probabilités. La précision à seuil 0,5 est
secondaire, car ton objectif est une **probabilité**. Si le modèle ne bat pas
les statistiques champion × rôle sur les matchs récents, garde les
statistiques comme solution simple et examine les données avant d'ajouter de
la complexité. Évite d'ajuster le réseau après avoir regardé le test : utilise
la validation pour les choix, puis réserve le test à la comparaison finale.

## 5. Prédire une composition

Après l'entraînement :

```bash
uv run --no-sync python -m mllol.model.predict \
  --blue 'Aatrox,Lee Sin,Ahri,Jinx,Thresh' \
  --red 'Gnar,Vi,Orianna,Ezreal,Nautilus' \
  --team blue
```

Remplace les noms par tes dix champions réels. Les espaces dans les noms sont
acceptés ; entoure chaque liste de guillemets. Pour obtenir la probabilité de
ton champion côté rouge, utilise `--team red`. Un champion absent de la
période d'entraînement provoque une erreur explicite plutôt qu'une estimation
peu fiable.

## Limites à expliquer dans le projet

- Les champions rarement joués ont des estimations incertaines ; leur fréquence figure dans les CSV.
- Les changements de patch, les joueurs et le niveau des équipes influencent la victoire mais ne figurent pas dans les variables du modèle.
- Le résultat est une association issue de matchs professionnels, pas l'effet causal d'un choix de champion.
- La prédiction suppose que les **deux compositions complètes** sont connues. Pour prédire pendant une draft incomplète, il faudrait un autre entraînement.
