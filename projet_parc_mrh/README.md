# Prédiction du parc de résidences principales par code INSEE

Ce projet construit une chaîne reproductible pour produire une série de part de résidences principales par commune et par année.

La cible est :

part_residences_principales = nb_residences_principales / nb_logements

Le pipeline sépare trois statuts :

- OBSERVEE pour les millésimes issus du Recensement.
- INTERPOLEE pour les années situées entre deux observations.
- PREDITE pour les années postérieures au dernier millésime communal.

La sortie métier existe sous deux formes :

- predictions_long.csv : code INSEE, année, valeur, statut et modèle.
- predictions_wide.csv : code INSEE et une colonne par année.

## Architecture

projet_parc_mrh/
├── configs/default.yml
├── data/reference/
├── docs/
├── scripts/
├── src/parc_mrh/
├── tests/
└── .github/workflows/ci.yml

## Méthode

La publication Insee du logement en 2023 diffuse les années 2012, 2017 et 2023 dans la géographie communale au 1er janvier 2026. Le pipeline exploite directement cette source harmonisée. Il n'ajoute donc pas de réallocation historique par table de passage dans la chaîne principale.

Deux familles de projection sont évaluées :

1. régression linéaire locale sur l'échelle logit ;
2. projection calibrée sur la trajectoire annuelle EAPL.

Le modèle retenu pour les années futures est celui qui obtient la MAE moyenne la plus faible sur les backtests 2017 et 2023.

Le projet ajoute un diagnostic de Breusch-Pagan pour l'hétéroscédasticité et une régression logistique secondaire. Cette dernière sert à produire une classification au-dessus d'un seuil de 80 % et à mesurer vrais positifs, faux positifs, faux négatifs et vrais négatifs.

## Installation Windows

Dans PowerShell :

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m pytest

## Exécution

python scripts\download_sources.py
python scripts\run_pipeline.py

Les résultats sont écrits dans data/processed/.

## Mise à jour

Lorsqu'un nouveau millésime communal apparaît :

1. ajoute ou remplace la source dans configs/default.yml ;
2. ajoute l'année dans observed_years ;
3. ajuste forecast_start_year ;
4. actualise la référence EAPL ;
5. lance les tests ;
6. relance le téléchargement ;
7. relance le pipeline ;
8. contrôle les backtests et les doublons code_insee/annee.

Les sources officielles sont documentées dans docs/METHODOLOGY.md.
