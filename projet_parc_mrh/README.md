# Prédiction du parc de résidences principales par code INSEE

Ce projet construit une chaîne reproductible pour produire une série de part de résidences principales par commune et par année.

La cible est :

`part_residences_principales = nb_residences_principales / nb_logements`

Le pipeline sépare trois statuts :

- `OBSERVEE` pour les millésimes issus du Recensement.
- `INTERPOLEE` pour les années situées entre deux observations.
- `PREDITE` pour les années postérieures au dernier millésime communal.

La sortie métier existe sous deux formes :

- `predictions_long.csv` : code INSEE, année, valeur, statut et modèle.
- `predictions_wide.csv` : code INSEE et une colonne par année.

## Architecture

```text
projet_parc_mrh/
├── configs/default.yml
├── data/reference/
├── docs/
├── scripts/
│   ├── download_sources.py
│   └── run_pipeline.py
├── src/parc_mrh/
│   ├── __init__.py
│   ├── __main__.py
│   └── pipeline.py
├── tests/
└── .github/workflows/ci.yml
```

## Méthode

Les points communaux observés utilisent les millésimes 2012, 2017 et 2023. La base 2023 est le jeu long `DS_RP_LOGEMENT_PRINC`. Les données historiques sont ramenées vers la géographie communale 2026 avec la table officielle de passage Insee.

Lorsqu'une commune historique se scinde en plusieurs communes actuelles, le projet répartit les effectifs historiques selon le nombre de logements observé en 2023 dans les communes cibles. Cette règle est une allocation de modélisation. Elle n'est pas présentée comme une donnée Insee.

Deux familles de projection sont évaluées :

1. régression linéaire locale sur la transformation logit de la part ;
2. projection calibrée sur la trajectoire annuelle EAPL.

Le modèle retenu pour les années futures est celui qui obtient la MAE moyenne la plus faible sur les backtests 2017 et 2023.

Le projet ajoute un diagnostic de Breusch-Pagan pour l'hétéroscédasticité et une régression logistique secondaire. Cette dernière sert à produire une classification au-dessus d'un seuil de 80 % et à mesurer vrais positifs, faux positifs, faux négatifs et vrais négatifs.

## Installation Windows

Dans PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m pytest
```

Si PowerShell bloque l'activation :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Exécution

Télécharge d'abord les sources :

```powershell
python scripts\download_sources.py
```

Puis lance le pipeline :

```powershell
python scripts\run_pipeline.py
```

Les résultats sont écrits dans `data/processed/`.

## Mise à jour

Lorsqu'un nouveau millésime communal apparaît :

1. ajoute la source dans `configs/default.yml` ;
2. ajoute l'année dans `observed_years` ;
3. ajuste `forecast_start_year` ;
4. actualise la référence EAPL ;
5. lance les tests ;
6. relance le téléchargement ;
7. relance le pipeline ;
8. contrôle les backtests et les doublons `code_insee/annee`.

Sources officielles : la publication Insee des bases logement, la page des tables de passage des communes et la publication EAPL sont référencées dans `configs/default.yml` et `docs/METHODOLOGY.md`.
