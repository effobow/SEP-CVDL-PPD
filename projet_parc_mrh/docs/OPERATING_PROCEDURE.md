# Procédure d'exploitation

## Première installation

```powershell
git clone https://github.com/effobow/SEP-CVDL-PPD.git
cd SEP-CVDL-PPD\projet_parc_mrh
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m pytest
```

## Première exécution

```powershell
python scripts\download_sources.py
python scripts\run_pipeline.py
```

## Contrôles

```powershell
python -m ruff check src tests scripts
python -m pytest
```

Puis contrôle :

- `data/processed/predictions_long.csv`
- `data/processed/predictions_wide.csv`
- `data/processed/backtest_metrics.csv`
- `data/processed/diagnostic_tests.csv`
- `data/processed/logistic_metrics.csv`
- `data/processed/model_registry.json`

## Nouvelle année

Quand l'Insee publie un nouveau millésime communal :

1. mettre à jour la source ;
2. ajouter le millésime aux années observées ;
3. ajouter les données EAPL ;
4. exécuter les tests ;
5. exécuter le téléchargement ;
6. exécuter le pipeline ;
7. vérifier les métriques ;
8. vérifier l'absence de doublons ;
9. examiner les codes géographiques non appariés.

## Publication Git

```powershell
git status
git add projet_parc_mrh
git commit -m "feat: update MRH prediction pipeline"
git push origin feature/parc-mrh-prevision-insee
```
