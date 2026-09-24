# Méthodologie

## Cible

La variable principale est la part des résidences principales dans le parc de logements.

part_rp = nb_rp / nb_logements

La transformation logit évite que les projections sortent de l'intervalle [0, 1].

## Source communale

La publication Insee « Logement en 2023 » fournit les données communales des logements, résidences principales, résidences secondaires, logements occasionnels et logements vacants. Elle indique que les données 2012, 2017 et 2023 sont diffusées dans la géographie en vigueur au 1er janvier 2026.

Le pipeline filtre le jeu harmonisé DS_RP_LOGEMENT_PRINC sur les communes, les logements et les dimensions totales. Il extrait :

- OCS = _T pour le total des logements ;
- OCS = DW_MAIN pour les résidences principales.

Les trois millésimes passent donc dans la même chaîne de préparation. Cela évite de mélanger plusieurs schémas et évite une allocation historique inutile.

## EAPL

Les EAPL fournissent une trajectoire annuelle du parc de logements. L'Insee publie les séries historiques avec des statuts révisé ou provisoire selon l'année.

Le projet utilise cette trajectoire comme variable externe pour les projections postérieures au dernier recensement communal.

## Modèles

### Baseline

Pour chaque commune :

logit(part_rp) ~ année

La régression linéaire sert de référence.

### Modèle EAPL

Pour chaque commune :

logit(p_t) = logit(p_2023) + beta_i * [logit(EAPL_t) - logit(EAPL_2023)]

beta_i mesure la sensibilité locale à la trajectoire nationale. Le coefficient est régularisé vers la médiane départementale et borné entre -3 et 3.

## Validation temporelle

Le projet respecte l'ordre temporel :

- prédiction de 2017 avec les données jusqu'en 2012 ;
- prédiction de 2023 avec les données jusqu'en 2017.

Les métriques sont MAE, RMSE, R², biais moyen et erreur absolue maximale.

Le modèle futur est choisi selon la MAE moyenne des backtests.

## Tests statistiques

Le test LM de Breusch-Pagan diagnostique l'hétéroscédasticité des erreurs de backtest.

La régression logistique est secondaire. Elle classe une commune au-dessus ou sous 80 % de résidences principales en 2023. Les sorties comprennent précision, rappel, F1, spécificité, ROC-AUC et matrice de confusion.

## Sorties

predictions_long.csv est la table de référence.

predictions_wide.csv répond au besoin d'un code INSEE et d'une colonne par année.

Les colonnes de statut et de modèle restent disponibles pour audit.

## Limites

Les années futures restent des estimations. Elles ne constituent pas des données officielles Insee.

La qualité d'une projection doit être contrôlée sur les backtests et réévaluée à chaque nouveau millésime.
