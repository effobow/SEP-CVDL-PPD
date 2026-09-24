# Méthodologie

## Cible

La variable principale est la part des résidences principales dans le parc de logements.

`part_rp = nb_rp / nb_logements`

La transformation logit évite que les projections sortent de l'intervalle [0, 1].

## Sources

Le Recensement de la population apporte les données communales. Le millésime 2023 utilisé ici est le jeu long `DS_RP_LOGEMENT_PRINC`, filtré sur :

- `GEO_OBJECT = COM`
- `RP_MEASURE = DWELLINGS`
- `TIME_PERIOD = 2023`
- `OBS_STATUS = A`
- toutes les autres dimensions à `_T`

Pour le stock de logements, `OCS = _T`. Pour les résidences principales, `OCS = DW_MAIN`.

Les EAPL fournissent une trajectoire annuelle nationale de référence.

## Géographie

La sortie est alignée sur la géographie communale au 1er janvier 2026. L'Insee indique que sa table de passage annuelle permet de comparer les communes sur les géographies depuis 2003.

Les historiques 2012 et 2017 sont convertis vers 2026 avant modélisation. Les effectifs sont agrégés avant de recalculer la part.

Pour une scission d'une commune historique en plusieurs communes 2026, le poids d'allocation utilise le nombre de logements 2023 des communes cibles. Cela évite une moyenne simple de pourcentages.

## Modèles

### Baseline

Pour chaque commune, une régression linéaire est ajustée sur :

`logit(part_rp) ~ année`

Elle sert de référence.

### Modèle EAPL

Pour chaque commune :

`logit(p_t) = logit(p_2023) + beta_i * [logit(EAPL_t) - logit(EAPL_2023)]`

`beta_i` mesure la sensibilité locale à la trajectoire nationale. Le coefficient est régularisé avec la médiane départementale et borné entre -3 et 3.

## Validation temporelle

Le projet évite un découpage aléatoire des années.

- 2017 est prédit avec les données disponibles jusqu'en 2012.
- 2023 est prédit avec les données disponibles jusqu'en 2017.

Les métriques sont MAE, RMSE, R², biais moyen et erreur absolue maximale.

Le modèle utilisé pour les années futures est choisi selon la MAE moyenne des backtests.

## Tests statistiques

Le test LM de Breusch-Pagan sert à diagnostiquer l'hétéroscédasticité des erreurs de backtest. Il ne remplace pas les métriques de prévision.

La régression logistique est secondaire. Elle transforme le problème en classification : commune au-dessus ou sous 80 % de résidences principales en 2023. Les sorties comprennent précision, rappel, F1, spécificité, ROC-AUC et matrice de confusion.

## Limites

Les années futures restent des estimations. Une valeur prédite ne doit pas être présentée comme une donnée officielle Insee.

L'incertitude augmente avec l'horizon. Une mise à jour du modèle doit être déclenchée à chaque nouveau millésime communal exploitable.
