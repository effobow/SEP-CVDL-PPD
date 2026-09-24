# Dictionnaire des sorties

| Colonne | Description |
|---|---|
| code_insee | Code commune dans la géographie 2026 |
| dep | Département |
| annee | Année |
| nb_logements | Nombre de logements observé |
| nb_residences_principales | Nombre de résidences principales observé |
| part_residences_principales | Résidences principales / logements |
| statut_donnee | OBSERVEE, INTERPOLEE ou PREDITE |
| modele | Recensement, interpolation_logit, linear_logit ou eapl_calibre_logit |
| beta_local | Sensibilité communale brute à la trajectoire EAPL |
| beta_regularise | Sensibilité après régularisation départementale |

predictions_long.csv est la table de référence. predictions_wide.csv sert aux usages qui demandent une colonne par année.
