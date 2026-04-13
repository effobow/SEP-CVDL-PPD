# SEP-CVDL-PPD

Ce dépôt contient notre travail autour du projet **SEP-CVDL**, consacré à la classification des émotions faciales.  
L’objectif a été de reprendre le projet d’origine, de reproduire ses premiers résultats, puis d’y apporter plusieurs améliorations pour rendre les expériences plus simples à lancer, mieux organisées et plus faciles à analyser.

## Objectifs du travail

Notre travail s’est structuré autour de plusieurs axes :

- tester le modèle sur des jeux de données externes, en particulier **JAFFE**
- rendre le script d’entraînement plus flexible grâce à des paramètres en ligne de commande
- automatiser le lancement de plusieurs modèles à la suite
- mieux organiser les sorties du pipeline
- améliorer la traçabilité des résultats produits

## Scripts principaux

Les scripts principaux du dépôt sont les suivants :

- `train_eval.py` : script d’entraînement principal
- `train_eval_preprocessed.py` : version adaptée pour les expériences avec prétraitement
- `eval_best_model.py` : évaluation d’un modèle entraîné
- `run_all_models_fer2013.bat` : lancement automatique de plusieurs modèles sur FER2013
- `run_all_models_fer2013_preprocessed.bat` : lancement automatique des modèles sur la version prétraitée
- `run_eval_all_preproc.bat` : évaluation des modèles prétraités

## Organisation des résultats

Les résultats sont regroupés dans le dossier `results/`, avec une organisation par type d’expérience :

- `results/axes_amelioration/jaffe/`  
  contient les résultats liés au test sur le dataset **JAFFE**

- `results/entrainement_tous_les_modeles/`  
  contient les résultats de comparaison entre plusieurs architectures sur **FER2013**

- `results/fer2013_pretraite/`  
  contient les résultats obtenus après relance des expériences avec une version **prétraitée** du dataset FER2013

Cette organisation permet de mieux distinguer les différentes étapes du projet et de retrouver plus facilement les sorties associées à chaque expérience.

## Modèles comparés

Dans nos expériences, nous avons comparé plusieurs architectures :

- GiMeFive
- GiMeFiveRes
- ResNet18
- ResNet34
- VGG

## Jeu de données externe

Nous avons testé le modèle sur le dataset **JAFFE** afin d’observer sa capacité de généralisation en dehors de son cadre d’origine.  
Nous avons également tenté une ouverture vers un autre dataset externe, **ExpW**, mais la version récupérée ne présentait pas une qualité suffisante pour être exploitée correctement dans l’étude finale.

## Contenu complémentaire

Le dossier `documentation_modeles_ajoutes/` regroupe des fichiers de travail complémentaires liés à l’ajout et à l’analyse de modèles supplémentaires.

---
Projet réalisé dans le cadre de notre master MLSD.

Dan Sebag
Gaspard Lugat
Mehdi Benayed
Dimitri Deramond
