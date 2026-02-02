README — FUSE (FUmerolle Segmentation Engine)
============================================

Objectif
--------
FUSE est un algorithme de détection et de segmentation de fumerolles en imagerie visible (RGB).
Il fonctionne sur une région d’intérêt (ROI) définie par l’utilisateur et combine :
- une détection « physique » simple basée sur l’image,
- des masques candidats générés par SAM,
- une fusion finale pour limiter les faux positifs.

Les résultats sont pensés comme des indicateurs robustes et reproductibles,
pas comme une segmentation parfaite pixel à pixel.


Dépendances
-----------
Python 3.9+ recommandé

Bibliothèques :
- numpy
- Pillow
- matplotlib
- scikit-image
- ultralytics (SAM)
- opencv-python (optionnel, selon évolutions)

Installation typique :
pip install numpy pillow matplotlib scikit-image opencv-python ultralytics

Poids du modèle SAM :
- sam_b.pt (chargé par défaut dans le code)


Principe général de l’algorithme
-------------------------------

1) Prétraitement et ROI
   - Découpage de la ROI dans l’image RGB.
   - Conversion en luminance et normalisation.
   - Objectif : travailler sur une image plus stable que le RGB brut.

2) Détection physique (pré-masque)
   - Calcul d’indices simples à partir de l’image (contraste, texture locale).
   - Identification d’anomalies compatibles avec une fumerolle.
   - Nettoyage morphologique.
   - Résultat : un masque grossier mais physiquement cohérent.

3) Génération de masques par SAM
   - Application de SAM uniquement dans la ROI.
   - Génération de plusieurs masques candidats à partir de points positifs/négatifs.
   - Ces masques peuvent inclure des faux positifs (rochers, relief, nuages).

4) Sélection et fusion
   - Évaluation des masques SAM avec des critères simples (cohérence, taille, forme).
   - Sélection du meilleur masque SAM.
   - Fusion finale : intersection entre le masque SAM sélectionné et le pré-masque physique.
   - Objectif : conserver uniquement la partie cohérente avec une fumerolle.

5) Validation et sortie
   - Rejet si le masque final est trop petit ou incohérent.
   - Sinon, sortie d’un masque final binaire et de métriques associées
     (aire, scores, indicateurs de dominance).


Entrées / Sorties
-----------------

Entrées :
- image RGB
- ROI = (x1, y1, x2, y2) en pixels
- S_min : aire minimale pour éviter le bruit
- show : affichage des figures de diagnostic

Sorties :
- detected : booléen (fumerolle détectée ou non)
- mask_final : masque binaire dans la ROI
- metrics : dictionnaire de métriques et/ou raisons de rejet


Limitations actuelles
---------------------
- La ROI peut inclure l’édifice volcanique, source majeure de faux positifs.
- SAM est coûteux en temps de calcul.
- L’étape d’extension du pré-masque peut devenir lente sur des zones larges.


TODO
----
- Masque de “clip volcan” :
  Exclure explicitement l’édifice (masque binaire ou polygone par caméra)
  avant la sélection et la fusion des masques.

- Sortie visuelle standardisée :
  Générer une image PNG de diagnostic (ROI, pré-masque, masque final, métriques).

- Amélioration des performances :
  - Réduire le coût de l’extension du pré-masque.
  - N’appeler SAM que si la détection physique est suffisamment robuste.
  - Approche coarse-to-fine (basse résolution → raffinement).

- Lecture et stockage séqentielle de milliers d'image à la suite
- Ajouter un point de départ de fumerolles -> critères morphologiques
- corriger les errors lors du run (RuntimeWarning)
