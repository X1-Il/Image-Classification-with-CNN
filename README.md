# Classification d'images avec CNN
Un réseau de neurones convolutif (CNN) pour la classification d'images. Le CNN est construit à l'aide de la bibliothèque Keras et TensorFlow en backend.
## Prérequis

Python 3.x
NumPy
Pandas
Matplotlib
Scikit-learn
Keras
TensorFlow

## Données
Le script attend un fichier CSV contenant les chemins vers les images et leurs étiquettes correspondantes. Le fichier CSV doit avoir deux colonnes : image_path et label. Assurez-vous de remplacer 'path_to_data.csv' par le chemin vers votre fichier de données.
Les images d'entraînement et de test doivent être organisées dans des répertoires distincts. Remplacez 'path_to_train_directory' et 'path_to_test_directory' par les chemins vers vos répertoires d'entraînement et de test respectivement.
## Modèle
Le script définit une architecture CNN simple avec deux couches convolutionnelles, deux couches de pooling, une couche fully-connected et une couche de sortie avec une activation sigmoid pour la classification binaire.
## Augmentation de données
Le script utilise la classe ImageDataGenerator de Keras pour effectuer une augmentation de données sur les images d'entraînement. Les transformations appliquées comprennent la rotation, le décalage horizontal et vertical, le cisaillement et le zoom.
## Entraînement
Le modèle est entraîné sur les données d'entraînement augmentées pendant 10 époques. Vous pouvez ajuster le nombre d'époques selon vos besoins.
## Évaluation
Après l'entraînement, le script évalue les performances du modèle sur les données de test en calculant la perte et la précision.
Visualisation
Le script trace les courbes de perte d'entraînement et de validation pour visualiser la convergence du modèle.
## Utilisation

1 - Assurez-vous d'avoir installé toutes les bibliothèques requises.
2 - Préparez vos données dans le format attendu (fichier CSV et répertoires d'images).
3 - Modifiez les chemins vers les fichiers de données et les répertoires d'images dans le script.
4 - Exécutez le script.
