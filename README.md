# 📊 Projets en Science des Données et Intelligence Artificielle

Ce dépôt contient une série de projets développés dans le cadre de ma formation en Data Science et Intelligence Artificielle. Chaque projet explore une problématique réelle, allant de la prédiction à la cybersécurité en passant par le traitement du langage et la vision par ordinateur.

---

## 🔍 Sommaire

- [Dashboard COVID-19](#1-dashboard-covid-19)
- [Prédiction de Survie - Titanic](#2-prédiction-de-survie---titanic)
- [Prédiction du Prix des Maisons](#3-prédiction-du-prix-des-maisons)
- [Détection de Fraude Bancaire](#4-détection-de-fraude-bancaire)
- [Analyse de Sentiments sur Twitter](#5-analyse-de-sentiments-sur-twitter)
- [Nettoyage et Visualisation de Données Publiques](#6-nettoyage-et-visualisation-de-données-publiques)
- [Détection d’Attaques DDoS](#7-détection-dattaques-ddos)
- [Système de Recommandation](#8-système-de-recommandation)
- [Reconnaissance d’Émotions Faciales](#9-reconnaissance-démotions-faciales)

---

## 📁 1. Dashboard COVID-19

**Objectif :** Développer un tableau de bord interactif qui affiche les données mondiales COVID-19 et prévoit les cas des prochains jours.

- **Tâches réalisées :**
  - Collecte de données à partir d’API.
  - Visualisation dynamique (cartes, courbes).
  - Prédiction des cas par séries temporelles (régression linéaire, modèles ARIMA).
  - Interface utilisateur avec Tkinter + backend FastAPI.

- **Tech stack :** Python, Pandas, Scikit-learn, FastAPI, Tkinter, Cassandra, Docker  
- **Données :** Johns Hopkins CSSE, WHO API  

---

## 📁 2. Prédiction de Survie - Titanic

**Objectif :** Prédire la probabilité de survie des passagers du Titanic à partir de leurs données personnelles.

- **Tâches réalisées :**
  - Analyse exploratoire (EDA).
  - Traitement des valeurs manquantes et encodage.
  - Modèles : logistic regression, random forest, KNN.

- **Tech stack :** Python, Pandas, Matplotlib, Scikit-learn  
- **Données :** Kaggle Titanic Dataset  
- **Résultat :** Score F1 ≈ 0.79 avec Random Forest

---

## 📁 3. Prédiction du Prix des Maisons

**Objectif :** Estimer la valeur d’une maison selon ses caractéristiques (surface, nb pièces, emplacement...).

- **Tâches réalisées :**
  - Analyse statistique et sélection de variables.
  - Normalisation des données.
  - Régression linéaire, Gradient Boosting, XGBoost.

- **Tech stack :** Python, Scikit-learn, Seaborn, XGBoost  
- **Données :** Ames Housing Dataset (Kaggle)  
- **Résultat :** RMSE ≈ 15 000

---

## 📁 4. Détection de Fraude Bancaire

**Objectif :** Identifier les transactions financières potentiellement frauduleuses dans un dataset très déséquilibré.

- **Tâches réalisées :**
  - Gestion de l’équilibre des classes avec SMOTE.
  - PCA pour la réduction de dimensionnalité.
  - Classification avec RandomForest et SVM.

- **Tech stack :** Python, Scikit-learn, SMOTE, Matplotlib  
- **Données :** Credit Card Fraud Detection (Kaggle)  
- **Résultat :** Recall ≈ 0.92 pour la classe frauduleuse

---

## 📁 5. Analyse de Sentiments sur Twitter

**Objectif :** Déterminer l'opinion (positive/négative/neutre) des utilisateurs concernant un sujet donné.

- **Tâches réalisées :**
  - Récupération de tweets avec Tweepy.
  - Prétraitement NLP : stopwords, stemming, tokenization.
  - Classification via TextBlob et Naive Bayes.

- **Tech stack :** Python, Tweepy, NLTK, TextBlob  
- **Données :** Tweets en direct via Twitter API v2  
- **Résultat :** Précision moyenne ≈ 85%

---

## 📁 6. Nettoyage et Visualisation de Données Publiques

**Objectif :** Nettoyer et explorer visuellement des datasets publics pour révéler des tendances.

- **Tâches réalisées :**
  - Suppression des valeurs manquantes et outliers.
  - Création de visualisations interactives.
  - Exportation des rapports Power BI.

- **Tech stack :** Python, Pandas, Matplotlib, Power BI  
- **Données :** Santé, éducation, démographie (data.gouv.fr)  

---

## 📁 7. Détection d’Attaques DDoS

**Objectif :** Utiliser des modèles de classification pour détecter automatiquement des attaques DDoS dans du trafic réseau.

- **Tâches réalisées :**
  - Préparation du dataset CICIDS2017.
  - Entraînement de CNN/LSTM pour la détection.
  - Matrices de confusion, F1, Recall.

- **Tech stack :** Python, Keras, Scikit-learn, Pandas  
- **Données :** CICIDS2017 (UNB ISCX)  
- **Résultat :** Accuracy ≈ 98.5%, Recall DDoS ≈ 0.96

---

## 📁 8. Système de Recommandation

**Objectif :** Recommander des articles (films, livres...) à un utilisateur selon ses goûts ou profils similaires.

- **Tâches réalisées :**
  - Méthodes : filtrage collaboratif + content-based.
  - Similarité cosinus, TF-IDF, matrice utilisateur-item.
  - Interface console simple.

- **Tech stack :** Python, Pandas, Scikit-learn, NLP  
- **Données :** MovieLens 100k  
- **Résultat :** Recommandations personnalisées en temps réel

---

## 📁 9. Reconnaissance d’Émotions Faciales

**Objectif :** Identifier les émotions humaines à partir de photos (joie, colère, peur, tristesse...).

- **Tâches réalisées :**
  - Détection de visage avec Haar cascades.
  - Entraînement d’un CNN sur FER2013.
  - Interface d'affichage des prédictions.

- **Tech stack :** Python, TensorFlow, OpenCV  
- **Données :** FER2013 (Kaggle)  
- **Résultat :** Accuracy ≈ 65% (multi-class)

---

## 📬 Contact

**Jalal Zerroudi**  
📧 [jalal.zerroudi@usmba.ac.ma]  
🌐 [https://jalal-zerroudi.github.io/]

---

## 📄 Licence

Ce projet est sous licence MIT – voir le fichier [LICENSE](./LICENSE) pour plus d’informations.
