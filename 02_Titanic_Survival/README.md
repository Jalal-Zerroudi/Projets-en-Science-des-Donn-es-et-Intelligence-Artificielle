
# Titanic Survival Prediction API

Ce projet propose une API web construite avec **FastAPI** pour prédire la survie de passagers du Titanic, à partir de caractéristiques comme leur âge, classe, sexe, etc. Il inclut également une visualisation des performances du modèle et une analyse statistique des données de test.

---

## Fonctionnalités

- **API REST** avec FastAPI pour la prédiction de survie (`/predict`)
- **Page HTML `/metrics`** affichant :
  - Performances du modèle (Accuracy, Precision, Recall, F1 Score)
  - Aperçu des données de test
  - Analyse statistique des données (âge moyen, répartition par sexe et classe)
- Modèle de machine learning entraîné avec `LogisticRegression` sur le dataset Titanic
- Backend proprement structuré (FastAPI + Jinja2)

---

## Structure du projet

```
Titanic_Survival/
├── CreateModels/
│   ├── TitanicDataset.csv
│   ├── main.ipynb           # Préparation, entraînement et sauvegarde du modèle
│
├── backend/
│   ├── main.py              # API FastAPI
│   ├── model_handler.py     # Classe de gestion du modèle
│   ├── model.pkl            # Modèle ML + X_test/y_test
│   └── templates/
│       └── metrics.html     # Page HTML avec analyse + performances
```

---

## Lancer l'application

### 1. Prérequis

- Python 3.8+
- Bibliothèques Python :
  ```
  pip install fastapi uvicorn scikit-learn pandas jinja2
  ```

### 2. Entraîner et sauvegarder le modèle

Dans `CreateModels/main.ipynb` :

```python
# Extrait final
with open("../backend/model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "X_test": X_test,
        "y_test": y_test
    }, f)
```

### 3. 🚀 Lancer FastAPI

Dans le dossier `backend` :

```bash
uvicorn main:app --reload
```

- Interface Swagger UI : [http://localhost:8000/docs](http://localhost:8000/docs)
- Analyse et performance : [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## 📌 Exemple d'appel API `/predict`

```json
POST /predict
{
  "Pclass": 3,
  "Sex": 1,
  "Age": 22.0,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": 0
}
```

Réponse :
```json
{
  "survived": false
}
```

---

## 🧠 Modèle utilisé

- `LogisticRegression` (Scikit-learn)
- Données issues du dataset Titanic (`TitanicDataset.csv`)
