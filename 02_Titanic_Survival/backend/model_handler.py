# model_handler.py

import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelHandler:
    def __init__(self, model_path: str):

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.X_test = None
        self.y_test = None

    def load_test_data(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def predict(self, features: list) -> int:
        return int(self.model.predict([features])[0])

    def get_metrics(self) -> dict:
        if self.X_test is None or self.y_test is None:
            return {
                "error": "Test data not loaded. Call load_test_data(X_test, y_test) first."
            }

        y_pred = self.model.predict(self.X_test)
        return {
            "accuracy": round(accuracy_score(self.y_test, y_pred), 4),
            "precision": round(precision_score(self.y_test, y_pred), 4),
            "recall": round(recall_score(self.y_test, y_pred), 4),
            "f1_score": round(f1_score(self.y_test, y_pred), 4)
        }
    
    def get_test_data_preview(self, n: int = 10):
        if self.X_test is not None:
            return self.X_test.head(n).to_dict(orient="records")
        return []

    def get_data_analysis(self) -> dict:
        if self.X_test is None:
            return {}

        df = self.X_test.copy()
        analysis = {
            "count": len(df),
            "mean_age": round(df["Age"].mean(), 2),
            "mean_fare": round(df["Fare"].mean(), 2),
            "sex_distribution": df["Sex"].value_counts().to_dict(),  # 1 = male, 0 = female
            "pclass_distribution": df["Pclass"].value_counts().to_dict()
        }
        return analysis


