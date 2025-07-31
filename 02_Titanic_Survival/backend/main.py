from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model_handler import ModelHandler
import pickle

# Charger le modèle
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model_handler = ModelHandler("model.pkl")
model_handler.load_test_data(data["X_test"], data["y_test"])

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(passenger: Passenger):
    features = [
        passenger.Pclass, passenger.Sex, passenger.Age,
        passenger.SibSp, passenger.Parch, passenger.Fare, passenger.Embarked
    ]
    prediction = model_handler.predict(features)
    return {"survived": bool(prediction)}

@app.get("/metrics", response_class=HTMLResponse)
def show_metrics(request: Request):
    metrics = model_handler.get_metrics()
    preview_data = model_handler.get_test_data_preview()
    analysis = model_handler.get_data_analysis()

    if "error" in metrics:
        return HTMLResponse(content=f"<h2>❌ Erreur : {metrics['error']}</h2>")

    return templates.TemplateResponse("metrics.html", {
        "request": request,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "data_preview": preview_data,
        "data_analysis": analysis
    })
