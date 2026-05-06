from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import os

app = FastAPI(title="Smart Parking Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join("models", "best_parking_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class ParkingInput(BaseModel):
    time_of_day: int
    weekday: int
    is_holiday: int
    temperature: float
    humidity: float
    windspeed: float
    utilization_type: int
    planning_area: int
    road_density: float
    latitude: float
    longitude: float
    capacity: int
    occupied_slots: int

@app.get("/")
def home():
    return {"message": "Smart Parking Prediction API is running"}

@app.post("/predict")
def predict(data: ParkingInput):
    try:
        df = pd.DataFrame([data.model_dump()])
        prediction = float(model.predict(df)[0])
        prediction = max(0.0, min(1.0, prediction))
        available_percent = round((1 - prediction) * 100, 2)

        return {
            "predicted_occupancy_rate": round(prediction, 4),
            "availability_percent": available_percent,
            "suggestion": "Parking available" if available_percent > 30 else "Try another lot"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")