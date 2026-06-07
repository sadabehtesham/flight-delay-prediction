from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "flights.csv"
SAMPLE_DATASET_PATH = BASE_DIR / "sample_flights.csv"

FEATURE_COLUMNS = [
    "MONTH",
    "DAY",
    "SCHEDULED_DEPARTURE",
    "DEPARTURE_DELAY",
    "TAXI_OUT",
    "SCHEDULED_TIME",
    "ELAPSED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL",
    "DIVERTED",
    "CANCELLED",
]


def load_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        flights = pd.read_csv(DATASET_PATH, low_memory=False)
    else:
        flights = pd.read_csv(SAMPLE_DATASET_PATH, low_memory=False)
    return flights.iloc[:10000].copy()


def train_model():
    flights = load_dataset()

    prepared = flights.drop(
        columns=[
            "YEAR",
            "FLIGHT_NUMBER",
            "AIRLINE",
            "TAIL_NUMBER",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "DEPARTURE_TIME",
            "WHEELS_OFF",
            "AIR_TIME",
            "WHEELS_ON",
            "DAY_OF_WEEK",
            "TAXI_IN",
            "CANCELLATION_REASON",
            "ARRIVAL_TIME",
            "ARRIVAL_DELAY",
        ],
        errors="ignore",
    )
    prepared = prepared.fillna(prepared.mean(numeric_only=True))
    prepared["result"] = (flights["ARRIVAL_DELAY"] > 15).astype(int)

    X = prepared[FEATURE_COLUMNS].to_numpy()
    y = prepared["result"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]),
    }

    return model, scaler, metrics


MODEL, SCALER, METRICS = train_model()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form.get(col, 0)) for col in FEATURE_COLUMNS]
    sample = np.array([values], dtype=float)
    sample_scaled = SCALER.transform(sample)

    prediction = int(MODEL.predict(sample_scaled)[0])
    probability = float(MODEL.predict_proba(sample_scaled)[0][1])
    label = "Likely delayed" if prediction == 1 else "Likely on time"

    return render_template(
        "result.html",
        label=label,
        probability=probability * 100,
        prediction=prediction,
        values=dict(zip(FEATURE_COLUMNS, values)),
        metrics=METRICS,
    )


if __name__ == "__main__":
    app.run(debug=True)
