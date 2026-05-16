from fastapi import FastAPI
from fastapi.responses import Response
from backend.inference import predict_sti_v2
import os
import json

app = FastAPI()

# =========================================================
# DATA DIR
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


@app.get("/")
def home():
    return {"message": "STI Prediction API running"}


@app.post("/predict")
def predict(data: dict):
    import time
    t0 = time.time()
    sti = predict_sti_v2(
        data["Sx"], data["Sy"],
        data["Rx"], data["Ry"]
    )
    t1 = time.time()
    return {
        "STI1": float(sti[0]),
        "STI2": float(sti[1]),
        "STI3": float(sti[2]),
        "inference_ms": (t1 - t0) * 1000
    }


@app.get("/frame")
def get_frame():
    """Serve latest camera frame as JPEG."""
    path = os.path.join(DATA_DIR, "frame.jpg")
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
            if data:
                return Response(content=data, media_type="image/jpeg")
    except:
        pass
    return Response(content=b"", status_code=404)


@app.get("/detection_data")
def get_detection_data():
    """Serve latest detection data as JSON."""
    path = os.path.join(DATA_DIR, "data.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                return json.loads(content)
    except:
        pass
    return {"Sx": None, "Sy": None, "receivers": []}