# Real-time Speech Intelligibility (STI) Monitoring System

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV-orange) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-red) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green) ![Backend](https://img.shields.io/badge/Backend-FastAPI-009688) ![Frontend](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)

The system utilizes Computer Vision to detect and track the positions of the sound source and receivers on a physical scale model. It then predicts the Speech Transmission Index (STI) using Deep Learning and visualizes the results in real-time according to the IEC 60268-16 international standard.

---

## Project Structure

```
NCKH_/
├── backend/          # FastAPI prediction server
│   ├── api.py        # API endpoints (/predict, /frame, /detection_data)
│   ├── inference.py  # STI inference pipeline (Surrogate + TwoBranchRayNet)
│   └── model_df.py   # Neural network architecture definition
│
├── engine/           # Camera processing engine
│   └── camera_engine.py  # ArUco detection, HSV segmentation, coordinate mapping
│
├── frontend/         # Streamlit dashboard
│   └── app_api_camera.py # Real-time STI monitoring UI
│
├── training/         # Neural network & surrogate model training notebooks
│   └── TwoBranchedRayNet.ipynb # Full pipeline (XGBoost + TwoBranchRayNet)
│
├── model/            # Pre-trained ML artifacts
│   ├── model.pth             # TwoBranchRayNet weights
│   ├── scaler_X.pkl          # Input feature scaler
│   ├── scaler_y.pkl          # Output target scaler
│   └── surrogate_model.pkl   # Surrogate model (84 ray features)
│
├── data/             # Runtime data (auto-generated)
│   ├── data.json     # Latest detection data (JSON)
│   └── frame.jpg     # Latest camera frame
│
```

---

## How to Run

### 1. Start Backend API
```bash
python -m uvicorn backend.api:app --reload
```

### 2. Start Camera Engine
```bash
python -m engine.camera_engine
```

### 3. Start Streamlit Dashboard
```bash
python -m streamlit run frontend/app_api_camera.py
```

---

## Model Performance Metrics
- **TwoBranchRayNet (STI Prediction):** R² = 0.985 | MAE = 0.0035 | RMSE = 0.0045 (on Test set).
- **Surrogate Model (XGBoost):** Synthesizes 84 complex acoustic ray features from just 5 spatial inputs with MAE = 0.204.
- **End-to-End System:** Stable latency at ~0.9 FPS on CPU (no GPU inference), fully satisfying the requirements for quasi-real-time monitoring.

---

## Hardware Requirements
- IP Camera (or Smartphone + IP Camera Lite app)
- Physical scale model with 4 ArUco markers (ID 0-3) placed at the corners
- Marked objects: Red (Source), Yellow (Receivers)
