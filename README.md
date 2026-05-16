# STI Real-time Monitoring System
## Hệ thống giám sát chỉ số STI (Speech Transmission Index) thời gian thực

Hệ thống sử dụng Computer Vision để nhận diện và theo dõi vị trí nguồn âm (speaker) và các điểm thu (receivers) trên sa bàn, sau đó dự đoán chỉ số STI bằng Deep Learning và trực quan hóa kết quả theo chuẩn IEC 60268-16.

---

## Cấu trúc Project

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
│   ├── frame.jpg     # Latest camera frame
│   └── perf_log.csv  # Performance log (FPS, latency)
│
├── markers/          # ArUco marker images (ID 0-3)
│
├── scripts/          # Utility & visualization scripts
│   ├── draw_grid_snapping.py      # Grid Snapping diagram
│   ├── draw_error_correction.py   # Error correction illustration
│   ├── draw_performance.py        # FPS & Latency charts
│   └── generate_aruco_board.py    # Generate printable ArUco board
│
└── docs/             # Research report & figures
    ├── Section_3.5_VI.md   # Report (Vietnamese)
    ├── Section_3.5_EN.md   # Report (English)
    └── figures/            # All report figures
```

---

## Hướng dẫn chạy

### 1. Khởi động Backend API
```bash
python -m uvicorn backend.api:app --reload
```

### 2. Khởi động Camera Engine
```bash
python -m engine.camera_engine
```

### 3. Khởi động Streamlit Dashboard
```bash
python -m streamlit run frontend/app_api_camera.py
```

---

## Hiệu năng Mô hình (Model Metrics)
- **TwoBranchRayNet (STI Prediction):** R² = 0.985 | MAE = 0.0035 | RMSE = 0.0045 (trên tập Test).
- **Surrogate Model (XGBoost):** Dự đoán 84 biến tia âm phức tạp chỉ từ 5 biến không gian cơ bản với MAE = 0.204.
- **Hệ thống (End-to-End):** Độ trễ ổn định ở mức ~0.9 FPS trên CPU (không dùng GPU inference), đáp ứng tốt yêu cầu giám sát cận thời gian thực.

---

## Yêu cầu phần cứng
- Camera IP (hoặc điện thoại + IP Camera Lite app)
- Sa bàn với 4 mã ArUco (ID 0-3) ở 4 góc
- Vật thể đánh dấu: Đỏ (Source), Vàng (Receivers)
