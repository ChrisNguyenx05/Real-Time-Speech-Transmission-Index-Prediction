import cv2
import numpy as np
import requests
import json
import os
import time

# =========================================================
# CONFIG
# =========================================================
API_URL = "http://127.0.0.1:8000/predict"
VIDEO_URL = "http://192.168.1.7:8081/video"

ROOM_WIDTH = 6.5
ROOM_HEIGHT = 4.0

GRID_ROWS = 8
GRID_COLS = 13
GRID_STEP = 0.5
GRID_OFFSET = 0.25

WARP_W = 1200
WARP_H = 800

SAVE_INTERVAL = 0.2

# =========================================================
# ARUCO CONFIG
# =========================================================
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Marker layout:
# 0 -------- 1
# | WORKSPACE|
# 3 -------- 2
REQUIRED_IDS = [0, 1, 2, 3]

# =========================================================
# PATH
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FRAME_PATH = os.path.join(DATA_DIR, "frame.jpg")
JSON_PATH = os.path.join(DATA_DIR, "data.json")

print("=" * 60)
print("CAMERA ENGINE")
print("DATA:", DATA_DIR)
print("=" * 60)

last_save_time = 0

# =========================================================
# STI CLASSIFICATION (IEC 60268-16)
# =========================================================
def classify_sti(sti):
    """Classify STI value per IEC 60268-16."""
    if sti > 0.76:
        return "Excellent", (105, 150, 5)   
    elif sti >= 0.65:
        return "High", (199, 132, 2)           
    elif sti >= 0.60:
        return "Good", (6, 119, 217)           
    else:
        return "Bad", (72, 29, 225)                       

# =========================================================
# SAVE IMAGE (Unicode-safe)
# =========================================================
def save_image(path, img):
    for attempt in range(3):
        try:
            ok, buf = cv2.imencode(
                ".jpg", img,
                [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                continue
            with open(path, "wb") as f:
                f.write(buf.tobytes())
            return True
        except Exception as e:
            print(f"SAVE IMAGE ERROR ({attempt+1}):", e)
        time.sleep(0.05)
    return False

# =========================================================
# SAVE JSON
# =========================================================
def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print("SAVE JSON ERROR:", e)
        return False

# =========================================================
# DRAW GRID
# =========================================================
def draw_real_grid(img):
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            real_x = GRID_OFFSET + j * GRID_STEP
            real_y = GRID_OFFSET + i * GRID_STEP
            px = int(real_x / ROOM_WIDTH * WARP_W)
            py = WARP_H - int(real_y / ROOM_HEIGHT * WARP_H)
            cv2.circle(img, (px, py), 2, (220, 220, 220), -1)
    return img

# =========================================================
# DRAW STI LEGEND
# =========================================================
def draw_legend(img):
    """Draw IEC 60268-16 legend on top-right corner."""
    x0, y0 = WARP_W - 280, 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + 270, y0 + 130), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + 270, y0 + 130), (100, 100, 100), 1)

    cv2.putText(img, "IEC 60268-16", (x0 + 10, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    items = [
        ("Excellent  > 0.76", (105, 150, 5)),
        ("High       0.65 - 0.76", (199, 132, 2)),
        ("Good       0.60 - 0.64", (6, 119, 217)),
        ("Bad        < 0.60", (72, 29, 225)),
    ]
    for i, (text, color) in enumerate(items):
        y = y0 + 45 + i * 22
        cv2.circle(img, (x0 + 18, y - 5), 5, color, -1)
        cv2.putText(img, text, (x0 + 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return img

# =========================================================
# ARUCO DETECTOR
# =========================================================
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# =========================================================
# ORDER POINTS
# =========================================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# =========================================================
# DETECT ARUCO BOARD
# =========================================================
def detect_aruco_board(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    display = frame.copy()

    if ids is None:
        return None, None, display

    ids = ids.flatten()
    cv2.aruco.drawDetectedMarkers(display, corners, ids)

    marker_dict = {}
    for i, marker_id in enumerate(ids):
        marker_dict[marker_id] = corners[i][0]

    for req_id in REQUIRED_IDS:
        if req_id not in marker_dict:
            return None, None, display

    # Inner workspace corners
    inner_top_left = marker_dict[0][2]
    inner_top_right = marker_dict[1][3]
    inner_bottom_right = marker_dict[2][0]
    inner_bottom_left = marker_dict[3][1]

    workspace_pts = np.array([
        inner_top_left, inner_top_right,
        inner_bottom_right, inner_bottom_left
    ], dtype=np.float32)

    cv2.polylines(display, [workspace_pts.astype(np.int32)],
                  True, (255, 0, 0), 3)
    cv2.putText(display, "WORKSPACE",
                (int(inner_top_left[0]), int(inner_top_left[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return workspace_pts, ids, display

# =========================================================
# WARP WORKSPACE
# =========================================================
def warp_workspace(frame, pts):
    rect = order_points(pts)
    dst = np.array([
        [0, 0], [WARP_W, 0],
        [WARP_W, WARP_H], [0, WARP_H]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (WARP_W, WARP_H))

# =========================================================
# DETECT OBJECTS
# =========================================================
def detect_objects(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # RED mask
    mask_red = (
        cv2.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255])) |
        cv2.inRange(hsv, np.array([170, 120, 80]), np.array([180, 255, 255]))
    )

    # YELLOW mask
    mask_yellow = cv2.inRange(
        hsv, np.array([15, 80, 80]), np.array([40, 255, 255])
    )

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

    # Source (red)
    S_pixel = None
    contours_red, _ = cv2.findContours(
        mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours_red:
        c = max(contours_red, key=cv2.contourArea)
        if cv2.contourArea(c) > 150:
            M = cv2.moments(c)
            if M["m00"] != 0:
                S_pixel = (
                    int(M["m10"] / M["m00"]),
                    int(M["m01"] / M["m00"])
                )

    # Receivers (yellow)
    receivers = []
    contours_yellow, _ = cv2.findContours(
        mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours_yellow:
        if cv2.contourArea(c) < 250:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        receivers.append((
            int(M["m10"] / M["m00"]),
            int(M["m01"] / M["m00"])
        ))

    # Remove duplicates
    unique = []
    for r in receivers:
        dup = False
        for u in unique:
            if np.sqrt((r[0]-u[0])**2 + (r[1]-u[1])**2) < 40:
                dup = True
                break
        if not dup:
            unique.append(r)

    return S_pixel, sorted(unique, key=lambda p: (p[1], p[0]))

# =========================================================
# COORDINATE HELPERS
# =========================================================
def pixel_to_real(px, py):
    return (px / WARP_W) * ROOM_WIDTH, ((WARP_H - py) / WARP_H) * ROOM_HEIGHT

def snap_grid(v):
    return round((v - GRID_OFFSET) / GRID_STEP) * GRID_STEP + GRID_OFFSET

# =========================================================
# API CALL
# =========================================================
def get_sti(Sx, Sy, Rx, Ry):
    try:
        r = requests.post(
            API_URL,
            json={"Sx": float(Sx), "Sy": float(Sy),
                  "Rx": float(Rx), "Ry": float(Ry)},
            timeout=1.0
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print("API ERROR:", e)
    return None

# =========================================================
# CAMERA
# =========================================================
cap = cv2.VideoCapture(VIDEO_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("CANNOT CONNECT CAMERA")
    exit()

print("CAMERA CONNECTED")

prev_time = time.time()
perf_history = []  # Log real performance

import atexit
import csv
def save_perf_log():
    if len(perf_history) > 0:
        with open(os.path.join(DATA_DIR, "perf_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fps", "cv_ms", "api_ms"])
            for p in perf_history:
                w.writerow([p["fps"], p["cv_ms"], p["api_ms"]])
        print(f"\n[INFO] Đã lưu log hiệu năng thực tế tại data/perf_log.csv ({len(perf_history)} frames).")

atexit.register(save_perf_log)

# =========================================================
# MAIN LOOP
# =========================================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("RECONNECT CAMERA...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(VIDEO_URL)
        continue

    frame_resized = cv2.resize(frame, (1280, 720))
    display = frame_resized.copy()

    t_frame_start = time.time()
    current_time = time.time()
    latency_ms = (current_time - prev_time) * 1000
    fps = 1000 / latency_ms if latency_ms > 0 else 0
    prev_time = current_time

    workspace_pts, ids, debug_display = detect_aruco_board(
        frame_resized
    )

    data = {
        "timestamp": time.time(),
        "Sx": None,
        "Sy": None,
        "receivers": []
    }

    # ----- BOARD DETECTED -----
    if workspace_pts is not None:

        warped = warp_workspace(frame_resized, workspace_pts)
        warped = draw_real_grid(warped)

        cv2.putText(warped, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw legend
        warped = draw_legend(warped)

        S_pixel, receivers = detect_objects(warped)
        
        cv_ms = (time.time() - t_frame_start) * 1000
        total_api_ms = 0

        # ----- SOURCE -----
        if S_pixel is not None:

            cv2.circle(warped, S_pixel, 12, (0, 0, 255), -1)
            cv2.circle(warped, S_pixel, 14, (255, 255, 255), 2)

            raw_Sx, raw_Sy = pixel_to_real(S_pixel[0], S_pixel[1])
            Sx, Sy = snap_grid(raw_Sx), snap_grid(raw_Sy)

            # Debug đo đạc sai số thực tế
            err_x = abs(raw_Sx - Sx)
            err_y = abs(raw_Sy - Sy)
            print(f"[MEASURE S] True/Snapped: ({Sx:.2f}, {Sy:.2f}) | Camera Raw: ({raw_Sx:.3f}, {raw_Sy:.3f}) | Error: {err_x*100:.1f}cm (X), {err_y*100:.1f}cm (Y)")

            data["Sx"] = float(Sx)
            data["Sy"] = float(Sy)

            cv2.putText(warped, f"S ({Sx:.2f}, {Sy:.2f})",
                        (S_pixel[0] + 18, S_pixel[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # ----- RECEIVERS -----
            for idx, (px, py) in enumerate(receivers):

                raw_Rx, raw_Ry = pixel_to_real(px, py)
                Rx, Ry = snap_grid(raw_Rx), snap_grid(raw_Ry)
                
                # Debug đo đạc sai số thực tế
                err_rx = abs(raw_Rx - Rx)
                err_ry = abs(raw_Ry - Ry)
                print(f"[MEASURE R{idx+1}] True/Snapped: ({Rx:.2f}, {Ry:.2f}) | Camera Raw: ({raw_Rx:.3f}, {raw_Ry:.3f}) | Error: {err_rx*100:.1f}cm (X), {err_ry*100:.1f}cm (Y)")

                result = get_sti(Sx, Sy, Rx, Ry)
                
                if result is None:
                    # Nếu không mở API, chỉ vẽ điểm vàng và tọa độ giống hệt y như Source
                    cv2.circle(warped, (px, py), 10, (0, 255, 255), -1)
                    cv2.circle(warped, (px, py), 12, (255, 255, 255), 2)

                    cv2.putText(warped, f"R{idx+1} ({Rx:.2f}, {Ry:.2f})",
                                (px + 18, py + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)
                    
                    # Push to streamlit with empty STI so it doesn't crash but shows dashes
                    data["receivers"].append({
                        "id": idx + 1,
                        "Rx": float(Rx),
                        "Ry": float(Ry),
                        "STIg": 0.0, "STIf": 0.0, "STIm": 0.0,
                        "rating_g": "-", "rating_f": "-", "rating_m": "-"
                    })
                    continue

                # KHI CÓ API (ONLINE)
                total_api_ms += result.get("inference_ms", 0)

                stig = result.get("STI1", 0)
                stif = result.get("STI2", 0)
                stim = result.get("STI3", 0)

                rate_g, col_g = classify_sti(stig)
                rate_f, col_f = classify_sti(stif)
                rate_m, col_m = classify_sti(stim)

                # Marker color based on worst rating
                worst_color = col_g
                for c in [col_f, col_m]:
                    if c == (72, 29, 225):
                        worst_color = c
                        break

                # Draw receiver marker
                cv2.circle(warped, (px, py), 10, worst_color, -1)
                cv2.circle(warped, (px, py), 12, (255, 255, 255), 2)

                # Label: "R1 (x, y)" (Giảm viền đen)
                label_text = f"R{idx+1} ({Rx:.2f}, {Ry:.2f})"
                cv2.putText(warped, label_text, (px + 18, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Viền đen mỏng hơn
                cv2.putText(warped, label_text, (px + 18, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Chữ chính

                # Các text STI (Làm nổi bật màu, giảm viền đen)
                sti_texts = [
                    (f"STIg: {stig:.3f} [{rate_g}]", col_g, py + 5),
                    (f"STIf: {stif:.3f} [{rate_f}]", col_f, py + 22),
                    (f"STIm: {stim:.3f} [{rate_m}]", col_m, py + 39)
                ]

                for text, color, y_pos in sti_texts:
                    # 1. Vẽ viền chữ màu Đen (Độ dày = 2, giảm đi so với 3)
                    cv2.putText(warped, text, (px + 18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                    # 2. Vẽ chữ màu chính đè lên trên (Tăng size lên 0.45 để nổi màu hơn)
                    cv2.putText(warped, text, (px + 18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                data["receivers"].append({
                    "id": idx + 1,
                    "Rx": float(Rx),
                    "Ry": float(Ry),
                    "STIg": float(stig),
                    "STIf": float(stif),
                    "STIm": float(stim),
                    "rating_g": rate_g,
                    "rating_f": rate_f,
                    "rating_m": rate_m
                })

        # Lưu log hiệu năng thật
        perf_history.append({"fps": fps, "cv_ms": cv_ms, "api_ms": total_api_ms})

        display = warped

    else:
        display = debug_display
        cv2.putText(display, "ARUCO BOARD NOT DETECTED",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # ----- SAVE -----
    now = time.time()
    if now - last_save_time > SAVE_INTERVAL:
        save_image(FRAME_PATH, display)
        save_json(JSON_PATH, data)
        last_save_time = now

    # ----- LOCAL DISPLAY -----
    cv2.imshow("Engine", display)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()