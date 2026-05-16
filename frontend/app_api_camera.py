import streamlit as st
import requests
import time
import pandas as pd
from PIL import Image
from io import BytesIO

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="STI Realtime Dashboard", layout="wide")

API_BASE = "http://127.0.0.1:8000"

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stDataFrame { font-size: 14px; }

    /* Colors exactly match OpenCV BGR: 
       Excellent: BGR(105, 150, 5) -> RGB(5, 150, 105) -> #059669
       High: BGR(199, 132, 2) -> RGB(2, 132, 199) -> #0284C7
       Good: BGR(6, 119, 217) -> RGB(217, 119, 6) -> #D97706
       Bad: BGR(72, 29, 225) -> RGB(225, 29, 72) -> #E11D48
    */
    .sti-excellent { color: #059669; font-weight: bold; }
    .sti-high { color: #0284C7; font-weight: bold; }
    .sti-good { color: #D97706; font-weight: bold; }
    .sti-bad { color: #E11D48; font-weight: bold; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #30475e;
    }
    .metric-card h4 {
        color: #e0e0e0;
        margin: 0 0 8px 0;
        font-size: 14px;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 22px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADERS (via API)
# =========================================================
def load_frame():
    try:
        r = requests.get(f"{API_BASE}/frame", timeout=2)
        if r.status_code == 200 and r.content:
            return Image.open(BytesIO(r.content))
    except:
        pass
    return None

def load_data():
    try:
        r = requests.get(f"{API_BASE}/detection_data", timeout=2)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

# =========================================================
# STI CLASSIFICATION
# =========================================================
def classify_sti(sti):
    if sti > 0.76:
        return "Excellent"
    elif sti >= 0.65:
        return "High"
    elif sti >= 0.60:
        return "Good"
    else:
        return "Bad"

def rating_html(rating):
    css_class = f"sti-{rating.lower()}"
    return f'<span class="{css_class}">{rating}</span>'

# =========================================================
# DASHBOARD
# =========================================================
st.markdown("<h1 style='text-align: left; color: #ffffff; margin-bottom: 20px;'>Realtime STI Multiple Receivers</h1>", unsafe_allow_html=True)

data = load_data()

# Layout: Camera [3] | Info [1]
cam_col, info_col = st.columns([3, 1], gap="medium")

# ----- CAMERA FEED -----
with cam_col:
    frame = load_frame()
    if frame is not None:
        st.image(frame, width='stretch')
    else:
        st.warning("⏳ Waiting for camera feed...")

# ----- INFO PANEL -----
with info_col:

    # Source info
    st.markdown("#### 🔴 Source")
    Sx = data.get("Sx") if data else None
    Sy = data.get("Sy") if data else None

    if Sx is not None:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Position</h4>
            <div class="value">({Sx:.2f}, {Sy:.2f})</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No source detected")

    st.markdown("---")

    # Receivers table
    st.markdown("#### 🟡 Receivers")

    receivers = data.get("receivers", []) if data else []

    if len(receivers) > 0:

        rows = []
        for r in receivers:
            stig = r.get("STIg", r.get("STI1", 0))
            stif = r.get("STIf", r.get("STI2", 0))
            stim = r.get("STIm", r.get("STI3", 0))
            rows.append({
                "ID": f"R{r['id']}",
                "Rx": round(r["Rx"], 2),
                "Ry": round(r["Ry"], 2),
                "STIg": round(stig, 3),
                "Rate_g": r.get("rating_g", classify_sti(stig)),
                "STIf": round(stif, 3),
                "Rate_f": r.get("rating_f", classify_sti(stif)),
                "STIm": round(stim, 3),
                "Rate_m": r.get("rating_m", classify_sti(stim)),
            })

        df = pd.DataFrame(rows)

        def style_rating(val):
            colors = {
                "Excellent": "#059669",
                "High": "#0284C7",
                "Good": "#D97706",
                "Bad": "#E11D48"
            }
            c = colors.get(val, "#FFFFFF")
            return f"color: {c}; font-weight: bold"

        styled = df.style.map(
            style_rating,
            subset=["Rate_g", "Rate_f", "Rate_m"]
        )

        st.dataframe(
            styled,
            width='stretch',
            hide_index=True,
            height=min(len(rows) * 40 + 40, 300)
        )
    else:
        st.info("No receivers detected")

    st.markdown("---")

    # IEC 60268-16 Reference
    st.markdown("#### 📊 IEC 60268-16")

    ref_data = {
        "Rating": ["Excellent", "High", "Good", "Bad"],
        "STI Range": ["> 0.76", "0.65 – 0.76", "0.60 – 0.64", "< 0.60"]
    }
    ref_df = pd.DataFrame(ref_data)

    def style_ref(val):
        colors = {
            "Excellent": "#059669",
            "High": "#0284C7",
            "Good": "#D97706",
            "Bad": "#E11D48"
        }
        c = colors.get(val, "#FFFFFF")
        return f"color: {c}; font-weight: bold"

    styled_ref = ref_df.style.map(
        style_ref, subset=["Rating"]
    )

    st.dataframe(
        styled_ref,
        width='stretch',
        hide_index=True,
        height=200
    )

# =========================================================
# AUTO REFRESH
# =========================================================
time.sleep(0.3)
st.rerun()