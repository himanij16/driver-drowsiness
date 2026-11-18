"""Streamlit demo.  streamlit run app.py"""
import cv2
import pandas as pd
import streamlit as st
import config as C
from pipeline import Pipeline

st.set_page_config(page_title="Driver Drowsiness Detector", layout="wide")
st.title("Driver Drowsiness Detector")
with st.sidebar:
    run = st.toggle("Start camera")
    cam = st.number_input("Camera index", 0, 5, C.CAMERA_INDEX)
    use_audio = st.checkbox("Cabin audio analysis", True)
    use_ble = st.checkbox("BLE smartwatch cross-validation", False)
    st.caption("Sit normally for the first 45 s while the system calibrates.")

c1, c2 = st.columns([3, 2])
video, banner, metrics, chart = c1.empty(), c2.empty(), c2.empty(), c2.empty()

if run:
    pipe = Pipeline(use_audio=use_audio, use_ble=use_ble)
    cap = cv2.VideoCapture(int(cam))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
    hist = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                st.error("Cannot read camera"); break
            out, s = pipe.process(frame)
            video.image(out[:, :, ::-1], channels="RGB", use_container_width=True)
            if s:
                lvl = s["level"]
                if lvl == "ALARM": banner.error(f"ALARM - drowsiness score {s['score']:.2f}")
                elif lvl == "WARN": banner.warning(f"Warning - score {s['score']:.2f}")
                elif lvl == "CALIBRATING": banner.info(f"Calibrating... {int(s['calib_progress']*100)}%")
                else: banner.success(f"Alert - score {s['score']:.2f}")
                f = lambda v, p="{:.1f}": "--" if v is None else p.format(v)
                metrics.markdown(
                    f"**PERCLOS** {f(s['perclos'], '{:.2f}')}  |  **Yawns/min** {f(s['yawn_freq'])}  |  "
                    f"**Nods/min** {f(s['nod_freq'])}\n\n"
                    f"**HR (rPPG)** {f(s['hr_rppg'], '{:.0f}')}  |  **HR (BLE)** {f(s['hr_ble'], '{:.0f}')}  |  "
                    f"**Resp/min** {f(s['resp_bpm'], '{:.0f}')}\n\n"
                    f"**Posture dev** {f(s['posture_dev'], '{:.2f}')}  |  **Audio idx** {f(s['audio_idx'], '{:.2f}')}")
                hist.append({"score": s["score"]}); hist = hist[-300:]
                if len(hist) % 5 == 0:
                    chart.line_chart(pd.DataFrame(hist), height=200)
    finally:
        cap.release(); pipe.close()
else:
    st.info("Toggle 'Start camera' in the sidebar.")
