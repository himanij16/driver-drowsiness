"""Ties every module together: acquisition -> feature extraction -> fusion -> alert -> CSV log."""
import csv
import os
import time
import cv2
import numpy as np
import config as C
from modules.face_features import FaceAnalyzer
from modules.posture import PostureAnalyzer
from modules.rppg import RPPG
from modules.respiration import Respiration
from modules.fusion import Fusion
from modules.alerts import AlertManager

LOG_FIELDS = ["time", "label", "ear", "mar", "iris_ratio", "pitch", "perclos", "blink_rate",
              "blink_vel", "yawn_freq", "nod_freq", "closed_sec", "posture_dev", "audio_idx",
              "audio_tempo", "audio_db", "hr_rppg", "hr_ble", "rmssd_rppg", "rmssd_ble",
              "hrv_dev", "resp_bpm", "rppg_quality", "score", "level"]
COLORS = {"OK": (0, 200, 0), "WARN": (0, 200, 255), "ALARM": (0, 0, 255)}


class Pipeline:
    def __init__(self, use_audio=True, use_ble=False, log=True):
        self.face, self.posture = FaceAnalyzer(), PostureAnalyzer()
        self.rppg, self.resp = RPPG(), Respiration()
        self.fusion, self.alerts = Fusion(), AlertManager()
        self.audio = self.ble = None
        if use_audio:
            from modules.audio_features import AudioMonitor
            self.audio = AudioMonitor(); self.audio.start()
        if use_ble:
            from modules.ble_hr import BLEHeartRate
            self.ble = BLEHeartRate(); self.ble.start()
        self.log_file = self.writer = None
        if log:
            os.makedirs("logs", exist_ok=True)
            self.log_file = open(time.strftime("logs/session_%Y%m%d_%H%M%S.csv"), "w", newline="")
            self.writer = csv.DictWriter(self.log_file, LOG_FIELDS)
            self.writer.writeheader()
        self.frame_i = 0
        self.state = {}
        self.recalibrate()

    # ---------- calibration ----------
    def recalibrate(self):
        self.calibrated = False
        self.calib_start = time.time()
        self.calib = {"ear": [], "pitch": [], "ratio": [], "tilt": []}
        self.hrv_samples = {"rppg": [], "ble": []}
        self.hrv_base = {"rppg": None, "ble": None}
        self.posture_res, self.last_tick = {}, 0.0
        self.resp_bpm = None
        self.rppg.reset()
        self.fusion.reset()

    def _finish_calibration(self):
        med = lambda v, d: float(np.median(v)) if len(v) >= 10 else d
        self.face.set_baseline(med(self.calib["ear"], C.EAR_DEFAULT_OPEN), med(self.calib["pitch"], 0.0))
        if len(self.calib["ratio"]) >= 10:
            self.posture.set_baseline(med(self.calib["ratio"], 0), med(self.calib["tilt"], 0))
        for k, v in self.hrv_samples.items():
            if len(v) >= 5:
                self.hrv_base[k] = float(np.median(v))
        self.face.reset_stats()
        self.fusion.reset()
        self.calibrated = True

    # ---------- per frame ----------
    def process(self, frame, label=None):
        now = time.time()
        self.frame_i += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = self.face.process(frame, rgb, now)
        if self.frame_i % 3 == 0 or not self.posture_res:
            self.posture_res = self.posture.process(rgb)
        pr = self.posture_res

        if face["found"]:
            self.rppg.add(frame, face["rois"], now)
        if pr.get("found") and pr["chest_roi"]:
            self.resp.add(frame, pr["chest_roi"], now)

        if not self.calibrated:
            if face["found"]:
                self.calib["ear"].append(face["ear"]); self.calib["pitch"].append(face["pitch"])
            if pr.get("found"):
                self.calib["ratio"].append(pr["ratio"]); self.calib["tilt"].append(pr["tilt"])

        if now - self.last_tick >= 1.0:
            self.last_tick = now
            self._tick(face, pr, now, label)
        return self._draw(frame, face), self.state

    # ---------- 1 Hz fusion step ----------
    def _tick(self, face, pr, now, label):
        rp = self.rppg.estimate()
        rb = self.resp.estimate()
        if rb: self.resp_bpm = rb
        au = self.audio.latest() if self.audio else None
        ble = self.ble.latest() if self.ble else None

        cur = {"rppg": rp["rmssd"] if rp else None, "ble": ble["rmssd"] if ble else None}
        for k, v in cur.items():                       # collect HRV baselines
            if v is not None and self.hrv_base[k] is None and len(self.hrv_samples[k]) < 30:
                self.hrv_samples[k].append(v)
                if self.calibrated and len(self.hrv_samples[k]) >= 5:
                    self.hrv_base[k] = float(np.median(self.hrv_samples[k]))
        if not self.calibrated and now - self.calib_start >= C.CALIB_SEC:
            self._finish_calibration()

        src = "ble" if cur["ble"] is not None else "rppg" if cur["rppg"] is not None else None
        hrv_dev = None
        if self.calibrated and src and self.hrv_base[src]:
            hrv_dev = max(0.0, (self.hrv_base[src] - cur[src]) / self.hrv_base[src])

        feats = {
            "perclos": face.get("perclos"), "yawn_freq": face.get("yawn_freq"),
            "nod_freq": face.get("nod_freq"), "closed_sec": face.get("closed_sec"),
            "posture_dev": pr.get("deviation") if pr.get("found") else None,
            "audio_idx": au["index"] if au else None, "hrv_dev": hrv_dev}
        fz = self.fusion.score(feats) if face["found"] else {"score": 0.0, "level": "OK", "contrib": {}}
        if not self.calibrated:
            fz["level"] = "CALIBRATING"
        else:
            self.alerts.trigger(fz["level"])

        self.state = {
            "calibrating": not self.calibrated,
            "calib_progress": min(1.0, (now - self.calib_start) / C.CALIB_SEC),
            "face": face["found"], "ear": face.get("ear"), "perclos": face.get("perclos"),
            "yawn_freq": face.get("yawn_freq"), "nod_freq": face.get("nod_freq"),
            "pitch": face.get("pitch"), "posture_dev": feats["posture_dev"],
            "audio_idx": feats["audio_idx"], "hr_rppg": rp["hr"] if rp else None,
            "hr_ble": ble["hr"] if ble else None, "hrv_dev": hrv_dev,
            "resp_bpm": self.resp_bpm, "score": fz["score"], "level": fz["level"],
            "contrib": fz["contrib"]}
        if self.writer and face["found"]:
            row = {
                "time": round(now, 2), "label": label, "ear": face["ear"], "mar": face["mar"],
                "iris_ratio": face["iris_ratio"], "pitch": face["pitch"], "perclos": face["perclos"],
                "blink_rate": face["blink_rate"], "blink_vel": face["blink_vel"],
                "yawn_freq": face["yawn_freq"], "nod_freq": face["nod_freq"],
                "closed_sec": face["closed_sec"], "posture_dev": feats["posture_dev"],
                "audio_idx": feats["audio_idx"], "audio_tempo": au["tempo"] if au else None,
                "audio_db": au["rms_db"] if au else None, "hr_rppg": rp["hr"] if rp else None,
                "hr_ble": ble["hr"] if ble else None, "rmssd_rppg": cur["rppg"], "rmssd_ble": cur["ble"],
                "hrv_dev": hrv_dev, "resp_bpm": self.resp_bpm,
                "rppg_quality": rp["quality"] if rp else None, "score": fz["score"], "level": fz["level"]}
            self.writer.writerow(row); self.log_file.flush()

    # ---------- overlay ----------
    def _draw(self, frame, face):
        out = frame.copy()
        s = self.state
        if face["found"]:
            for (x1, y1, x2, y2) in face["rois"]:
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 1)
        if not s:
            return out
        lvl = s["level"]
        col = COLORS.get(lvl, (200, 200, 200))
        def put(txt, y, c=(255, 255, 255)):
            cv2.putText(out, txt, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        fmt = lambda v, f="{:.2f}": "--" if v is None else f.format(v)
        if s["calibrating"]:
            put(f"CALIBRATING {int(s['calib_progress']*100)}% - sit normally, look ahead", 30, col)
        else:
            put(f"{lvl}  score {s['score']:.2f}", 30, col)
        put(f"EAR {fmt(s['ear'])}  PERCLOS {fmt(s['perclos'])}", 60)
        put(f"HR rPPG {fmt(s['hr_rppg'], '{:.0f}')}  BLE {fmt(s['hr_ble'], '{:.0f}')}  Resp {fmt(s['resp_bpm'], '{:.0f}')}", 90)
        put(f"Yawn/min {fmt(s['yawn_freq'], '{:.1f}')}  Nod/min {fmt(s['nod_freq'], '{:.1f}')}", 120)
        if lvl == "ALARM":
            cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 255), 12)
            cv2.putText(out, "WAKE UP! TAKE A BREAK", (out.shape[1] // 2 - 230, out.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        return out

    def close(self):
        if self.audio: self.audio.stop()
        if self.ble: self.ble.stop()
        if self.log_file: self.log_file.close()
