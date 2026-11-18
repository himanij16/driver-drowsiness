# Driver Drowsiness Detector (multi-modal, contactless)

MCA synopsis project: fuses **eye closure (PERCLOS), yawning, head nods, posture, cabin-audio
fatigue index and HRV (rPPG, cross-checked with a BLE smartwatch)** into one drowsiness score
(Equation 5.1 of the synopsis).

## Folder map (what each file is)
| File | Purpose | Synopsis link |
|---|---|---|
| `config.py` | All thresholds, weights w1..w6, camera/audio settings | Ch. 5.1 |
| `modules/face_features.py` | MediaPipe Face Mesh: EAR, PERCLOS, blink rate/velocity, yawn, head pose, nods, iris ratio, skin ROIs | Obj. 3 |
| `modules/posture.py` | MediaPipe Pose: posture deviation from baseline, chest ROI | Obj. 3 |
| `modules/rppg.py` | POS rPPG -> heart rate + RMSSD (HRV) | Obj. 2 |
| `modules/respiration.py` | Optical-flow respiration rate | Intro |
| `modules/ble_hr.py` | `bleak` BLE Heart Rate Service (0x180D) client | Obj. 2 |
| `modules/audio_features.py` | librosa tempo / RMS / centroid -> AudioFatigueIndex | Obj. 4 |
| `modules/fusion.py` | Weighted fusion, EMA smoothing, thresholds, safety overrides | Obj. 5 |
| `modules/alerts.py` | Visual + audible alarm | Ch. 5 step 8 |
| `modules/dsp.py` | Band-pass + spectral peak helpers | - |
| `pipeline.py` | Orchestrates everything, calibration, 1 Hz fusion tick, CSV logging | Ch. 5.2 |
| `main.py` | OpenCV desktop app (keys for labelling data) | - |
| `app.py` | Streamlit demo UI | Ch. 4 |
| `evaluation/ablation.py` | Ablation vs PERCLOS-only baseline, learns weights | Obj. 5 |
| `evaluation/hr_validation.py` | rPPG vs smartwatch: MAE, RMSE, r, Bland-Altman | Obj. 2 |
| `tests/test_modules.py` | Unit tests (synthetic rPPG, fusion, BLE parser, audio) | - |

## Build order (step by step)
1. **Environment**: Python 3.10/3.11. `python -m venv venv`, activate, `pip install -r requirements.txt`.
2. **Test the maths first** (no camera needed): `pytest tests -q`.
3. **Camera + face features**: `python main.py --no-audio`. Check EAR/PERCLOS react when you close your eyes.
4. **Calibration**: stay still and alert for 45 s (press `c` to redo). It sets your open-eye EAR, head pitch, posture and HRV baselines.
5. **rPPG**: use steady, front-facing light, stay still; HR appears after about 10-15 s. Compare with a watch.
6. **Audio**: play music in the room and watch `audio_idx` change.
7. **BLE watch**: enable "broadcast heart rate" on the watch/band (Apple Watch does not expose standard BLE HR; chest straps and many Garmin/Amazfit/Xiaomi bands do), then `python main.py --ble`.
8. **Collect labelled data**: press `a` while alert, `d` while acting/being drowsy (e.g. late-night, tired, or simulated). Logs go to `logs/session_*.csv`. Record 5+ sessions, ideally several people.
9. **Evaluate**: `python -m evaluation.ablation` and `python -m evaluation.hr_validation`.
10. **Tune weights**: copy learned weights into `config.WEIGHTS`, re-run.
11. **Demo**: `streamlit run app.py`.

## Honest limitations (put these in your report)
- Iris diameter is only a *proxy* for pupil size; true pupillometry needs IR imaging.
- rPPG HRV (RMSSD) at 30 fps is noisy; trends vs. your own baseline are more reliable than absolute values. Hence the BLE cross-check.
- The audio index is a heuristic (slow/quiet/dull = less stimulation). Treat its weight as something to validate, not assume.
- Head-nod detection uses absolute pitch deviation (sign of pitch is ambiguous from monocular PnP).
- Collect data with seated volunteers, not while actually driving. This is a research prototype, not a safety-certified product.
- Get consent from participants for video/audio recording.
