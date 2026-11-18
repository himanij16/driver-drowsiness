"""Desktop runner.  python main.py [--camera 0] [--no-audio] [--ble]
Keys: q quit | c recalibrate | d label 'drowsy'(1) | a label 'alert'(0) | n clear label"""
import argparse
import cv2
import config as C
from pipeline import Pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=C.CAMERA_INDEX)
ap.add_argument("--no-audio", action="store_true")
ap.add_argument("--ble", action="store_true", help="cross-validate HR with a BLE smartwatch")
args = ap.parse_args()

cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)
pipe = Pipeline(use_audio=not args.no_audio, use_ble=args.ble)
label = None
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, state = pipe.process(frame, label)
        cv2.imshow("Driver Drowsiness Detector", out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        elif k == ord("c"): pipe.recalibrate()
        elif k == ord("d"): label = 1
        elif k == ord("a"): label = 0
        elif k == ord("n"): label = None
finally:
    cap.release(); cv2.destroyAllWindows(); pipe.close()
