#!/usr/bin/env bash
set -euo pipefail

cd ~/pupil
source ~/pupil/Minki_pupilTest/bin/activate

export DISPLAY=:1
export QT_X11_NO_MITSHM=1

python pupil_gaze_yolo26_segfull_jetson.py \
  --view \
  --device 0 \
  --model yolo26n-seg.pt \
  --roi 224 \
  --imgsz 224 \
  --full-imgsz 384 \
  --conf 0.30 \
  --iou 0.45 \
  --gaze-radius 70