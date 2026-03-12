#!/usr/bin/env bash
set -euo pipefail

cd ~/pupil
source ~/pupil/Minki_pupilTest/bin/activate
export DISPLAY=:1

python pupil_gaze_yolo26_seg_jetson.py \
  --view \
  --half \
  --device 0 \
  --model yolo26n-seg.pt \
  --roi 256 \
  --imgsz 256 \
  --conf 0.25 \
  --iou 0.45 \
  --gaze-radius 80