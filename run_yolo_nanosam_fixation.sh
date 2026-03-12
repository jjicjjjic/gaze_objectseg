#!/usr/bin/env bash
set -e

VENV="$HOME/pupil/Minki_pupilTest"
APP_DIR="$HOME/pupil"
NANOSAM_REPO="$HOME/pupil/nanosam"

export DISPLAY=":1"
source "$VENV/bin/activate"
export PYTHONPATH="$NANOSAM_REPO:$PYTHONPATH"

cd "$APP_DIR"

"$VENV/bin/python" pupil_gaze_fixation_yolo26_nanosam_jetson.py \
  --view \
  --conflate \
  --half \
  --imgsz 224 \
  --seed-roi 220 \
  --gaze-radius 120 \
  --lock-radius 60 \
  --lock-duration 1.0 \
  --unlock-radius 100 \
  --unlock-duration 0.8 \
  --object-expand-scale 2.2 \
  --object-margin 24 \
  --selected-keep-margin 36 \
  --selected-timeout 1.0 \
  --refine-size 96 \
  --nanosam-cache-ttl 10.0