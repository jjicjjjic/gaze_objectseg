#!/usr/bin/env bash
set -e

VENV="$HOME/pupil/Minki_pupilTest"
APP_DIR="$HOME/pupil"
NANOSAM_REPO="$HOME/pupil/nanosam"

export DISPLAY=":1"
source "$VENV/bin/activate"
export PYTHONPATH="$NANOSAM_REPO:$PYTHONPATH"

cd "$APP_DIR"

"$VENV/bin/python" pupil_gaze_yolo26_nanosam_jetson.py \
  --view \
  --conflate \
  --half \
  --roi 160 \
  --imgsz 160 \
  --refine-size 96 \
  --nanosam-only-on-fix \
  --nanosam-cooldown 1.5 \
  --nanosam-cache-ttl 1.0 \
  --stable-hit-min 3