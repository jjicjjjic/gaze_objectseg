#!/usr/bin/env bash
set -e

VENV="$HOME/pupil/Minki_pupilTest"
PUPIL_SRC="$HOME/pupil/pupil_src"

export DISPLAY=":1"
source "$VENV/bin/activate"

cd "$PUPIL_SRC"
python main.py capture
