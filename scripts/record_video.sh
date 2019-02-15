#!/usr/bin/env sh
# Record a video.

export PYTHONPATH=$(dirname "$PWD")
export VIDEO=../resources/output.avi

python3.6 ../video_recorder.py -o $VIDEO
# python3.6 ../video_recorder.py
