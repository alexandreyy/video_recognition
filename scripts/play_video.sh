#!/usr/bin/env sh
# Play video.

export PYTHONPATH=$(dirname "$PWD")
export VIDEO=../resources/demo_1.avi

python3.6 ../video_player.py -i $VIDEO
# python3.6 ../video_player.py
