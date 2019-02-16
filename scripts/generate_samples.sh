#!/usr/bin/env sh
# Generate random samples.

export PYTHONPATH=$(dirname "$PWD")
export FORGD_VIDEO_DIR_PATH="$HOME/Datasets/UCF101/videos_foreground"
export BACKD_VIDEO_DIR_PATH="$HOME/Datasets/UCF101/videos_background"
 
python3.6 ../data_generator.py -f $FORGD_VIDEO_DIR_PATH -b $BACKD_VIDEO_DIR_PATH 
