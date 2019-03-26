#!/usr/bin/env sh
# Create tfrecords.

export PYTHONPATH=$(dirname "$PWD")
export FORGD_VIDEO_DIR_PATH="$HOME/Datasets/UCF101/videos_foreground"
export BACKD_VIDEO_DIR_PATH="$HOME/Datasets/UCF101/videos_background"
export TFRECORD_PATH="../resources/ucf101.tfrecord"
 
python3.6 ../create_dataset.py -t $TFRECORD_PATH -a "train" -f $FORGD_VIDEO_DIR_PATH -b $BACKD_VIDEO_DIR_PATH
python3.6 ../create_dataset.py -t $TFRECORD_PATH -a "test" -f $FORGD_VIDEO_DIR_PATH -b $BACKD_VIDEO_DIR_PATH
python3.6 ../create_dataset.py -t $TFRECORD_PATH -a "validation" -f $FORGD_VIDEO_DIR_PATH -b $BACKD_VIDEO_DIR_PATH
