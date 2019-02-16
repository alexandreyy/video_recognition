#!/usr/bin/env sh
# Prepare dataset to train/test video classifier.

export PYTHONPATH=$(dirname "$PWD")
export CURRENT_DIR=$PWD
export DATASET_DIR="$HOME/Datasets/UCF101"
export DATASET_URI="http://crcv.ucf.edu/data/UCF101/UCF101.rar"
export VIDEO_DIR="$DATASETDIR/videos_foreground"

# Download dataset.
cd $DATASET_DIR
wget $DATASET_URI

# Extract videos.
mkdir -p $VIDEO_DIR
rar e UCF101.rar $VIDEO_DIR
 
# Organize dataset.
cd $CURRENT_DIR
python3.6 ../prepare_dataset.py -i $VIDEO_DIR -o $DATASET_DIR
