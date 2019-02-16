#!/usr/bin/env sh
# Prepare dataset to train/test video classifier.

export PYTHONPATH=$(dirname "$PWD")
export CURRENT_DIR=$PWD
export DATASET_DIR="$HOME/Datasets/UCF101"
export DATASET_1_URI="http://crcv.ucf.edu/data/UCF101/UCF101.rar"
export DATASET_2_URI="https://storage.googleapis.com/thumos14_files/TH14_background_set_mp4.zip"
export VIDEO_1_DIR="$DATASET_DIR/videos_foreground"
export VIDEO_2_DIR="$DATASET_DIR/videos_background"

# Download dataset 1.
# cd $DATASET_DIR
# wget $DATASET_1_URI
# mkdir -p $VIDEO_1_DIR
# rar e UCF101.rar $VIDEO_1_DIR
 
# Download dataset 2.
cd $DATASET_DIR
wget $DATASET_2_URI
mkdir -p $VIDEO_2_DIR
echo $VIDEO_2_DIR
unzip TH14_background_set_mp4.zip

# Organize dataset.
cd $CURRENT_DIR
python3.6 ../prepare_dataset.py -i $VIDEO_1_DIR
