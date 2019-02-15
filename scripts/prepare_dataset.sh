#!/usr/bin/env sh
# Prepare dataset to train/test video classifier.
export CURRENTDIR=$PWD
export PYTHONPATH=$(dirname "$PWD")
export DATASETDIR="$HOME/Datasets/UCF101"
export VIDEODIR="$DATASETDIR/videos"
export DATASETURI="http://crcv.ucf.edu/data/UCF101/UCF101.rar"

# Download dataset.
cd $DATASETDIR
wget $DATASETURI

# Extract videos.
mkdir -p $VIDEODIR
rar e UCF101.rar $VIDEODIR
 
# Organize dataset.
cd $CURRENTDIR
python3.6 ../prepare_dataset.py -i $VIDEODIR -o $DATASETDIR
