#!/usr/bin/env sh
# Generate batch samples.

export PYTHONPATH=$(dirname "$PWD")
export TFRECORD_PATH="../resources/ucf101.tfrecord"
export PHASE="train"
 
python3.6 ../batch_generator.py -a $PHASE -t $TFRECORD_PATH
