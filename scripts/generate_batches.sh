#!/usr/bin/env sh
# Generate batch samples.

export PYTHONPATH=$(dirname "$PWD")
export PHASE="train"
 
python3.6 ../batch_generator.py -a $PHASE
