#!/usr/bin/env sh
# Generate random samples.
export PYTHONPATH=$(dirname "$PWD")

python3.6 ../data_generator.py
