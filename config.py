"""
Project settings.
"""

import os
from pathlib import Path
from multiprocessing import cpu_count

# =============================================================================
# Project parameters.

NUM_PROCESS = cpu_count() - 2

# =============================================================================
# Dataset parameters.

RESOURCES_DIR = os.path.dirname(os.path.realpath(__file__)) + "/resources"
DATASET_DIR_PATH = str(Path.home()) + "/Datasets/UCF101"
FORGD_VIDEO_DIR_PATH = DATASET_DIR_PATH + "/videos_foreground"
BACKD_VIDEO_DIR_PATH = DATASET_DIR_PATH + "/videos_background"
TFRECORD_PATH = RESOURCES_DIR + "/ucf101.tfrecord"

# =============================================================================
# Data generation parameters.

TRAIN_TEST_SPLIT_RATIO = 0.8
PREPROCESS_VIDEO_WIDTH = 171
PREPROCESS_VIDEO_HEIGHT = 128
CNN_FRAME_SIZE = 32
CNN_VIDEO_WIDTH = 112
CNN_VIDEO_HEIGHT = 112
PRELOAD_SAMPLES = 50
FRAMES_BY_SECOND = 30
BATCH_SIZE = 4
MAX_SAMPLES_BY_VIDEO = 4

# =============================================================================
# CNN parameters.

MODEL_DIR = RESOURCES_DIR + "/model"
TRAIN_STEPS = -1
LABEL_SIZE = 103
LEARNING_RATE = 0.0005
NUM_TEST_BATCHES = 1
DISPLAY_TRAIN_LOSS_STEP = 10
DISPLAY_TEST_LOSS_STEP = 100
