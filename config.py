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

# =============================================================================
# Data generation parameters.

TRAIN_TEST_SPLIT_RATIO = 0.8
INPUT_VIDEO_WIDTH = 320
INPUT_VIDEO_HEIGHT = 240
INPUT_FRAME_SIZE = 30
PRELOAD_SAMPLES = 50
FRAMES_BY_SECOND = 30
BATCH_SIZE = 4
