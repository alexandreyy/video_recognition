"""
Project settings.
"""

import os
from pathlib import Path


RESOURCES_DIR = os.path.dirname(os.path.realpath(__file__)) + "/resources"
DATASET_DIR_PATH = str(Path.home()) + "/Datasets/UCF101"
FORGD_VIDEO_DIR_PATH = str(Path.home()) + "/Datasets/UCF101/videos"
BACKD_VIDEO_DIR_PATH = str(Path.home()) + "/Datasets/UCF101/videos_background"
