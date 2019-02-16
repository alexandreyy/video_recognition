"""
Generate video samples.
"""

import argparse
import os
import random

import numpy as np

from config import (BACKD_VIDEO_DIR_PATH, FORGD_VIDEO_DIR_PATH,
                    INPUT_FRAME_SIZE, INPUT_VIDEO_HEIGHT, INPUT_VIDEO_WIDTH,
                    PRELOAD_SAMPLES_BY_LABEL, TRAIN_TEST_SPLIT_RATIO)
from utils.path_utils import get_files_in_directory


def train_generator(forgd_video_dir, backd_video_dir,
                    split_ratio=TRAIN_TEST_SPLIT_RATIO,
                    video_width=INPUT_VIDEO_WIDTH,
                    video_height=INPUT_VIDEO_HEIGHT,
                    frame_size=INPUT_FRAME_SIZE,
                    preload_samples_by_label=PRELOAD_SAMPLES_BY_LABEL):
    """
    Generate sample from train dataset.
    """

    return sample_generator(forgd_video_dir, backd_video_dir, dataset="train",
                            split_ratio=split_ratio, video_width=video_width,
                            video_height=video_height, frame_size=frame_size,
                            preload_samples_by_label=preload_samples_by_label)


def test_generator(forgd_video_dir, backd_video_dir,
                   split_ratio=TRAIN_TEST_SPLIT_RATIO,
                   video_width=INPUT_VIDEO_WIDTH,
                   video_height=INPUT_VIDEO_HEIGHT,
                   frame_size=INPUT_FRAME_SIZE,
                   preload_samples_by_label=PRELOAD_SAMPLES_BY_LABEL):
    """
    Generate sample from test dataset.
    """

    return sample_generator(forgd_video_dir, backd_video_dir, dataset="test",
                            split_ratio=split_ratio, video_width=video_width,
                            video_height=video_height, frame_size=frame_size,
                            preload_samples_by_label=preload_samples_by_label)


def sample_generator(forgd_video_dir, backd_video_dir, dataset="train",
                     split_ratio=TRAIN_TEST_SPLIT_RATIO,
                     video_width=INPUT_VIDEO_WIDTH,
                     video_height=INPUT_VIDEO_HEIGHT,
                     frame_size=INPUT_FRAME_SIZE,
                     preload_samples_by_label=PRELOAD_SAMPLES_BY_LABEL):
    """
    Generate sample from video directory.
    """

    # Create dataset file lists.
    labels = get_labels(forgd_video_dir)
    total_labels = len(labels)
    data = []
    frame_gen = []

    # Add foreground videos.
    for index_label in range(total_labels):
        label_dir = forgd_video_dir + "/" + labels[index_label]
        video_paths = get_files_in_directory(label_dir)

        if len(video_paths) > 0:
            split_index = int(len(video_paths) * split_ratio)

            if dataset == "train":
                data.append(video_paths[:split_index])
            else:
                data.append(video_paths[split_index:])

        frame_gen.append(None)

    # Add background videos.
    video_paths = get_files_in_directory(backd_video_dir)
    split_index = int(len(video_paths) * split_ratio)
    frame_gen.append(None)

    print(len(frame_gen))
    exit(0)

    if dataset == "train":
        data.insert(0, video_paths[:split_index])
    else:
        data.insert(0, video_paths[split_index:])

    labels.insert(0, "background")

    select_label = random.randint(1, total_labels)
    video_path = data[select_label][0]
    f_gen = frames_generator(video_path, frame_size=frame_size)

    try:
        forgd_frames = next(f_gen)
    except StopIteration:
        f_gen = frames_generator(video_path, frame_size=frame_size)

    try:
        backd_frames = next(f_gen)
    except StopIteration:
        f_gen = frames_generator(video_path, frame_size=frame_size)

    yield forgd_frames, backd_frames, select_label


def frames_generator(video_path, frame_size=INPUT_FRAME_SIZE):
    """
    Generate frame from video_path.
    """

    frame = np.zeros((frame_size, video_height, video_width, 3))

    for _i in range(10):
        yield frame


def get_labels(video_dir):
    """
    Get labels from video directory.
    """

    label_dirs = sorted(os.listdir(video_dir))

    return label_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate random video samples.')
    parser.add_argument('-f', '--foreground-video-dir', type=str,
                        default=FORGD_VIDEO_DIR_PATH,
                        help='The foreground video directory.')
    parser.add_argument('-b', '--background-video-dir', type=str,
                        default=BACKD_VIDEO_DIR_PATH,
                        help='The background video directory.')
    parser.add_argument('-s', '--train-test-split-ratio', type=float,
                        default=TRAIN_TEST_SPLIT_RATIO,
                        help='The ratio to split the dataset in'
                             'train and test.')
    parser.add_argument('-w', '--video-width', type=int,
                        default=INPUT_VIDEO_WIDTH,
                        help='The video width.')
    parser.add_argument('-e', '--video-height', type=int,
                        default=INPUT_VIDEO_HEIGHT,
                        help='The video height.')
    parser.add_argument('-d', '--frame-size', type=int,
                        default=INPUT_FRAME_SIZE,
                        help='The frame size.')

    args = parser.parse_args()
    forgd_video_dir = args.foreground_video_dir
    backd_video_dir = args.background_video_dir
    video_width = args.video_width
    video_height = args.video_height
    frame_size = args.frame_size
    train_test_split_ratio = args.train_test_split_ratio

    # Create generator.
    generator = sample_generator(forgd_video_dir, backd_video_dir,
                                 split_ratio=TRAIN_TEST_SPLIT_RATIO,
                                 video_width=video_width,
                                 video_height=video_height,
                                 frame_size=frame_size)

    # Generate samples.
    last_batch = False
    while not last_batch:
        try:
            forgd_frames, backd_frames, label = next(generator)
            print(forgd_frames.shape, backd_frames.shape, label)
            exit(0)

        except StopIteration:
            last_batch = True
