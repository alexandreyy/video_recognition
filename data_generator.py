"""
Generate video samples.
"""

import argparse
import os

from config import BACKD_VIDEO_DIR_PATH, FORGD_VIDEO_DIR_PATH
from utils.path_utils import get_files_in_directory


def train_generator(forgd_video_dir, backd_video_dir, split_ratio=0.8):
    """
    Generate sample from train dataset.
    """

    return data_generator(forgd_video_dir, backd_video_dir, dataset="train",
                          split_ratio=split_ratio)


def test_generator(forgd_video_dir, backd_video_dir, split_ratio=0.8):
    """
    Generate sample from test dataset.
    """

    return data_generator(forgd_video_dir, backd_video_dir, dataset="test",
                          split_ratio=split_ratio)


def data_generator(forgd_video_dir, backd_video_dir, dataset="train",
                   split_ratio=0.8):
    """
    Generate sample from video directory.
    """

    # Create dataset file lists.
    labels, labels_dict = get_labels(forgd_video_dir)
    data = dict()

    for label in labels:
        label_dir = forgd_video_dir + "/" + label
        video_paths = sorted(get_files_in_directory(label_dir))

        if len(video_paths) > 0:
            split_index = int(len(video_paths) * split_ratio)

            if dataset == "train":
                data[label] = video_paths[:split_index]
            else:
                data[label] = video_paths[split_index:]

    print(len(data[label]))
    exit(0)

    yield 0, 0


def get_labels(video_dir):
    """
    Get labels from video directory.
    """

    label_dirs = sorted(os.listdir(video_dir))
    labels_dict = dict()

    i = 0
    for label in label_dirs:
        labels_dict[label] = i
        i += 1

    return label_dirs, labels_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate random video samples.')
    parser.add_argument('-f', '--foreground-video-dir', type=str,
                        default=FORGD_VIDEO_DIR_PATH,
                        help='The foreground video directory.')
    parser.add_argument('-b', '--background-dir', type=str,
                        default=BACKD_VIDEO_DIR_PATH,
                        help='The background video directory.')

    args = parser.parse_args()
    forgd_video_dir = args.foreground_video_dir
    backd_video_dir = args.background_video_dir
    generator = data_generator(forgd_video_dir, backd_video_dir)

    last_batch = False
    while not last_batch:
        try:
            frames, label = next(generator)
            print(frames, label)

        except StopIteration:
            last_batch = True
