"""
Generate video samples.
"""

import argparse
import cv2
import os
import random

import numpy as np

from config import (BACKD_VIDEO_DIR_PATH, FORGD_VIDEO_DIR_PATH,
                    INPUT_FRAME_SIZE, PRELOAD_SAMPLES,
                    TRAIN_TEST_SPLIT_RATIO)
from utils.path_utils import get_files_in_directory


def train_generator(forgd_video_dir, backd_video_dir,
                    split_ratio=TRAIN_TEST_SPLIT_RATIO,
                    frame_size=INPUT_FRAME_SIZE,
                    preload_samples=PRELOAD_SAMPLES):
    """
    Generate sample from train dataset.
    """

    return sample_generator(forgd_video_dir, backd_video_dir, dataset="train",
                            split_ratio=split_ratio, frame_size=frame_size,
                            preload_samples=preload_samples,
                            phase="train")


def validation_generator(forgd_video_dir, backd_video_dir,
                         split_ratio=TRAIN_TEST_SPLIT_RATIO,
                         frame_size=INPUT_FRAME_SIZE,
                         preload_samples=PRELOAD_SAMPLES):
    """
    Generate sample from validation dataset.
    """

    return sample_generator(forgd_video_dir, backd_video_dir,
                            dataset="validation",
                            split_ratio=split_ratio, frame_size=frame_size,
                            preload_samples=preload_samples,
                            phase="train")


def test_generator(forgd_video_dir, backd_video_dir,
                   split_ratio=TRAIN_TEST_SPLIT_RATIO,
                   frame_size=INPUT_FRAME_SIZE,
                   preload_samples=PRELOAD_SAMPLES):
    """
    Generate sample from test dataset.
    """

    return sample_generator(forgd_video_dir, backd_video_dir,
                            dataset="test",
                            split_ratio=split_ratio, frame_size=frame_size,
                            preload_samples=preload_samples,
                            phase="test")


def get_labeled_video(forgd_video_dir, backd_video_dir, dataset="train",
                      split_ratio=TRAIN_TEST_SPLIT_RATIO):
    """
    Get video paths by label.
    """

    # Create dataset file lists.
    labels = ["background"]
    data = []

    # Add background videos.
    video_paths = get_files_in_directory(backd_video_dir)
    split_index = int(len(video_paths) * split_ratio)

    if dataset == "train":
        data.append(video_paths[:split_index])
    else:
        data.append(video_paths[split_index:])

    # Add foreground videos.
    labels.extend(get_labels(forgd_video_dir))
    total_labels = len(labels)

    for index_label in range(1, total_labels):
        label_dir = forgd_video_dir + "/" + labels[index_label]
        video_paths = get_files_in_directory(label_dir)

        if len(video_paths) > 0:
            split_index = int(len(video_paths) * split_ratio)

            if dataset == "train":
                data.append(video_paths[:split_index])
            else:
                data.append(video_paths[split_index:])

    return data, labels


def sample_generator(forgd_video_dir, backd_video_dir, dataset="train",
                     split_ratio=TRAIN_TEST_SPLIT_RATIO,
                     frame_size=INPUT_FRAME_SIZE,
                     preload_samples=PRELOAD_SAMPLES,
                     phase="train"):
    """
    Generate sample from video directory.
    """

    paths, labels = get_labeled_video(forgd_video_dir, backd_video_dir,
                                      dataset, split_ratio)
    total_labels = len(labels)
    labels = []
    backd_gens = []
    forgd_gens = []
    augment_data = phase == "train"
    preload_samples = int(preload_samples / 2)
    count_forgd_gen_labels = dict()

    for i in range(total_labels):
        if len(paths[i]) > 0:
            if augment_data:
                if i > 0:
                    labels.append(i)
                else:
                    for _j in range(preload_samples):
                        video_path = paths[0][random.randint(
                            0, len(paths[0]) - 1)]
                        backd_gens.append(
                            frames_generator(video_path, frame_size,
                                             augment_data))
            else:
                labels.append(i)

    while len(forgd_gens) > 0 or len(labels) > 0:
        if augment_data:
            backd_frames = None

            while backd_frames is None:
                try:
                    index_gen = random.randint(0, len(backd_gens) - 1)
                    backd_frames = next(backd_gens[index_gen])
                except StopIteration:
                    backd_frames = None
                    video_path = paths[0][random.randint(0,
                                                         len(paths[0]) - 1)]
                    backd_gens[index_gen] = frames_generator(video_path,
                                                             frame_size,
                                                             augment_data)

        if len(forgd_gens) > 0:
            select_gen = random.choice(forgd_gens)

            try:
                if augment_data:
                    forgd_frames = next(select_gen[1])
                    yield forgd_frames, backd_frames, select_gen[0]
                else:
                    forgd_frames = next(select_gen[1])
                    yield forgd_frames, select_gen[0]
            except StopIteration:
                forgd_gens.remove(select_gen)

                if len(paths[select_gen[0]]) == 0:
                    if augment_data:
                        count_forgd_gen_labels[select_gen[0]] -= 1

                        if count_forgd_gen_labels[select_gen[0]] <= 0:
                            return
                    elif select_gen[0] in labels:
                        labels.remove(select_gen[0])

        while len(forgd_gens) < preload_samples and len(labels) > 0:
            select_label = random.choice(labels)

            if len(paths[select_label]) > 0:
                video_path = paths[select_label].pop(
                    random.randint(0, len(paths[select_label]) - 1))
                forgd_gens.append((select_label,
                                   frames_generator(video_path,
                                                    frame_size,
                                                    augment_data)))

                if augment_data:
                    if select_label in count_forgd_gen_labels.keys():
                        count_forgd_gen_labels[select_label] += 1
                    else:
                        count_forgd_gen_labels[select_label] = 1

            elif select_label in labels:
                labels.remove(select_label)


def frames_generator(video_path, frame_size=INPUT_FRAME_SIZE,
                     augment_data=True):
    """
    Generate frame from video_path.
    """

    if os.path.exists(video_path):
        # Capture data from video.
        cap = cv2.VideoCapture(video_path)

        if cap.isOpened():
            index_frame = 0
            jump_random = 0
            frames = []

            while True:
                ret, frame = cap.read()
                if augment_data:
                    jump_random -= 1

                if ret:
                    if jump_random <= 0:
                        if len(frames) >= frame_size:
                            frames.pop(0)

                        frames.append(frame)
                        index_frame += 1

                        if index_frame >= frame_size:
                            index_frame = 0

                            if augment_data:
                                jump_random = random.randint(0, frame_size)

                            yield np.array(frames)
                else:
                    # Break the loop.
                    break

        # When everything done, release the video capture.
        cap.release()


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
    parser.add_argument('-d', '--frame-size', type=int,
                        default=INPUT_FRAME_SIZE,
                        help='The frame size.')

    args = parser.parse_args()
    forgd_video_dir = args.foreground_video_dir
    backd_video_dir = args.background_video_dir
    frame_size = args.frame_size
    train_test_split_ratio = args.train_test_split_ratio

    # Create generator.
    generator = sample_generator(forgd_video_dir, backd_video_dir,
                                 split_ratio=TRAIN_TEST_SPLIT_RATIO,
                                 frame_size=frame_size,
                                 phase="train")

    # Generate samples.
    last_sample = False
    i = 0
    while not last_sample:
        try:
            data = next(generator)
            if len(data) == 3:
                forgd_frames, backd_frames, label = data
#                 cv2.imshow('frame',
#                            np.vstack((forgd_frames[0], backd_frames[0])))

            else:
                forgd_frames, label = data
#                 cv2.imshow('frame', forgd_frames[0])
#             cv2.waitKey(0)
            print(i)
            i += 1

        except StopIteration:
            last_sample = True
            print("End")
