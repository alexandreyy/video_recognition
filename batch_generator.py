"""
Generate batches to train/test model.
"""

import argparse

import cv2

from config import (BACKD_VIDEO_DIR_PATH, BATCH_SIZE, CNN_FRAME_SIZE,
                    CNN_VIDEO_HEIGHT, CNN_VIDEO_WIDTH, FORGD_VIDEO_DIR_PATH,
                    FRAMES_BY_SECOND, PRELOAD_SAMPLES, TRAIN_TEST_SPLIT_RATIO)
from data_generator import get_labels, sample_generator
import numpy as np


def batch_generator(forgd_video_dir, backd_video_dir, batch_size=BATCH_SIZE,
                    width=CNN_VIDEO_WIDTH, height=CNN_VIDEO_HEIGHT,
                    dataset="train", split_ratio=TRAIN_TEST_SPLIT_RATIO,
                    frame_size=CNN_FRAME_SIZE,
                    preload_samples=PRELOAD_SAMPLES,
                    phase="train", fps=FRAMES_BY_SECOND):
    """
    Generate batches to train/test model.
    """

    labels = ["background"]
    labels.extend(get_labels(forgd_video_dir))
    batch_forgd = np.zeros([batch_size, frame_size, height, width, 3],
                           dtype=np.float)
    batch_labels = np.zeros([batch_size, len(labels)], dtype=np.uint8)
    is_train = phase == "train"

    if is_train:
        batch_backd = np.zeros([batch_size, frame_size, height, width, 3])

    index_batch = 0
    last_batch = False
    generator = sample_generator(forgd_video_dir, backd_video_dir,
                                 split_ratio=split_ratio, dataset=dataset,
                                 preload_samples=preload_samples,
                                 frame_size=frame_size, phase=phase, fps=fps)

    while not last_batch:
        while index_batch < batch_size and not last_batch:
            try:
                if is_train:
                    forgd_frames, backd_frames, label = next(generator)
                    batch_backd[index_batch] = backd_frames
                else:
                    forgd_frames, label = next(generator)

                batch_forgd[index_batch] = forgd_frames
                batch_labels[index_batch] = label
                index_batch += 1
            except StopIteration:
                last_batch = True

        index_batch = 0
        batch_forgd /= 255.

        if is_train:
            batch_backd /= 255.
            yield batch_forgd, batch_backd, batch_labels
        else:
            yield batch_forgd, batch_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate batch samples.')
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
                        default=CNN_FRAME_SIZE, help='The frame size.')
    parser.add_argument('-p', '--fps', type=int, default=FRAMES_BY_SECOND,
                        help='The input video file.')
    parser.add_argument('-a', '--batch-size', type=int, default=BATCH_SIZE,
                        help='Size of the batch.')
    parser.add_argument('-w', '--width', type=int, default=CNN_VIDEO_WIDTH,
                        help='The video width.')
    parser.add_argument('-e', '--height', type=int, default=CNN_VIDEO_HEIGHT,
                        help='The video height.')

    args = parser.parse_args()
    args = parser.parse_args()
    forgd_video_dir = args.foreground_video_dir
    backd_video_dir = args.background_video_dir
    frame_size = args.frame_size
    train_test_split_ratio = args.train_test_split_ratio
    fps = args.fps
    batch_size = args.batch_size
    height = args.height
    width = args.width

    # Create generator.
    generator = batch_generator(forgd_video_dir, backd_video_dir,
                                batch_size, width, height,
                                split_ratio=train_test_split_ratio,
                                frame_size=frame_size, phase="train", fps=fps)

    labels = ["background"]
    labels.extend(get_labels(forgd_video_dir))

    # Generate batch samples.
    last_batch = False
    i = 0

    while not last_batch:
        try:
            data = next(generator)
            if len(data) == 3:
                batch_forgd, batch_backd, batch_labels = data
                forgd_frames = batch_forgd[-1]
                backd_frames = batch_backd[-1]
                label = batch_labels[-1]
                print(labels[1 + np.argmax(label[1:])])

                for i in range(forgd_frames.shape[0]):
                    forgd_frame = forgd_frames[i]
                    backd_frame = backd_frames[i]
                    cv2.imshow('frame',
                               np.vstack((forgd_frame, backd_frame)))
                    cv2.waitKey(0)
            else:
                batch_forgd, batch_labels = data
                forgd_frames = batch_forgd[-1]
                label = batch_labels[-1]
                print(labels[np.argmax(label)])

                for i in range(forgd_frames.shape[0]):
                    forgd_frame = forgd_frames[i]
                    cv2.imshow('frame', forgd_frame)
                    cv2.waitKey(0)
        except StopIteration:
            last_batch = True
