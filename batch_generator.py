"""
Generate batches to train/test model.
"""

import argparse
import cv2
import time

from config import (BATCH_SIZE, CNN_FRAME_SIZE, CNN_VIDEO_HEIGHT,
                    CNN_VIDEO_WIDTH, RESOURCES_DIR, FORGD_VIDEO_DIR_PATH)
from data_generator import get_labels
from data_record import DataRecord
import data_record
import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate batch samples.')
    parser.add_argument('-d', '--frame-size', type=int,
                        default=CNN_FRAME_SIZE, help='The frame size.')
    parser.add_argument('-s', '--batch-size', type=int, default=BATCH_SIZE,
                        help='Size of the batch.')
    parser.add_argument('-w', '--width', type=int, default=CNN_VIDEO_WIDTH,
                        help='The video width.')
    parser.add_argument('-e', '--height', type=int, default=CNN_VIDEO_HEIGHT,
                        help='The video height.')
    parser.add_argument('-t', '--tfrecord_path', type=str,
                        default=RESOURCES_DIR,
                        help='The input video file.')
    parser.add_argument('-a', '--phase', type=str,
                        default="train",
                        help='train/test/validation phase.')
    parser.add_argument('-f', '--foreground-video-dir', type=str,
                        default=FORGD_VIDEO_DIR_PATH,
                        help='The foreground video directory.')
    args = parser.parse_args()
    frame_size = args.frame_size
    batch_size = args.batch_size
    height = args.height
    width = args.width
    tfrecord_path = args.tfrecord_path
    phase = args.phase
    tfrecord_path = tfrecord_path.replace(".tfrecord", "_%s.tfrecord" % phase)
    forgd_video_dir = args.foreground_video_dir

    labels = ["Background"]
    labels.extend(get_labels(forgd_video_dir))
    data_record = DataRecord(frame_size, height, width, phase, batch_size)
    data_record.open(tfrecord_path)

    if phase == 'test':
        t_forgd_frames, t_label = data_record.get_next()
    else:
        t_forgd_frames, t_backd_frames, t_label = data_record.get_next()

    session = tf.Session()
    while True:
        try:
            if phase == 'test':
                batch_forgd, batch_labels = session.run(
                    [t_forgd_frames, t_label])
                forgd_frames = batch_forgd[-1]
                label = batch_labels[-1]

                if label[0] == 1:
                    print(labels[1 + np.argmax(label[1:])])
                else:
                    print("Background")

                for i in range(forgd_frames.shape[0]):
                    forgd_frame = forgd_frames[i]
                    cv2.imshow('frame', forgd_frame)

                    time.sleep(0.01)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                batch_forgd, batch_backd, batch_labels = session.run(
                    [t_forgd_frames, t_backd_frames, t_label])
                batch_forgd, batch_backd, batch_labels
                forgd_frames = batch_forgd[-1]
                backd_frames = batch_backd[-1]
                label = batch_labels[-1]
                print(labels[1 + np.argmax(label[1:])])

                for i in range(forgd_frames.shape[0]):
                    forgd_frame = forgd_frames[i]
                    backd_frame = backd_frames[i]
                    cv2.imshow('frame',
                               np.vstack((forgd_frame, backd_frame)))
                    time.sleep(0.01)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except StopIteration:
            last_batch = True

    session.close()
