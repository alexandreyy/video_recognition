"""
Generate batches.
"""

import argparse
import time

import cv2
import numpy as np
import tensorflow as tf

from config import (BATCH_SIZE, CNN_FRAME_SIZE, CNN_VIDEO_HEIGHT,
                    CNN_VIDEO_WIDTH, FORGD_VIDEO_DIR_PATH, TFRECORD_PATH)
from data_generator import get_labels
from data_record import DataRecord


class BatchGenerator:
    """
    Batch generator.
    """

    def __init__(self, dataset, session=None, tfrecord_path=TFRECORD_PATH,
                 frame_size=CNN_FRAME_SIZE, height=CNN_VIDEO_HEIGHT,
                 width=CNN_VIDEO_WIDTH, batch_size=BATCH_SIZE):
        if session is None:
            self.session = tf.Session()
        else:
            self.session = session

        if dataset not in tfrecord_path:
            tfrecord_path = tfrecord_path.replace(".tfrecord",
                                                  "_%s.tfrecord" % dataset)
        self.data_record = DataRecord(frame_size, height, width, dataset,
                                      batch_size)
        self.data_record.open(tfrecord_path)
        self.next = self.data_record.get_next()

    def get_next(self):
        """
        Generate batches.
        """

        while True:
            try:
                return self.session.run(self.next)
            except Exception as e:
                print(str(e))
                self.data_record.reset(self.session)

    def close(self):
        """
        Close session.
        """

        self.session.close()


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
                        default=TFRECORD_PATH, help='The tfrecord path.')
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
    forgd_video_dir = args.foreground_video_dir

    labels = ["Background"]
    labels.extend(get_labels(forgd_video_dir))
    batch_generator = BatchGenerator(phase, None, tfrecord_path,
                                     frame_size, height,
                                     width, batch_size)

    while True:
        if phase == 'test':
            batch_forgd, batch_labels = batch_generator.get_next()
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
            batch_forgd, batch_backd, batch_labels = \
                batch_generator.get_next()
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
