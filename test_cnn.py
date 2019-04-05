"""
Test the CNN.
"""

import argparse
import cv2
import time
from batch_generator import BatchGenerator
from cnn_model import VideoRecognitionCNN
from config import (MODEL_DIR, TFRECORD_PATH, CNN_FRAME_SIZE, BATCH_SIZE,
                    CNN_VIDEO_WIDTH, CNN_VIDEO_HEIGHT, FORGD_VIDEO_DIR_PATH)
from data_generator import get_labels
import numpy as np
from numpy import argsort


def get_top_classes(probs, labels, total=5):
    """
    Get top classes.
    """

    selected_labels = argsort(-probs) < total
    result = [labels[i] for i in np.where(selected_labels)[0]]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the CNN.')
    parser.add_argument('-m', '--cnn-model-path', type=str,
                        help='Path of CNN weights', default=MODEL_DIR)
    parser.add_argument('-t', '--tfrecord_path', type=str,
                        default=TFRECORD_PATH, help='The tfrecord path.')
    parser.add_argument('-d', '--frame-size', type=int,
                        default=CNN_FRAME_SIZE, help='The frame size.')
    parser.add_argument('-s', '--batch-size', type=int, default=BATCH_SIZE,
                        help='Size of the batch.')
    parser.add_argument('-w', '--width', type=int, default=CNN_VIDEO_WIDTH,
                        help='The video width.')
    parser.add_argument('-e', '--height', type=int, default=CNN_VIDEO_HEIGHT,
                        help='The video height.')
    parser.add_argument('-f', '--foreground-video-dir', type=str,
                        default=FORGD_VIDEO_DIR_PATH,
                        help='The foreground video directory.')

    args = parser.parse_args()
    tfrecord_path = args.tfrecord_path
    model_path = args.cnn_model_path + "/model.ckpt"

    frame_size = args.frame_size
    batch_size = args.batch_size
    height = args.height
    width = args.width
    forgd_video_dir = args.foreground_video_dir

    # Test network.
    model = VideoRecognitionCNN()
    model.load(model_path)

    args = parser.parse_args()
    frame_size = args.frame_size
    batch_size = args.batch_size
    height = args.height
    width = args.width
    tfrecord_path = args.tfrecord_path
    forgd_video_dir = args.foreground_video_dir

    labels = get_labels(forgd_video_dir)
    batch_generator = BatchGenerator("train", None, tfrecord_path,
                                     frame_size, height,
                                     width, 1)

    while True:
        batch_forgd, batch_backd, batch_labels = \
            batch_generator.get_next()
#         batch_forgd, batch_labels = batch_generator.get_next()
        action, probs = model.predict(batch_forgd[0])

        forgd_frames = batch_forgd[-1]
        label = batch_labels[-1]

        if label[0] == 1:
            label = labels[np.argmax(label[1:])]
        else:
            continue
            # label = "Background"

        top_classes = get_top_classes(probs, labels)
        print(label in top_classes, label, top_classes)

        for i in range(forgd_frames.shape[0]):
            forgd_frame = forgd_frames[i]
            cv2.imshow('frame', forgd_frame)

            time.sleep(0.01)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.waitKey()
