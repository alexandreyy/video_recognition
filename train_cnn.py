"""
Train the CNN.
"""

import argparse

from cnn_model import VideoRecognitionCNN
from config import (BACKD_VIDEO_DIR_PATH, FORGD_VIDEO_DIR_PATH, MODEL_DIR,
                    TFRECORD_PATH, TRAIN_STEPS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the CNN.')
    parser.add_argument('-n', '--number-steps', type=int, default=TRAIN_STEPS,
                        help='Number of steps to train network')
    parser.add_argument('-m', '--cnn-model-path', type=str,
                        help='Path of CNN weights', default=MODEL_DIR)
    parser.add_argument('-t', '--tfrecord_path', type=str,
                        default=TFRECORD_PATH, help='The tfrecord path.')

    args = parser.parse_args()
    num_steps = args.number_steps
    tfrecord_path = args.tfrecord_path
    model_path = args.cnn_model_path + "/model.ckpt"

    # Train network.
    model = VideoRecognitionCNN()
    model.fit(tfrecord_path, model_path, num_steps)
