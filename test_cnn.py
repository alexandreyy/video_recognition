"""
Test the CNN.
"""

import argparse

from cnn_model import VideoRecognitionCNN
from config import (MODEL_DIR, TFRECORD_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the CNN.')
    parser.add_argument('-m', '--cnn-model-path', type=str,
                        help='Path of CNN weights', default=MODEL_DIR)
    parser.add_argument('-t', '--tfrecord_path', type=str,
                        default=TFRECORD_PATH, help='The tfrecord path.')

    args = parser.parse_args()
    tfrecord_path = args.tfrecord_path
    model_path = args.cnn_model_path + "/model.ckpt"

    # Test network.
    model = VideoRecognitionCNN()
    model.load(model_path)
