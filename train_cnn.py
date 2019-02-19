"""
Train the CNN.
"""

import argparse

from cnn_model import VideoRecognitionCNN
from config import (BACKD_VIDEO_DIR_PATH, FORGD_VIDEO_DIR_PATH, MODEL_PATH,
                    TRAIN_STEPS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the CNN.')
    parser.add_argument('-n', '--number-steps', type=int, default=TRAIN_STEPS,
                        help='Number of steps to train network')
    parser.add_argument('-w', '--weight-cnn-path', type=str,
                        help='Path of CNN weights', default=MODEL_PATH)
    parser.add_argument('-f', '--foreground-video-dir', type=str,
                        default=FORGD_VIDEO_DIR_PATH,
                        help='The foreground video directory.')
    parser.add_argument('-b', '--background-video-dir', type=str,
                        default=BACKD_VIDEO_DIR_PATH,
                        help='The background video directory.')

    args = parser.parse_args()
    num_steps = args.number_steps
    forgd_video_dir = args.foreground_video_dir
    backd_video_dir = args.background_video_dir
    model_path = args.weight_cnn_path + "/model.ckpt"

    # Train network.
    model = VideoRecognitionCNN()
    model.fit(forgd_video_dir, backd_video_dir, model_path, num_steps)
    model.close_session()
