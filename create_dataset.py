"""
Create tfrecord database.
"""

import argparse

from config import (BACKD_VIDEO_DIR_PATH, CNN_FRAME_SIZE, CNN_VIDEO_HEIGHT,
                    CNN_VIDEO_WIDTH, FORGD_VIDEO_DIR_PATH, FRAMES_BY_SECOND,
                    MAX_SAMPLES_BY_VIDEO, RESOURCES_DIR,
                    TRAIN_TEST_SPLIT_RATIO)
from data_generator import sample_generator
from data_record import DataRecord


def create_dataset(tfrecord_path, forgd_video_dir, backd_video_dir,
                   split_ratio=TRAIN_TEST_SPLIT_RATIO,
                   frame_size=CNN_FRAME_SIZE, width=CNN_VIDEO_WIDTH,
                   height=CNN_VIDEO_HEIGHT, phase="train",
                   fps=FRAMES_BY_SECOND,
                   max_samples_by_video=MAX_SAMPLES_BY_VIDEO):
    """ Generate a list of all pairs of patches and segmentation maps
    and save them in the tfrecord file.

    Args:
        list_directories (list): the image directories.
        tfrecord_path (str): the path to save the tfrecord file.
        stride (int): the stride between patches.
        image_patch_size (int): the patch size.
        preload_size (int): the number of images loaded and shuffled by batch.
    """

    # Create data record.
    tfrecord_path = tfrecord_path.replace(".tfrecord", "_%s.tfrecord" % phase)
    data_record = DataRecord(frame_size, height, width)
    data_record.open(tfrecord_path, mode="w")

    # Create generator.
    generator = sample_generator(forgd_video_dir, backd_video_dir,
                                 split_ratio=split_ratio,
                                 frame_size=frame_size, width=width,
                                 height=height, phase=phase, fps=fps,
                                 max_samples_by_video=max_samples_by_video)

    # Generate samples.
    last_sample = False

    while not last_sample:
        try:
            data = next(generator)
            data_record.write(data)
        except StopIteration:
            last_sample = True

    data_record.close()


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
                        default=CNN_FRAME_SIZE, help='The frame size.')
    parser.add_argument('-w', '--width', type=int, default=CNN_VIDEO_WIDTH,
                        help='The video width.')
    parser.add_argument('-e', '--height', type=int, default=CNN_VIDEO_HEIGHT,
                        help='The video height.')
    parser.add_argument('-p', '--fps', type=int, default=FRAMES_BY_SECOND,
                        help='The input video file.')
    parser.add_argument('-m', '--max-samples-by-video', type=int,
                        default=MAX_SAMPLES_BY_VIDEO,
                        help='The input video file.')
    parser.add_argument('-t', '--tfrecord_path', type=str,
                        default=RESOURCES_DIR,
                        help='The input video file.')
    parser.add_argument('-a', '--phase', type=str,
                        default="train",
                        help='train/test/validation phase.')

    args = parser.parse_args()
    forgd_video_dir = args.foreground_video_dir
    backd_video_dir = args.background_video_dir
    max_samples_by_video = args.max_samples_by_video
    frame_size = args.frame_size
    height = args.height
    width = args.width
    train_test_split_ratio = args.train_test_split_ratio
    fps = args.fps
    tfrecord_path = args.tfrecord_path
    phase = args.phase

    # Create generator.
    create_dataset(tfrecord_path, forgd_video_dir, backd_video_dir,
                   split_ratio=train_test_split_ratio,
                   frame_size=frame_size, width=width, height=height,
                   phase=phase, fps=fps,
                   max_samples_by_video=max_samples_by_video)
