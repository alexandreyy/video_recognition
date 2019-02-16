"""
Prepare dataset to train/test video classifier.
"""

import argparse
import os
import re

from config import FORGD_VIDEO_DIR_PATH
from utils.path_utils import get_file_name, get_files_in_directory


def get_video_label(video_path):
    """
    Get video label.
    """

    try:
        video_name = get_file_name(video_path)
        regex_result = re.compile('^v_(.*)_(.*)_(.*)').match(video_name)
        label = regex_result.group(1)
    except Exception:
        return None

    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare dataset to train/test video classifier.')
    parser.add_argument('-i', '--video-dir', type=str,
                        default=FORGD_VIDEO_DIR_PATH,
                        help='The video directory.')

    args = parser.parse_args()
    video_dir = args.video_dir
    video_paths = get_files_in_directory(video_dir)

    for video_path in video_paths:
        video_label = get_video_label(video_path)

        if video_label is not None:
            video_dir = FORGD_VIDEO_DIR_PATH + "/" + video_label

            if not os.path.exists(video_dir):
                print("Creating directory %s" % video_dir)
                os.makedirs(video_dir)

            new_video_path = os.path.join(video_dir,
                                          os.path.basename(video_path))

            print("Moving video to %s" % new_video_path)
            os.rename(video_path, new_video_path)
