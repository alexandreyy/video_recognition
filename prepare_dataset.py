"""
Prepare dataset to train/test video classifier.
"""

import argparse
import os
import re
from multiprocessing import Pool
import cv2

from config import (BACKD_VIDEO_DIR_PATH, FORGD_VIDEO_DIR_PATH,
                    PREPROCESS_VIDEO_HEIGHT, PREPROCESS_VIDEO_WIDTH,
                    NUM_PROCESS)
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


def adjust_frame_size(frame, width=PREPROCESS_VIDEO_WIDTH,
                      height=PREPROCESS_VIDEO_HEIGHT):
    """
    Adjust frame size.
    """

    frame_height, frame_width = frame.shape[:2]

    if frame_height == height and frame_width == width:
        return frame
    else:
        ratio = height / frame_height
        new_width = int(frame_width * ratio)

        if new_width >= width:
            frame = cv2.resize(frame, (new_width, height))
            start = int((new_width - width) / 2)
            frame = frame[:, start:(width + start)]
        else:
            ratio = width / frame_width
            new_height = int(frame_height * ratio)
            frame = cv2.resize(frame, (width, new_height))
            start = int((new_height - height) / 2)
            frame = frame[start:(height + start)]

    return frame


def adjust_video_size(input_path, output_path, width=PREPROCESS_VIDEO_WIDTH,
                      height=PREPROCESS_VIDEO_HEIGHT):
    """
    Adjust frames to fit width and height.
    """

    if not os.path.exists(output_path):
        cap = cv2.VideoCapture(input_path)

        # Check if camera opened successfully.
        if cap.isOpened() is False:
            print("Unable to read camera feed or load file.")
            return

        # Define the codec and create VideoWriter object.
        # The output is stored in 'outpy.avi' file.
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        if frame_width != width or frame_height != height:
            print("Adjusting size of video %s" % input_path)
            out = cv2.VideoWriter(output_path,
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                  fps, (width, height))
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = adjust_frame_size(frame, width, height)
                out.write(frame)

            # When everything done, release the video capture and
            # video write objects.
            out.release()
            cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare dataset to train/test video classifier.')
    parser.add_argument('-f', '--foreground-video-dir', type=str,
                        default=FORGD_VIDEO_DIR_PATH,
                        help='The foreground video directory.')
    parser.add_argument('-b', '--background-video-dir', type=str,
                        default=BACKD_VIDEO_DIR_PATH,
                        help='The background video directory.')
    parser.add_argument('-w', '--width', type=int,
                        default=PREPROCESS_VIDEO_WIDTH,
                        help='The video width.')
    parser.add_argument('-e', '--height', type=int,
                        default=PREPROCESS_VIDEO_HEIGHT,
                        help='The video height.')

    args = parser.parse_args()
    foreground_video_dir = args.foreground_video_dir
    background_video_dir = args.background_video_dir
    height = args.height
    width = args.width
    video_paths = get_files_in_directory(foreground_video_dir)

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

    video_paths = get_files_in_directory(background_video_dir)
    for i in os.listdir(foreground_video_dir):
        directory = foreground_video_dir + "/" + i
        video_paths.extend(get_files_in_directory(directory))

    pool = Pool(NUM_PROCESS)
    works = []

    for video_path in video_paths:
        ext = os.path.basename(video_path).split(".")[-1]
        output_path = video_path.replace("." + ext, "_adjusted" + ".avi")
        works.append(pool.apply_async(adjust_video_size,
                                      args=(video_path, output_path,
                                            width, height)))

    pool.close()
    pool.join()
