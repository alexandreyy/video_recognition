"""
Capture camera video or play a video file.
"""

import argparse

import cv2

from config import FRAMES_BY_SECOND


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Capture camera video or play a video file.')
    parser.add_argument('-i', '--input-video-file', type=str,
                        default="",
                        help='The input video file.')
    parser.add_argument('-f', '--fps', type=str,
                        default=FRAMES_BY_SECOND,
                        help='The input video file.')

    args = parser.parse_args()
    source_video = args.input_video_file
    fps = args.fps

    if source_video == "":
        # Capture data from camera.
        cap = cv2.VideoCapture(0)
    else:
        # Capture data from video.
        cap = cv2.VideoCapture(source_video)
        video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully.
    if cap.isOpened() is False:
        print("Unable to read camera feed or load file.")

    i = 0
    frame_index = 0
    video_time = 0
    expected_frame_index = 0
    expected_time = 0

    while True:
        delta = expected_time - video_time

        if delta >= 0:
            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1
            video_time = frame_index / video_fps
        else:
            expected_frame_index += 1.0
            expected_time = expected_frame_index / fps

            # Display the resulting frame.
            cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the video capture and
    # video write objects.
    cap.release()

    # Closes all the frames.
    cv2.destroyAllWindows()
