'''
Capture camera video or play a video file.
'''

import argparse

import cv2

if __name__ == "__main__":
    """
    Capture camera video or play a video file.
    """

    parser = argparse.ArgumentParser(
        description='Capture camera video or play a video file.')
    parser.add_argument('-i', '--input-video-file', type=str,
                        default="",
                        help='The input video file.')

    args = parser.parse_args()
    source_video = args.input_video_file

    if source_video == "":
        # Capture data from camera.
        cap = cv2.VideoCapture(0)
    else:
        # Capture data from video.
        cap = cv2.VideoCapture(source_video)

    # Check if camera opened successfully.
    if cap.isOpened() is False:
        print("Unable to read camera feed or load file.")

    while True:
        ret, frame = cap.read()

        if ret:
            # Display the resulting frame.
            cv2.imshow('frame', frame)

            # Press Q on keyboard to stop recording.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop.
        else:
            break

    # When everything done, release the video capture and video write objects.
    cap.release()

    # Closes all the frames.
    cv2.destroyAllWindows()
