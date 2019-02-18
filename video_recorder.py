"""
Record video from the webcam.
"""

import argparse

import cv2

from config import FRAMES_BY_SECOND


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Record video from the webcam .')
    parser.add_argument('-o', '--output-video-file', type=str,
                        default="output.avi",
                        help='The output video file.')
    parser.add_argument('-f', '--fps', type=str,
                        default=FRAMES_BY_SECOND,
                        help='The input video file.')
    args = parser.parse_args()
    output_video_file = args.output_video_file
    fps = args.fps

    # Create a VideoCapture object.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Check if camera opened successfully.
    if cap.isOpened() is False:
        print("Unable to read camera feed.")

    # Default resolutions of the frame are obtained.
    # The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.
    # The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(output_video_file,
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if ret:
            # Write the frame into the file 'output.avi'.
            out.write(frame)

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
    out.release()

    # Closes all the frames.
    cv2.destroyAllWindows()
