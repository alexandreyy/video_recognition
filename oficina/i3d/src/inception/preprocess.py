import tensorflow as tf
import numpy as np
import cv2


class Clip:
    def __init__(self, num_frames=64, start_frame=-1):
        self.num_frames = num_frames
        self.start_frame = start_frame

    def __call__(self, video):
        video = tf.while_loop(lambda v: tf.less(tf.shape(v)[0], self.num_frames),
                              lambda v: tf.concat([v, v[:self.num_frames - tf.shape(v)[0]]], 0),
                              [video])

        if self.start_frame == -1:
            start = tf.random_uniform((), minval=0, maxval=tf.maximum(1, tf.shape(video)[0] - self.num_frames), dtype=tf.int32)

        return video[start: start + self.num_frames]


class RandomCrop:
    def __init__(self, num_frames=64, frame_size=224, num_channels=3):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_channels = num_channels

    def __call__(self, video):
        return tf.random_crop(video, [self.num_frames, self.frame_size, self.frame_size, self.num_channels])


class CenterCrop:
    def __init__(self, num_frames=64, frame_size=224, num_channels=3):
        self.num_frames = num_frames
        self.frame_size = tf.constant(frame_size, dtype=tf.int32)
        self.num_channels = num_channels

    def __call__(self, video):
        x_size = tf.shape(video)[1]
        y_size = tf.shape(video)[2]

        x_start = tf.cast((x_size - self.frame_size) / tf.constant(2), tf.int32)
        y_start = tf.cast((y_size - self.frame_size) / tf.constant(2), tf.int32)

        return video[:, x_start: x_start+self.frame_size, y_start: y_start+self.frame_size, :]


class Flip:
    def __call__(self, video):
        return tf.cond(tf.random_uniform(()) < 0.5,
                       lambda: tf.image.flip_left_right(video),
                       lambda: video)


class Normalize:
    def __init__(self, min=-1.0, max=1.0):
        self.min = min
        self.max = max

    def __call__(self, video):
        video = tf.cast(video, tf.float32)
        return (video / tf.maximum(tf.reduce_max(video), 1e-4)) * (self.max - self.min) - self.min


class PreprocessStack():
    def __init__(self, *preprocess):
        self.preprocess = preprocess

    def __call__(self, video, label):
        for f in self.preprocess:
            video = f(video)

        return video, label


def resize(img, size=256, center=224):
    if min(img.shape[:-1]) == size:
        return img

    rows, cols = img.shape[:-1]
    ratio = float(rows) / cols

    rows, cols = (int(ratio*size), size) if ratio > 1.0 else (size, int(size/ratio))

    img = cv2.resize(img, (rows, cols))

    s_rows, s_cols = (rows-center)//2, (cols-center)//2

    img = img[s_cols:s_cols+center, s_rows:s_rows+center, :]

    return img

def normalize(img, size=256, center=224):
    return (img.astype(np.float32) / np.max(img)) * 2 - 1


def rgb(frames, size=256, center=224):
    new_frames = []

    for frame in frames:
        new_frames += [normalize(resize(frame, size, center), size, center)]

    return new_frames

def flow (X, size=256, center=224):
    X_flow = []
    optical_flow = cv2.DualTVL1OpticalFlow_create()

    for i in range(len(X)):
        X[i] = cv2.cvtColor(cv2.resize(X[i], (224, 224)), cv2.COLOR_BGR2GRAY)

    for i in range(1, len(X)):
        flow = np.zeros(X[i].shape)
        #flow = cv2.calcOpticalFlowFarneback(X[i-1], X[i], flow=flow, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flow = optical_flow.calc(X[i-1], X[i], None)

        flow[flow >= 20] = 20
        flow[flow <= -20] = -20
        # scale to [-1, 1]
        max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
        flow = flow / max_val(flow)

        #flow = normalize(np.clip(flow, -20.0, 20.0), size, center)
 