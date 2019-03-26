"""
Read and write data from/to tfrecord.
"""

import numpy as np
import tensorflow as tf

from config import CNN_FRAME_SIZE, CNN_VIDEO_HEIGHT, CNN_VIDEO_WIDTH


def _bytes_feature(value):
    '''
    Create array features.
    '''

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    '''
    Create int features.
    '''

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class DataRecord:
    """
    Read and write tfrecord data.
    """

    def __init__(self, frame_size=CNN_FRAME_SIZE, height=CNN_VIDEO_HEIGHT,
                 width=CNN_VIDEO_WIDTH):
        self.frame_size = frame_size
        self.height = height
        self.width = width
        self.channels = 3

    def open(self, tfrecord_path="", mode="w"):
        """
        Open tfrecord file in write mode.
        """

        self.session = tf.Session()

        if tfrecord_path != "":
            self.tfrecord_path = tfrecord_path

        if mode == "w":
            self.writer = tf.python_io.TFRecordWriter(path=self.tfrecord_path)
        else:
            return

    def close(self):
        """
        Close tfrecord file.
        """

        self.writer.close()
        self.session.close()

    def write(self, data):
        """
        Write data in tfrecord file.
        """

        with self.session.as_default():
            if len(data) == 3:
                forgd_frames, backd_frames, label = data
                forgd_frames = forgd_frames.reshape(
                    (self.frame_size * self.height, self.width, self.channels))
                backd_frames = backd_frames.reshape(
                    (self.frame_size * self.height, self.width, self.channels))

                forgd_frames = tf.image.encode_jpeg(
                    forgd_frames, 'rgb', 95).eval()
                backd_frames = tf.image.encode_jpeg(
                    backd_frames, 'rgb', 95).eval()

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'forgd_frames': _bytes_feature(forgd_frames),
                            'backd_frames': _bytes_feature(backd_frames),
                            'label': _bytes_feature(label.tostring())}))
            else:
                forgd_frames, label = data
                forgd_frames = forgd_frames.reshape(
                    (self.frame_size * self.height, self.width, self.channels))
                forgd_frames = tf.image.encode_jpeg(
                    forgd_frames, 'rgb', 95).eval()

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'forgd_frames': _bytes_feature(forgd_frames),
                            'label': _bytes_feature(label.tostring())}))

            self.writer.write(example.SerializeToString())

    def decode_train(self, string_record):
        """
        Decode tfrecord data.
        """

        example = tf.train.Example()
        example.ParseFromString(string_record)
        forgd_frames = (
            example.features.feature['forgd_frames'].bytes_list.value[0])
        backd_frames = (
            example.features.feature['backd_frames'].bytes_list.value[0])
        label = (
            example.features.feature['label'].bytes_list.value[0])
        seg = np.fromstring(seg_string, dtype=np.uint8)
        seg = seg.reshape(
            (self.image_patch_size, self.image_patch_size, 1))
        label = np.fromstring(label_string, dtype=np.uint8)
        label = label.reshape(
            (self.image_patch_size, self.image_patch_size, 1))

        if return_weld_img:
            img_string = (
                example.features.feature['img'].bytes_list.value[0])
            img = np.fromstring(img_string, dtype=np.uint8)
            img = img.reshape(
                (self.image_patch_size, self.image_patch_size, 1))
            return seg, label, img
        else:
            return seg, label
