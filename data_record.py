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
                 width=CNN_VIDEO_WIDTH, phase="train", batch_size=4):
        self.frame_size = frame_size
        self.height = height
        self.width = width
        self.channels = 3
        self.phase = "train"
        self.batch_size = batch_size

    def open(self, tfrecord_path="", mode="w"):
        """
        Open tfrecord file in write mode.
        """

        self.session = tf.Session()

        if tfrecord_path != "":
            self.tfrecord_path = tfrecord_path

        if mode == "w":
            self.writer = tf.python_io.TFRecordWriter(self.tfrecord_path)
        else:
            self.reader = tf.data.TFRecordDataset(self.tfrecord_path)
            return self.decode()

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

    def decode(self):
        """
        Decode tfrecord data.
        """

        if self.phase == "train":
            features = {
                'forgd_frames': tf.FixedLenSequenceFeature([], tf.string),
                'backd_frames': tf.FixedLenSequenceFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)}
        else:
            features = {
                'forgd_frames': tf.FixedLenSequenceFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)}

        def parse_data_train(example):
            """
            Parse train tfrecord data.
            """

            parsed_example = tf.parse_single_example(example, features)

            forgd_frames = tf.image.decode_jpeg(
                parsed_example['forgd_frames'])
            backd_frames = tf.image.decode_jpeg(
                parsed_example['backd_frames'])
            label = tf.cast(parsed_example['label'], tf.int8)
            return forgd_frames, backd_frames, label

        def parse_data_test(example):
            """
            Parse test/validation tfrecord data.
            """

            parsed_example = tf.parse_single_example(example, features)

            forgd_frames = tf.image.decode_jpeg(
                parsed_example['forgd_frames'])
            label = tf.cast(parsed_example['label'], tf.int8)
            return forgd_frames, label

        if self.phase == "train":
            dataset = self.reader.map(parse_data_train)
        else: 
            dataset = self.reader.map(parse_data_test)
        
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self.batch_size)
        return dataset
