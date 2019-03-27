"""
Read and write data from/to tfrecord.
"""

import tensorflow as tf

from config import (CNN_FRAME_SIZE, CNN_VIDEO_HEIGHT, CNN_VIDEO_WIDTH,
                    BATCH_SIZE)


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
                 width=CNN_VIDEO_WIDTH, phase="train", batch_size=BATCH_SIZE):
        self.frame_size = frame_size
        self.height = height
        self.width = width
        self.channels = 3
        self.phase = phase
        self.batch_size = batch_size
        self.session = None
        self.reader = None
        self.writer = None
        self.iterator = None

    def open(self, tfrecord_path="", mode="r"):
        """
        Open tfrecord file in write mode.
        """

        if tfrecord_path != "":
            self.tfrecord_path = tfrecord_path
        self.mode = mode

        if mode == "w":
            self.session = tf.Session()
            self.writer = tf.python_io.TFRecordWriter(self.tfrecord_path)
        else:
            self.reader = self.get_reader()
            self.iterator = self.reader.make_one_shot_iterator()

    def close(self):
        """
        Close tfrecord file.
        """

        if self.mode == "w" and self.session is not None:
            self.writer.close()
            self.session.close()

    def write(self, data):
        """
        Write data in tfrecord file.
        """

        if self.writer is not None:
            with self.session.as_default():
                if len(data) == 3:
                    forgd_frames, backd_frames, label = data
                    forgd_frames = forgd_frames.reshape(
                        (self.frame_size * self.height,
                         self.width, self.channels))
                    backd_frames = backd_frames.reshape(
                        (self.frame_size * self.height,
                         self.width, self.channels))

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
                        (self.frame_size * self.height, self.width,
                         self.channels))
                    forgd_frames = tf.image.encode_jpeg(
                        forgd_frames, 'rgb', 95).eval()

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'forgd_frames': _bytes_feature(forgd_frames),
                                'label': _bytes_feature(label.tostring())}))

                self.writer.write(example.SerializeToString())

    def get_reader(self):
        """
        Decode tfrecord data.
        """

        self.reader = tf.data.TFRecordDataset(self.tfrecord_path)

        if self.phase == "test":
            features = {
                'forgd_frames': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)}
        else:
            features = {
                'forgd_frames': tf.FixedLenFeature([], tf.string),
                'backd_frames': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)}

        def parse_data_train(example):
            """
            Parse train tfrecord data.
            """

            parsed_example = tf.parse_single_example(example, features)

            forgd_frames = tf.image.decode_jpeg(parsed_example['forgd_frames'])
            backd_frames = tf.image.decode_jpeg(parsed_example['backd_frames'])
            label = parsed_example['label']
            forgd_frames = tf.reshape(forgd_frames,
                                      shape=[self.frame_size, self.height,
                                             self.width, self.channels])
            backd_frames = tf.reshape(backd_frames,
                                      shape=[self.frame_size, self.height,
                                             self.width, self.channels])
            label = tf.decode_raw(label, tf.int8)
            forgd_frames = tf.cast(forgd_frames, tf.float32) / 255.
            backd_frames = tf.cast(backd_frames, tf.float32) / 255.

            return forgd_frames, backd_frames, label

        def parse_data_test(example):
            """
            Parse test/validation tfrecord data.
            """

            parsed_example = tf.parse_single_example(example, features)

            forgd_frames = tf.image.decode_jpeg(parsed_example['forgd_frames'])
            forgd_frames = tf.reshape(forgd_frames,
                                      shape=[self.frame_size, self.height,
                                             self.width, self.channels])
            forgd_frames = tf.cast(forgd_frames, tf.float32) / 255.
            label = parsed_example['label']
            label = tf.decode_raw(label, tf.int8)
            return forgd_frames, label

        if self.phase == "test":
            dataset = self.reader.map(parse_data_test)
        else:
            dataset = self.reader.map(parse_data_train)

        dataset = dataset.repeat()
        dataset = dataset.shuffle(100)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_next(self):
        """
        Get next data from iterator.
        """

        if self.iterator is not None:
            return self.iterator.get_next()
        else:
            return None
