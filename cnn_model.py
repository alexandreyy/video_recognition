"""
CNN Model.
"""

import os
import pickle
import re

import numpy as np
import tensorflow as tf

from batch_generator import BatchGenerator
from config import (BATCH_SIZE, CNN_FRAME_SIZE, CNN_VIDEO_HEIGHT,
                    CNN_VIDEO_WIDTH, DISPLAY_TEST_LOSS_STEP,
                    DISPLAY_TRAIN_LOSS_STEP, FORGD_VIDEO_DIR_PATH, LABEL_SIZE,
                    LEARNING_RATE, MODEL_DIR, NUM_TEST_BATCHES, TFRECORD_PATH)
from data_generator import get_labels


class VideoRecognitionCNN:
    """
    Video Recognition CNN:.
    """

    def __init__(self, phase="test"):
        self.sess = None
        self.var = dict()
        self.phase = "test"

    def build_weight(self, name, w_shape):
        """
        Create or load weight.
        """

        if name not in self.weights:
            if os.path.exists(self.model_meta_path):
                self.weights[name] = tf.get_variable(name, shape=w_shape)
            else:
                self.weights[name] = tf.Variable(tf.random_normal(w_shape),
                                                 name=name)
            self.weights_shape[name] = w_shape

        return self.weights[name]

    def r2_1d(self, name, x, w_shape, down_sampling=False):
        """
        R(2+1)D convolution.
        """

        # Weight shape -> [filter_time, filter_height, filter_width,
        #                  in_channels, out_channels]
        m = int(np.prod(w_shape) / (np.prod(w_shape[1:4]) +
                                    w_shape[0] * w_shape[4]) + 0.5)

        if down_sampling:
            strides = [1, 2, 2, w_shape[3], 1]
        else:
            strides = [1, 1, 1, w_shape[3], 1]

        name = name + "_1"
        with tf.variable_scope(name):
            conv_1 = self.conv3d(name, x, [1, w_shape[1],
                                           w_shape[2], w_shape[3], m],
                                 strides=strides)
            conv_1 = tf.nn.relu(conv_1)

        name = name + "_2"
        with tf.variable_scope(name):
            conv_2 = self.conv3d(name, conv_1,
                                 [3, 1, 1, m, w_shape[4]],
                                 strides=[1, 1, 1, m, 1])
            conv_2 = tf.nn.relu(conv_2)

        return conv_2

    def resnet_block(self, name, x, in_channels, out_channels,
                     down_sampling=False):
        """
        Resnet block.
        """

        previous_layer = x

        name = name + "_1"
        with tf.variable_scope(name):
            conv_1 = self.r2_1d(name, x, [3, 3, 3, in_channels, out_channels],
                                down_sampling=down_sampling)
            conv_1 = tf.nn.relu(conv_1)

        name = name + "_2"
        with tf.variable_scope(name):
            conv_2 = self.r2_1d(name, conv_1,
                                [3, 3, 3, out_channels, out_channels])

        if (in_channels != out_channels) or down_sampling:
            name = name + "_shortcut"

            if down_sampling:
                strides = [1, 2, 2, in_channels, 1]
            else:
                strides = [1, 1, 1, in_channels, 1]

            with tf.variable_scope(name):
                previous_layer = self.conv3d(
                    name, x, [1, 1, 1, in_channels, out_channels],
                    strides=strides)

        name = name + "_out"
        with tf.variable_scope(name):
            result = tf.nn.relu(conv_2 + previous_layer)

        return result

    def conv2d(self, name, x, w_shape, strides=[1, 1, 1, 1]):
        """
        2D convolution.
        """

        # Initialize or load weights.
        # Weight shape -> [filter_height, filter_width, in_channels,
        #                  out_channels]
        weight = self.build_weight(name + "_w", w_shape)

        # Apply convolution.
        x = tf.nn.conv2d(x, weight, strides=strides,
                         padding='SAME')

        return x

    def conv3d(self, name, x, w_shape, strides=[1, 1, 1, 1, 1]):
        """
        3D convolution.
        """

        # Initialize or load weights.
        # Weight shape -> [filter_depth, filter_height, filter_width,
        #                  in_channels, out_channels]
        weight = self.build_weight(name + "_w", w_shape)

        # Apply convolution.
        x = tf.nn.conv3d(x, weight, strides=strides, padding='SAME')

        return x

    def maxpool2d(self, x, k=2, c=1):
        """
        Max pooling 2D.
        """

        return tf.nn.max_pool(x, ksize=[1, k, k, c], strides=[1, k, k, c],
                              padding='SAME')

    def maxpool3d(self, x, t=1, k=2, c=1):
        """
        Max pooling 3D.
        """

        return tf.nn.max_pool3d(x, ksize=[1, t, k, k, c],
                                strides=[1, t, k, k, c], padding='SAME')

    def fc(self, name, x, w_shape):
        """
        Fully connected layer.
        """

        # Initialize or load weights.
        # Weight shape -> [input_size, output_size]
        # Bias shape -> [output_size]
        weight = self.build_weight(name + "_w", w_shape)
        bias = self.build_weight(name + "_b", [w_shape[-1]])

        # Apply operation.
        x = tf.add(tf.matmul(x, weight), bias)

        return x

    def init_model_paths(self, model_path):
        """
        Initialize model paths.
        """

        self.model_path = model_path
        self.dir_model_path = os.path.dirname(model_path)
        self.dir_log_path = self.dir_model_path + "/logs"
        self.train_info_path = self.dir_model_path + "/train_info.pkl"

        if os.path.exists(self.dir_model_path):
            checkpoint_path = os.path.join(self.dir_model_path, "checkpoint")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path) as f:
                    content = f.readlines()
                    regex_result = re.compile(
                        'model_checkpoint_path: "(.*)"\n').match(content[0])
                    last_model_path = regex_result.group(1)
                    self.model_meta_path = last_model_path + ".meta"
            else:
                self.model_meta_path = ""
        else:
            os.makedirs(self.dir_model_path)
            self.model_meta_path = ""

        if not os.path.exists(self.dir_log_path):
            os.makedirs(self.dir_log_path)

    def model(self, X, dropout=1.0, reuse=False, name="test"):
        """
        CNN model.
        """

        with tf.variable_scope(name, reuse=reuse):
            name = "conv_1"
            with tf.variable_scope(name):
                conv_1 = self.r2_1d(name, X, [1, 7, 7, self.input_size[3], 64])

            name = "conv_2x"
            with tf.variable_scope(name):
                conv_2x1 = self.resnet_block(name + "1", conv_1, 64, 64)
                conv_2x2 = self.resnet_block(name + "2", conv_2x1, 64, 64)
                conv_2x3 = self.resnet_block(name + "3", conv_2x2, 64, 64)

            name = "fc"
            with tf.variable_scope(name):
                conv_2x3 = self.maxpool3d(conv_2x3, t=16, k=16, c=1)
                conv_2x3 = tf.reshape(conv_2x3, [-1, 896])
                layer_2_fc = self.fc(name, conv_2x3,
                                     [896, self.label_size])

            name = "output_action"
            with tf.variable_scope(name):
                output_action = tf.nn.sigmoid(layer_2_fc[:1])

            name = "output_classes"
            with tf.variable_scope(name):
                output_classes = tf.nn.softmax(layer_2_fc[1:])

            return [output_action, output_classes]

    def build_model(self, input_size=[CNN_FRAME_SIZE, CNN_VIDEO_HEIGHT,
                                      CNN_VIDEO_WIDTH, 3],
                    label_size=LABEL_SIZE):
        """
        Build CNN model.
        """

        self.input_size = input_size
        self.label_size = label_size
        self.dropout = 0.9
        self.weights = dict()
        self.weights_shape = dict()
        self.input = dict()
        self.layers = dict()

        # Construct model.
        with tf.variable_scope('model'):
            with tf.variable_scope("architecture"):
                self.input["input_video"] = tf.placeholder(
                    shape=[None, self.input_size[0], self.input_size[1],
                           self.input_size[2], self.input_size[3]],
                    dtype=tf.float32, name="input_video")

                if self.phase == "train":
                    self.input["input_background_video"] = tf.placeholder(
                        shape=[None, self.input_size[0], self.input_size[1],
                               self.input_size[2], self.input_size[3]],
                        dtype=tf.float32, name="input_background_video")

                    self.input["input_label"] = tf.placeholder(
                        shape=[None, label_size],
                        dtype=tf.float32, name="input_label")

                    self.layers["train_foreground"] = self.model(
                        self.input["input_video"], self.dropout,
                        name="train_foreground")

                    self.layers["train_background"] = self.model(
                        self.input["input_background_video"], self.dropout,
                        reuse=True, name="train_background")
                else:
                    self.layers["test"] = self.model(self.input["input_video"])

    def build_loss(self, sparse_value=0.01):
        """
        Compute loss.
        """

        with tf.variable_scope("cross_entropy_classes"):
            cross_entropy_classes = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.layers["train_foreground"]
                [len(self.layers["train_foreground"]) - 1],
                labels=self.input["input_label"][1:])
            cross_entropy_classes = tf.reduce_mean(cross_entropy_classes)

        with tf.variable_scope("cross_entropy_action"):
            cross_entropy_action = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.layers["train_foreground"]
                [len(self.layers["train_foreground"]) - 2],
                labels=self.input["input_label"][:1])
            cross_entropy_action = tf.reduce_mean(cross_entropy_action)

            cross_entropy_non_action = \
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.layers["train_background"]
                    [len(self.layers["train_background"]) - 2],
                    labels=0 * self.input["input_label"][:1])
            cross_entropy_non_action = tf.reduce_mean(cross_entropy_non_action)
            cross_entropy_action = cross_entropy_action + \
                cross_entropy_non_action

        with tf.variable_scope("sparse_loss"):
            weight_sum = 0
            for var_name in self.weights_shape:
                if 'w' in var_name:
                    weight_sum += tf.reduce_mean(
                        tf.square(self.weights[var_name]))

            sparse_loss = sparse_value * weight_sum

        loss = cross_entropy_classes + cross_entropy_action + sparse_loss

        return loss, cross_entropy_classes, cross_entropy_action

    def optimizer(self, loss, learning_rate=0.0005):
        """
        CNN Optimizer.
        """

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        return optimizer

    def load_graph(self):
        """
        Load CNN graph.
        """

        self.saver = tf.train.Saver()
        self.open_session()

        if self.model_meta_path != "":
            print("Restoring graph from disk.")
            self.saver.restore(self.sess, tf.train.latest_checkpoint(
                self.dir_model_path))
        else:
            print("Initializing graph.")
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def load_train_info(self):
        """
        Load CNN graph.
        """

        if os.path.exists(self.train_info_path):
            print("Loading train info.")
            with open(self.train_info_path, 'rb') as handle:
                train_info = pickle.load(handle)
        else:
            print("Initializing train info.")
            train_info = dict()
            train_info["step"] = -1
            train_info["best_test_lost"] = 999999999

        return train_info

    def save_train_info(self, train_info):
        """
        Save train information.
        """

        with open(self.train_info_path, 'wb') as handle:
            pickle.dump(train_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_summaries(self, loss_function):
        """
        Create tensorflow summaries.
        """

        # Create summaries.
        train_writer = tf.summary.FileWriter(self.dir_log_path + "/train",
                                             graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(self.dir_log_path + "/validation",
                                            graph=tf.get_default_graph())

        # Create loss summary.
        loss_summary = tf.summary.scalar('loss', loss_function)

        return train_writer, test_writer, loss_summary

    def load(self, model_path="./model/model.ckpt",
             input_size=[CNN_FRAME_SIZE, CNN_VIDEO_HEIGHT, CNN_VIDEO_WIDTH, 3],
             label_size=LABEL_SIZE):
        """
        Load CNN weights.
        """

        # Initialize model paths.
        self.init_model_paths(model_path)

        # Load model.
        tf.reset_default_graph()
        self.build_model(input_size, label_size)
        self.saver = tf.train.Saver()
        self.load_graph()

    def open_session(self):
        """
        Open session.
        """

        if self.sess is None:
            self.sess = tf.Session()

    def close_session(self):
        """
        Close session.
        """

        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def fit(self, tfrecord_path=TFRECORD_PATH,
            model_dir=MODEL_DIR, num_steps=-1,
            input_size=[CNN_FRAME_SIZE, CNN_VIDEO_HEIGHT, CNN_VIDEO_WIDTH, 3],
            batch_size=BATCH_SIZE, label_size=LABEL_SIZE,
            learning_rate=LEARNING_RATE, num_test_batches=NUM_TEST_BATCHES,
            display_train_loss_step=DISPLAY_TRAIN_LOSS_STEP,
            display_test_loss_step=DISPLAY_TEST_LOSS_STEP):
        """
        Fit CNN model.
        """

        # TODO: Remove this line after defining the right label size.
        label_size = len(get_labels(FORGD_VIDEO_DIR_PATH)) + 1

        # Initialize model paths.
        model_path = model_dir + "/model.ckpt"
        self.init_model_paths(model_path)
        self.phase = "train"

        # Initialize model.
        tf.reset_default_graph()
        self.build_model(input_size, label_size)

        # Create loss.
        with tf.variable_scope("loss_error"):
            loss_function, cross_entropy_classes, cross_entropy_action = \
                self.build_loss()

        # Create optimization function.
        optimizer = self.optimizer(loss_function, learning_rate)

        # Create summaries.
        train_writer, test_writer, loss_summary = self.create_summaries(
            loss_function)

        # Start train session.
        self.open_session()
        train_info = self.load_train_info()
        self.load_graph()

        # Create batch generators.
        train_generator = BatchGenerator(
            "train", self.sess, tfrecord_path, self.input_size[0],
            self.input_size[1], self.input_size[2], batch_size)
        test_generator = BatchGenerator(
            "validation", self.sess, tfrecord_path, self.input_size[0],
            self.input_size[1], self.input_size[2], batch_size)

        while train_info["step"] < num_steps or num_steps == -1:
            # Get train batch.
            forgd_samples, backd_samples, labels = train_generator.get_next()

            if train_info["step"] % display_train_loss_step == 0:
                train_loss_s, error_classes, error_action, loss_train_val, \
                    _opt_val = self.sess.run(
                        [loss_summary, cross_entropy_classes,
                            cross_entropy_action, loss_function, optimizer],
                        feed_dict={
                            self.input["input_video"]: forgd_samples,
                            self.input["input_background_video"]:
                            backd_samples,
                            self.input["input_label"]: labels})
                train_writer.add_summary(train_loss_s, train_info["step"])
                print('Step %i: train loss: %f,'
                      ' classes loss: %f, action loss: %f'
                      % (train_info["step"], loss_train_val,
                         error_classes, error_action))
            else:
                _opt_val, loss_train_val = self.sess.run(
                    [optimizer, loss_function],
                    feed_dict={self.input["input_video"]: forgd_samples,
                               self.input["input_background_video"]:
                               backd_samples,
                               self.input["input_label"]: labels})

            self.save_train_info(train_info)
            train_writer.flush()

            # Display test loss and input/output images.
            if train_info["step"] % display_test_loss_step == 0:
                test_loss_list = []
                error_classes_list = []
                error_action_list = []

                batch_index = 0
                while batch_index < num_test_batches:
                    forgd_samples, backd_samples, labels = \
                        test_generator.get_next()

                    batch_index += 1
                    if batch_index < num_test_batches:
                        loss_test_val, error_classes, error_action = \
                            self.sess.run(
                                [loss_function, cross_entropy_classes,
                                 cross_entropy_action],
                                feed_dict={
                                    self.input["input_video"]: forgd_samples,
                                    self.input["input_background_video"]:
                                    backd_samples,
                                    self.input["input_label"]: labels})
                    else:
                        loss_s, loss_test_val, error_classes, error_action = \
                            self.sess.run(
                                [loss_summary, loss_function,
                                 cross_entropy_classes, cross_entropy_action],
                                feed_dict={
                                    self.input["input_video"]: forgd_samples,
                                    self.input["input_background_video"]:
                                    backd_samples,
                                    self.input["input_label"]: labels})

                test_loss_list.append(loss_test_val)
                error_classes_list.append(error_classes)
                error_action_list.append(error_action)
                loss_test_val = np.mean(test_loss_list)

                if loss_test_val < train_info["best_test_lost"]:
                    train_info["best_test_lost"] = loss_test_val
                    self.saver.save(self.sess, model_path,
                                    global_step=train_info["step"])

                print('Step %i: validation loss: %f,'
                      ' best validation loss: %f, classes loss: %f, '
                      'action loss: %f'
                      % (train_info["step"],
                         loss_test_val, train_info["best_test_lost"],
                         np.mean(error_classes_list),
                         np.mean(error_action_list)))
                test_writer.add_summary(loss_s, train_info["step"])
                test_writer.flush()
                self.save_train_info(train_info)

            train_info["step"] += 1
        self.close_session()

    def predict(self, X):
        """
        Predict.
        """

        return X
