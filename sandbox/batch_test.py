import random

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
#     sess = tf.InteractiveSession()
#
#     batch_size = 10
#
#     class generator:
#
#         def __call__(self):
#             for _i in range(100):
#                 yield np.random.random((128, 128, 3))
#
#     dataset = tf.data.Dataset().from_generator(
#         generator(), output_types=tf.uint8,
#         output_shapes=(tf.TensorShape([128, 128, 3])))
#
#     dataset = dataset.batch(batch_size=10)
#     it = dataset.make_initializable_iterator()
#     el = it.get_next()
#
#     with tf.Session() as sess:
#         for i in range(100):
#             sess.run(it.initializer)
#             print('batch', sess.run(el).shape)

    # from generator
    sequence = np.array([[[1]], [[2], [3]], [[3], [4], [5]]])

    def generator():
        for el in sequence:
            yield el

    dataset = tf.data.Dataset().batch(3).from_generator(
        generator, output_types=tf.int64,
        output_shapes=(tf.TensorShape([None, 1])))
    it = dataset.make_initializable_iterator()
    el = it.get_next()

    with tf.Session() as sess:
        sess.run(it.initializer)

        print("-", sess.run(el))
        print("-", sess.run(el))
        print("-", sess.run(el))
