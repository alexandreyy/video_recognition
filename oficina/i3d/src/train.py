import cv2
import numpy as np
import tensorflow as tf

from inception.inception import InceptionI3d, load_kinetics_weights
from mobilenet.mobilenet import load_imagenet_weights, add_mobilenet_logits
from inception.preprocess import Clip, RandomCrop, Normalize, PreprocessStack, Flip, CenterCrop

import glob
from helpers.video_trainer import load_from_record, get_data_size


# FLAGS
tf_flags = tf.app.flags
FLAGS = tf_flags.FLAGS

tf_flags.DEFINE_integer('num_classes',      25,         'Number of classes considered')
tf_flags.DEFINE_integer('frame_size',       224,        'The default frame input size')
tf_flags.DEFINE_integer('num_frames',       32,         'Default number of frames for training')
tf_flags.DEFINE_integer('num_channels',     3,          'Number of channels (3 for rgb, 2 for optical flow)')
tf_flags.DEFINE_integer('batch_size',       4,          'Batch size')
tf_flags.DEFINE_integer('num_epochs',       100,        'Total number of epochs')
tf_flags.DEFINE_integer('train_loops',      2,          'Number of steps to spend in the training set before running a full evaluation on the test set')
tf_flags.DEFINE_integer('display_steps',    10,         'Number of steps to spend before displaying the current summaries')
tf_flags.DEFINE_float('learning_rate',      1e-3,       'Initial learning rate')
tf_flags.DEFINE_float('dropout_prob',       0.5,        'Dropout probability for training')

tf_flags.DEFINE_string('train_data_pattern', '../data/hmdb/hmdb_train*.tfrecord', 'Path for the training data (a tfrecord file)')
tf_flags.DEFINE_string('val_data_pattern', '../data/hmdb/hmdb_test*.tfrecord', 'Path for the evaluation data (a tfrecord file)')
tf_flags.DEFINE_string('kinetics_model', "../data/kinetics/rgb_imagenet/model.ckpt", 'Path for the checkpoint file (optional)')
tf_flags.DEFINE_string('checkpoints_model', "../data/checkpoints/model.ckpt", 'Path for the checkpoint file (optional)')
tf_flags.DEFINE_string('summaries_path', '../data/summaries', 'Path for summaries data')



tf_flags.DEFINE_integer('train_entries', get_data_size(FLAGS.train_data_pattern), 'Number of training entries')
tf_flags.DEFINE_integer('val_entries', get_data_size(FLAGS.val_data_pattern), 'Number of training entries')
tf_flags.DEFINE_integer('train_steps', (FLAGS.train_entries * FLAGS.train_loops) // FLAGS.batch_size,  'Number of steps to spend in the training set before running a full evaluation on the test set')
tf_flags.DEFINE_integer('val_steps', FLAGS.val_entries // FLAGS.batch_size,  'Number of steps to spend in the training set before running a full evaluation on the test set')

tf_flags.DEFINE_integer('from_kinetics',      1,   'If training is executed from kinetics')
tf_flags.DEFINE_integer('from_checkpoints',   0,   'If training is executed from an initial checkpoint')


def get_summary(summary_dict):
    summary = tf.Summary()

    for name, value in summary_dict.items():
        summary.value.add(tag=name, simple_value=value)

    return summary



def train(train_data, val_data, build_model_fn):

    # The data input placeholder
    input_data = tf.placeholder(tf.string, shape=[])

    # A common iterator for both training and validation, allowing for easy switch of the datasets
    iterator = tf.data.Iterator.from_string_handle(input_data, train_data.output_types, train_data.output_shapes)
    inputs, labels = iterator.get_next()


    # Create iterators to training and valiation data
    train_iterator = train_data.make_one_shot_iterator()
    val_iterator = val_data.make_initializable_iterator()


    # Placeholders for training/inference
    training = tf.placeholder(tf.bool, shape=(), name="training")
    dropout_prob = tf.placeholder(tf.float32, shape=(), name="dropoub_prob")

    sess = tf.Session()

    logits, vars_to_initialize = build_model_fn(sess, inputs, training, dropout_prob)


    # Loss and accuracy for monitoring the training
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    accuracy_op = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))

    # Optimizer and optimizer operation
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    with tf.control_dependencies(update_ops):
        optimize_op = optimizer.minimize(loss_op)

    vars_to_initialize += optimizer.variables()

    # Initialize the new layer and the optimizer variables
    sess.run(tf.variables_initializer(vars_to_initialize))


    # Create the inputs for the data input placeholder
    train_input = sess.run(train_iterator.string_handle())
    val_input = sess.run(train_iterator.string_handle())


    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(FLAGS.summaries_path, sess.graph)


    best_val_loss = 1e8
    #best_val_acc = 0.0


    for epoch in range(FLAGS.num_epochs):
        train_loss, val_loss = 0.0, 0.0
        train_acc, val_acc = 0.0, 0.0

        for step in range(1, FLAGS.train_steps + 1):
            _, loss, accuracy = sess.run([optimize_op, loss_op, accuracy_op], feed_dict={input_data: train_input, training: True, dropout_prob: FLAGS.dropout_prob})

            train_loss += np.sum(loss)
            train_acc += accuracy

            if step % FLAGS.display_steps == 0:
                print("Epoch: {}    Step: {}    Loss: {}    Acc: {}".format(epoch, step, train_loss / step, train_acc / step))

        summary_writer.add_summary(get_summary({'train_loss': train_loss / FLAGS.train_steps, 'train_accuracy': train_acc / FLAGS.train_steps}), epoch)


        sess.run(val_iterator.initializer)

        for step in range(1, FLAGS.val_steps + 1):
            loss, accuracy = sess.run([loss_op, accuracy_op], feed_dict={input_data: val_input, training: False, dropout_prob: 1.0})

            val_loss += np.sum(loss)
            val_acc += accuracy

        mean_val_loss = val_loss / FLAGS.val_steps
        mean_val_acc = val_acc / FLAGS.val_steps

        print('\n\nVALIDATION\n\nLoss: {}    Acc: {}\n\n'.format(mean_val_loss, mean_val_acc))

        if mean_val_loss < best_val_loss:
            print('LOSS IMPROVED FROM: {}   TO: {}\n\n'.format(best_val_loss, mean_val_loss))
            best_val_loss = mean_val_loss
            saver.save(sess, FLAGS.checkpoints_model)

        summary_writer.add_summary(get_summary({'val_loss': mean_val_loss, 'val_accuracy': mean_val_acc}), epoch)
        summary_writer.flush()



def main():
    # Train data preprocessing stack
    train_data_preprocess = PreprocessStack(
        Clip(FLAGS.num_frames),
        RandomCrop(FLAGS.num_frames, FLAGS.frame_size, FLAGS.num_channels),
        Flip(),
        Normalize()
    )

    # Validation data preprocessing stack
    val_data_preprocess = PreprocessStack(
        #Clip(FLAGS.num_frames),
        CenterCrop(FLAGS.num_frames, FLAGS.frame_size, FLAGS.num_channels),
        Normalize()
    )

    # Load and preprocessing of the training dataset
    train_data = load_from_record(FLAGS.train_data_pattern, shuffle_buffer=100, repeat=True,
                                batch_size=FLAGS.batch_size, preprocess=train_data_preprocess)
    val_data = load_from_record(FLAGS.val_data_pattern, shuffle_buffer=None, repeat=False,
                                batch_size=FLAGS.batch_size, preprocess=val_data_preprocess)


    def build_inception_fn(sess, inputs, training, dropout_prob):

        vars_to_initialize = []

        with tf.variable_scope('inception'):
            if FLAGS.from_kinetics:
                # Initialize the i3d model with the kinetics weights
                net_out, net = load_kinetics_weights(sess, inputs, FLAGS.kinetics_model, net_type="RGB", training=training, dropout_prob=dropout_prob, logits=False)
                logits = InceptionI3d.add_logits(net_out, FLAGS.num_classes, training=training, name='Logits')

                vars_to_initialize += [var for var in tf.global_variables() if 'Logits' in var.name]

            else:
                # Create a blank model with FLAGS.num_classes
                model = InceptionI3d(num_classes=FLAGS.num_classes)
                logits, net = model(inputs, training=training, dropout_prob=dropout_prob, logits=True, logits_name='Logits')

                if FLAGS.from_checkpoints:
                    # Restore from a checkpoint with the same number of classes
                    restorer = tf.train.Saver(reshape=True)
                    restorer.restore(sess, FLAGS.checkpoints_model)

                else:
                    # If there is no checkpoint, initialize everything
                    vars_to_initialize += [tf.global_variables()]

        return logits, vars_to_initialize


    def build_mobilenet_fn(sess, inputs, training, dropout_prob):
        vars_to_initialize = []

        net_out, net = load_imagenet_weights(sess, inputs, '../data/mobilenet/mobilenet_v1_1.0_224.ckpt', 
                                             add_logits=False, training=training, dropout_prob=dropout_prob)
        logits = add_mobilenet_logits(net_out, FLAGS.num_classes, 'Logits')

        vars_to_initialize += [var for var in tf.global_variables() if 'Logits' in var.name]

        return logits, vars_to_initialize



    train(train_data, val_data, build_mobilenet_fn)


if __name__ == '__main__':
    main()