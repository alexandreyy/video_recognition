import tensorflow as tf
import numpy as np
import cv2


def sep_conv_3d(inputs, output_filters, kernel_size, strides=[1, 1, 1, 1, 1], training=False, padding='SAME'):
    input_shape = inputs.get_shape().as_list()
    input_filters = input_shape[-1]


    depth_weights = tf.Variable(np.ones((kernel_size[0], kernel_size[1], kernel_size[2], input_filters, 1), dtype=np.float32))

    res = [tf.nn.conv3d(tf.expand_dims(inputs[..., i], -1), tf.expand_dims(depth_weights[..., i, :], -1), strides, padding=padding) for i in range(input_filters)]
    res = tf.concat(res, len(input_shape) - 1)
    res = tf.layers.batch_normalization(res, training=training, scale=True)
    res = tf.nn.relu6(res)


    point_weights = tf.Variable(np.ones((1, 1, 1, input_filters, output_filters), dtype=np.float32))
    
    res = tf.nn.conv3d(res, point_weights, [1, 1, 1, 1, 1], padding='SAME')
    res = tf.layers.batch_normalization(res, training=training, scale=True)
    res = tf.nn.relu6(res)

    return res


def get_assign(new, org):
    assign = []

    for x, y in zip(new, org):
        if 'kernel' in x.name or 'Variable' in x.name:
            assign += [ tf.assign(x, tf.tile(tf.expand_dims(y, 0), [x.shape[0].value] + [1] * len(y.shape)) / x.shape[0].value ) ]
        else:
            assign += [ tf.assign(x, y) ]

    return assign



def add_mobilenet_logits(net_out, num_classes=1001, name='Logits'):
    logits = tf.layers.conv3d(net_out, num_classes, (1, 1, 1), (1, 1, 1), padding='same', use_bias=True, name=name)
    logits = tf.squeeze(logits, [2, 3])
    logits = tf.reduce_mean(logits, axis=1)

    return logits


def mobilenet(inputs, training=False, dropout_prob=1.0, add_logits=True):
    with tf.variable_scope('mobilenet'):
        
        net = []
        net += [ tf.layers.conv3d(inputs, 32, (3, 3, 3), (1, 2, 2), padding='same', use_bias=False) ]
        net += [ tf.layers.batch_normalization(net[-1], training=training, scale=True) ]
        net += [ tf.nn.relu6(net[-1]) ]

        net += [ sep_conv_3d(net[-1], 64,   (3, 3, 3), [1, 1, 1, 1, 1], training) ]
        net += [ sep_conv_3d(net[-1], 128,  (3, 3, 3), [1, 1, 2, 2, 1], training) ]
        net += [ sep_conv_3d(net[-1], 128,  (3, 3, 3), [1, 1, 1, 1, 1], training) ]
        net += [ sep_conv_3d(net[-1], 256,  (3, 3, 3), [1, 1, 2, 2, 1], training) ]
        net += [ sep_conv_3d(net[-1], 256,  (3, 3, 3), [1, 1, 1, 1, 1], training) ]
        net += [ sep_conv_3d(net[-1], 512,  (3, 3, 3), [1, 1, 2, 2, 1], training) ]
        net += [ sep_conv_3d(net[-1], 512,  (3, 3, 3), [1, 1, 1, 1, 1], training) ]
        net += [ sep_conv_3d(net[-1], 512,  (3, 3, 3), [1, 1, 1, 1, 1], training) ]
        net += [ sep_conv_3d(net[-1], 512,  (3, 3, 3), [1, 1, 1, 1, 1], training) ]
        net += [ sep_conv_3d(net[-1], 512,  (3, 3, 3), [1, 1, 1, 1, 1], training) ]
        net += [ sep_conv_3d(net[-1], 512,  (3, 3, 3), [1, 1, 1, 1, 1], training) ]
        net += [ sep_conv_3d(net[-1], 1024, (3, 3, 3), [1, 1, 2, 2, 1], training) ]
        net += [ sep_conv_3d(net[-1], 1024, (3, 3, 3), [1, 1, 1, 1, 1], training) ]

        net += [ tf.layers.average_pooling3d(net[-1], (1, 7, 7), (1, 1, 1)) ]
        net += [ tf.nn.dropout(net[-1], dropout_prob) ]

        if add_logits:
            net += [ add_mobilenet_logits(net[-1], 1001) ]

        return net[-1], net



def load_imagenet_weights(sess, inputs, weights, add_logits=True, training=False, dropout_prob=1.0):
    logits, net = mobilenet(inputs, training, dropout_prob, add_logits)
    
    restorer = tf.train.import_meta_graph(weights + '.meta', clear_devices=True)
    restorer.restore(sess, weights)

    end_point = 137 if add_logits else 135

    org_vars = [v for v in sess.graph.get_collection('variables') if 'MobilenetV1' in v.name][:end_point]
    new_vars = [v for v in sess.graph.get_collection('variables') if 'mobilenet' in v.name]


    assign = get_assign(new_vars, org_vars)

    sess.run(assign)

    return logits, net




# img = cv2.imread('t1.jpeg')
# img = cv2.resize(img, (224, 224))
# img = img.astype(np.float32)
# img = img.reshape((1, 1, 224, 224, 3))
# img = np.tile(img, (1, 32, 1, 1, 1))
# img = 2 * ((img / 255) - 0.5)
# 
# 
# 
# 
# inputs = tf.constant(img, dtype=tf.float32)
# 
# sess = tf.session()
# 
# 
# net, _ = load_imagenet_weights(sess, inputs, '../../data/mobilenet/mobilenet_v1_1.0_224.ckpt', false)
# 
# 
# res = sess.run(net)[0]
# 
# print(res.shape, '\n\n')
# print(res)

# for r in res:
#     pos = np.argsort(r)[::-1]
#     print(pos[:5]-1)
#     print(r[pos[:5]], '\n\n')