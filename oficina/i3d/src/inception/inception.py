import tensorflow as tf
import os


def conv3d (inputs, filters, kernel_size=[1, 1, 1], strides=[1, 1, 1], activation=tf.nn.relu, 
            regularizer_weight=1e-7, use_bn=True, use_bias=False, training=False, padding="same", name="Conv3D"):

    with tf.variable_scope(name):
        net = tf.layers.conv3d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                               use_bias=use_bias, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-7))

        if use_bn:
            net = tf.layers.batch_normalization(net, training=training, scale=False)

        net = activation(net)

    return net


def inception3d (inputs, filters, training=False, name="Inception3d"):
    with tf.variable_scope(name):

        with tf.variable_scope("Branch_0"):
            branch_0 = conv3d(inputs, filters[0][0], kernel_size=[1, 1, 1], training=training, name="Conv3d_0a_1x1")
        
        with tf.variable_scope("Branch_1"):
            branch_1 = conv3d(inputs, filters[1][0], kernel_size=[1, 1, 1], training=training, name="Conv3d_0a_1x1")
            branch_1 = conv3d(branch_1, filters[1][1], kernel_size=[3, 3, 3], training=training, name="Conv3d_0b_3x3")

        with tf.variable_scope("Branch_2"):
            branch_2 = conv3d(inputs, filters[2][0], kernel_size=[1, 1, 1], training=training, name="Conv3d_0a_1x1")
            branch_2 = conv3d(branch_2, filters[2][1], kernel_size=[3, 3, 3], training=training, name="Conv3d_0b_3x3")

        with tf.variable_scope("Branch_3"):
            branch_3 = tf.nn.max_pool3d(inputs, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding="SAME", name="MaxPool3d_0a_3x3")
            branch_3 = conv3d(branch_3, filters[3][0], kernel_size=[1, 1, 1], training=training, name="Conv3d_0b_1x1")

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
    
    return net



def convert_kinetics_variable_name (var_name, scope, net_type='RGB'):
    replaces = [
        (scope, 'inception_i3d'),
        ('conv3d/kernel', 'conv_3d/w'),
        ('conv3d/bias', 'conv_3d/b'),
        ('batch_normalization', 'batch_norm'),
        ('Mixed_5b/Branch_2/Conv3d_0b_3x3', 'Mixed_5b/Branch_2/Conv3d_0a_3x3'),
        (':0', '')
    ]

    new_name = var_name

    for replace in replaces:
        new_name = new_name.replace(*replace)

    new_name = net_type + '/' + new_name

    return new_name



def load_kinetics_weights(sess, input_tensor, weights, num_classes=400, net_type='RGB', 
                          training=False, dropout_prob=1.0, logits=True):
    model = InceptionI3d(num_classes=num_classes)
    logits, endpoints = model(input_tensor, training=training, dropout_prob=dropout_prob, logits=logits)

    variable_map = {}

    for variable in tf.global_variables():
        variable_map[convert_kinetics_variable_name(variable.name, sess.graph.get_name_scope(), net_type)] = variable

    saver = tf.train.Saver(var_list=variable_map, reshape=True)

    saver.restore(sess, weights)

    return logits, endpoints



class InceptionI3d ():

    SIZE = 224

    ENDPOINTS = (
        'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3', 
        'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 
        'Mixed_4f', 'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'Logits', 'Predictions',
    )

    def __init__ (self, num_classes=400, frame_reduction=2):
        self.num_classes = num_classes
        self.frame_reduction = frame_reduction

    def __call__(self, inputs, training=False, dropout_prob=1.0, logits=True, logits_name='Conv3d_0c_1x1'):
        input_shape = inputs.get_shape().as_list()

        if input_shape[2] < InceptionI3d.SIZE or input_shape[3] < InceptionI3d.SIZE or input_shape[2] % 32 != 0 or input_shape[3] % 32 != 0:
            raise ValueError("Invalid input video. Minimum dimensions must be: (4, 224, 224) - (frames, width, height), \
                                where width and height must be multiples of 32")

        spatial_squeeze = (input_shape[2] == InceptionI3d.SIZE) and (input_shape[3] == InceptionI3d.SIZE)

        net =  [ inputs ]
        net += [ conv3d(net[-1], 64, kernel_size=[7, 7, 7], strides=[2, 2, 2], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ tf.nn.max_pool3d(net[-1], ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding="SAME", name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ conv3d(net[-1], 64, kernel_size=[1, 1, 1], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ conv3d(net[-1], 192, kernel_size=[3, 3, 3], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ tf.nn.max_pool3d(net[-1], ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding="SAME", name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ inception3d(net[-1], [[64], [96, 128], [16, 32], [32]], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ inception3d(net[-1], [[128], [128, 192], [32, 96], [64]], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ tf.nn.max_pool3d(net[-1], ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding="SAME", name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ inception3d(net[-1], [[192], [96, 208], [16, 48], [64]], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ inception3d(net[-1], [[160], [112, 224], [24, 64], [64]], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ inception3d(net[-1], [[128], [128, 256], [24, 64], [64]], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ inception3d(net[-1], [[112], [144, 288], [32, 64], [64]], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ inception3d(net[-1], [[256], [160, 320], [32, 128], [128]], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ tf.nn.max_pool3d(net[-1], ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding="SAME", name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ inception3d(net[-1], [[256], [160, 320], [32, 128], [128]], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]
        net += [ inception3d(net[-1], [[384], [192, 384], [48, 128], [128]], training=training, name=InceptionI3d.ENDPOINTS[len(net)-1]) ]

        with tf.variable_scope(InceptionI3d.ENDPOINTS[len(net)-1]):
            net += [ tf.nn.avg_pool3d(net[-1], ksize=[1, self.frame_reduction, 7, 7, 1], strides=[1, 1, 1, 1, 1], padding="VALID") ]
            net += [ tf.nn.dropout(net[-1], dropout_prob) ]
            
            if not logits or not self.num_classes:
                return net[-1], net
            
            logits = InceptionI3d.add_logits(net[-1], self.num_classes, training=training, name=logits_name, spatial_squeeze=spatial_squeeze)
            net += [ logits ]
    
        return logits, net

    @staticmethod
    def add_logits(inputs, num_classes, training=False, name='Logits', spatial_squeeze=True):
        logits = conv3d(inputs, num_classes, kernel_size=[1, 1, 1], activation=tf.identity, use_bn=False, use_bias=True, training=training, name=name)
        
        if spatial_squeeze:
            logits = tf.squeeze(logits, [2, 3], name="SpatialSqueeze")

        logits = tf.reduce_mean(logits, axis=1)

        return logits
