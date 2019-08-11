"""
Derived from: https://github.com/ry/tensorflow-resnet
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages


NUM_BLOCKS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
FC_WEIGHT_STDDEV = 0.01


class ResNetModel(object):

    def __init__(self, is_training, depth=50, num_classes=1000):
        self.is_training = is_training
        self.num_classes = num_classes
        self.depth = depth

        if depth in NUM_BLOCKS:
            self.num_blocks = NUM_BLOCKS[depth]
        else:
            raise ValueError('Depth is not supported; it must be 50, 101 or 152')


    def inference(self, x):
        # Scale 1
        img_rs = tf.reshape(x, [-1, 88, 88, 3])
        with tf.variable_scope('img_scale1'):
            s1_bn = bn(img_rs, is_training=self.is_training)
            s1_conv = conv(s1_bn, ksize=3, stride=1, filters_out=64)
            s1 = tf.nn.relu(s1_conv)
            # Scale 2
        with tf.variable_scope('img_scale2'):
            s2_mp = tf.nn.max_pool(s1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            s2_bn = bn(s2_mp, is_training=self.is_training)
            s2_conv = conv(s2_bn, ksize=3, stride=1, filters_out=128)
            s2 = tf.nn.relu(s2_conv)
        # Scale 3
        with tf.variable_scope('img_scale3'):
            s3_mp = tf.nn.max_pool(s2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            s3_bn = bn(s3_mp, is_training=self.is_training)
            s3_conv = conv(s3_bn, ksize=3, stride=1, filters_out=256)
            s3 = tf.nn.relu(s3_conv)
        s4_mp = tf.nn.max_pool(s3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # post-net
        avg_pool = tf.reduce_mean(s4_mp, reduction_indices=[1, 2], name='avg_pool2')


        return avg_pool

    def loss(self, batch_x, batch_y=None):
        y_predict = self.inference(batch_x)
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=batch_y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([cross_entropy_mean] + regularization_losses)
        return self.loss

    def optimize(self, loss, learning_rate, train_layers=[]):
        trainable_var_names = ['weights', 'biases', 'beta', 'gamma']
        var_list = [v for v in tf.trainable_variables() if
            v.name.split(':')[0].split('/')[-1] in trainable_var_names and
            contains(v.name, train_layers)]
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list)

        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss]))

        batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)

        return tf.group(train_op, batchnorm_updates_op)

    def load_original_weights(self, weight_path, session):
        weights_path = weight_path + 'ResNet-L{}.npy'.format(self.depth)
        weights_dict = np.load(weights_path, encoding='bytes').item()

        for op_name in weights_dict:
            parts = op_name.split('/')

            # if contains(op_name, skip_layers):
            #     continue

            if parts[0] == 'fc' and self.num_classes != 1000:
                continue

            full_name = "{}:0".format(op_name)
            var = [v for v in tf.global_variables() if v.name == full_name][0]
            session.run(var.assign(weights_dict[op_name]))


"""
Helper methods
"""
def _get_variable(name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
    "A little wrapper around tf.get_variable to do weight decay"

    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer,
                           trainable=trainable)

def conv(x, ksize, stride, filters_out):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=shape, dtype='float', initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

def bn(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

def stack(x, is_training, num_blocks, stack_stride, block_filters_internal):
    for n in range(num_blocks):
        block_stride = stack_stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x, is_training, block_filters_internal=block_filters_internal, block_stride=block_stride)
    return x


def block(x, is_training, block_filters_internal, block_stride):
    filters_in = x.get_shape()[-1]

    m = 4
    filters_out = m * block_filters_internal
    shortcut = x

    with tf.variable_scope('a'):
        a_conv = conv(x, ksize=1, stride=block_stride, filters_out=block_filters_internal)
        a_bn = bn(a_conv, is_training)
        a = tf.nn.relu(a_bn)

    with tf.variable_scope('b'):
        b_conv = conv(a, ksize=3, stride=1, filters_out=block_filters_internal)
        b_bn = bn(b_conv, is_training)
        b = tf.nn.relu(b_bn)

    with tf.variable_scope('c'):
        c_conv = conv(b, ksize=1, stride=1, filters_out=filters_out)
        c = bn(c_conv, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or block_stride != 1:
            shortcut_conv = conv(x, ksize=1, stride=block_stride, filters_out=filters_out)
            shortcut = bn(shortcut_conv, is_training)

    return tf.nn.relu(c + shortcut)


def fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)
    weights = _get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer())
    return tf.nn.xw_plus_b(x, weights, biases)

def contains(target_str, search_arr):
    rv = False

    for search_str in search_arr:
        if search_str in target_str:
            rv = True
            break

    return rv


def visit1_network(visit, is_training):
    # visit shape[174, 26]
    # 对visit数据进行转换
    visit_rs = tf.reshape(visit, [-1, 174, 24, 1])
    with tf.variable_scope('visit1_scale1'):
        s1_bn = bn(visit_rs, is_training=is_training)
        s1_conv = conv(s1_bn, ksize=3, stride=1, filters_out=64)
        s1 = tf.nn.relu(s1_conv)
        # Scale 2
    with tf.variable_scope('visit1_scale2'):
        s2_mp = tf.nn.max_pool(s1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        s2_bn = bn(s2_mp, is_training=is_training)
        s2_conv = conv(s2_bn, ksize=3, stride=1, filters_out=128)
        s2 = tf.nn.relu(s2_conv)
    # Scale 3
    with tf.variable_scope('visit1_scale3'):
        s3_mp = tf.nn.max_pool(s2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        s3_bn = bn(s3_mp, is_training=is_training)
        s3_conv = conv(s3_bn, ksize=3, stride=2, filters_out=512)
        s3 = tf.nn.relu(s3_conv)
    #s4_mp = tf.nn.max_pool(s3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # post-net
    avg_pool = tf.reduce_mean(s3, reduction_indices=[1, 2], name='avg_pool2')

    return avg_pool
def visit2_network(visit, is_training):
    # visit shape[174, 26]
    # 对visit数据进行转换
    visit_rs = tf.reshape(visit, [-1, 174, 24, 1])
    with tf.variable_scope('visit2_scale1'):
        s1_bn = bn(visit_rs, is_training=is_training)
        s1_conv = conv(s1_bn, ksize=3, stride=1, filters_out=32)
        s1 = tf.nn.relu(s1_conv)
        # Scale 2
    with tf.variable_scope('visit2_scale2'):
        s2_mp = tf.nn.max_pool(s1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        s2_bn = bn(s2_mp, is_training=is_training)
        s2_conv = conv(s2_bn, ksize=3, stride=1, filters_out=64)
        s2 = tf.nn.relu(s2_conv)
    # Scale 3
    with tf.variable_scope('visit2_scale3'):
        s3_mp = tf.nn.max_pool(s2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        s3_bn = bn(s3_mp, is_training=is_training)
        s3_conv = conv(s3_bn, ksize=3, stride=2, filters_out=128)
        s3 = tf.nn.relu(s3_conv)
    #s4_mp = tf.nn.max_pool(s3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # post-net
    avg_pool = tf.reduce_mean(s3, reduction_indices=[1, 2], name='avg_pool2')

    return avg_pool
def get_net_output(fc_image, fc_visit1, fc_visit2, classNum, KEEP_PROB):
    with tf.variable_scope("fc"):
        """
         visit_image_concat = tf.concat([fc_image, fc_visit], 1, name='visit_image_concat')
        net_output = fc(visit_image_concat, num_units_out=classNum)
        """
        visit1_drop = dropout(fc_visit1, KEEP_PROB)
        visit2_drop = dropout(fc_visit2, KEEP_PROB)
        fc_image_drop = dropout(fc_image, KEEP_PROB)
        visit_image_concat = tf.concat([fc_image_drop, visit1_drop, visit2_drop], 1, name='visit_image_concat')
        net_output = fc(visit_image_concat, num_units_out=classNum)
    return net_output
def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)