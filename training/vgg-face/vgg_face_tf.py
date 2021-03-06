import tensorflow as tf

__weights_dict = dict()

is_train = False

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    data            = tf.placeholder(tf.float32, shape = (None, 224, 224, 3), name = 'data')
    conv1_1_pad     = tf.pad(data, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv1_1         = convolution(conv1_1_pad, group=1, strides=[1, 1], padding='VALID', name='conv1_1')
    relu1_1         = tf.nn.relu(conv1_1, name = 'relu1_1')
    conv1_2_pad     = tf.pad(relu1_1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv1_2         = convolution(conv1_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv1_2')
    relu1_2         = tf.nn.relu(conv1_2, name = 'relu1_2')
    pool1_pad       = tf.pad(relu1_2, paddings = [[0, 0], [0, 1], [0, 1], [0, 0]], constant_values=float('-Inf'))
    pool1           = tf.nn.max_pool(pool1_pad, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool1')
    conv2_1_pad     = tf.pad(pool1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv2_1         = convolution(conv2_1_pad, group=1, strides=[1, 1], padding='VALID', name='conv2_1')
    relu2_1         = tf.nn.relu(conv2_1, name = 'relu2_1')
    conv2_2_pad     = tf.pad(relu2_1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv2_2         = convolution(conv2_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv2_2')
    relu2_2         = tf.nn.relu(conv2_2, name = 'relu2_2')
    pool2_pad       = tf.pad(relu2_2, paddings = [[0, 0], [0, 1], [0, 1], [0, 0]], constant_values=float('-Inf'))
    pool2           = tf.nn.max_pool(pool2_pad, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool2')
    conv3_1_pad     = tf.pad(pool2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv3_1         = convolution(conv3_1_pad, group=1, strides=[1, 1], padding='VALID', name='conv3_1')
    relu3_1         = tf.nn.relu(conv3_1, name = 'relu3_1')
    conv3_2_pad     = tf.pad(relu3_1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv3_2         = convolution(conv3_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv3_2')
    relu3_2         = tf.nn.relu(conv3_2, name = 'relu3_2')
    conv3_3_pad     = tf.pad(relu3_2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv3_3         = convolution(conv3_3_pad, group=1, strides=[1, 1], padding='VALID', name='conv3_3')
    relu3_3         = tf.nn.relu(conv3_3, name = 'relu3_3')
    pool3_pad       = tf.pad(relu3_3, paddings = [[0, 0], [0, 1], [0, 1], [0, 0]], constant_values=float('-Inf'))
    pool3           = tf.nn.max_pool(pool3_pad, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool3')
    conv4_1_pad     = tf.pad(pool3, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_1         = convolution(conv4_1_pad, group=1, strides=[1, 1], padding='VALID', name='conv4_1')
    relu4_1         = tf.nn.relu(conv4_1, name = 'relu4_1')
    conv4_2_pad     = tf.pad(relu4_1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_2         = convolution(conv4_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv4_2')
    relu4_2         = tf.nn.relu(conv4_2, name = 'relu4_2')
    conv4_3_pad     = tf.pad(relu4_2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv4_3         = convolution(conv4_3_pad, group=1, strides=[1, 1], padding='VALID', name='conv4_3')
    relu4_3         = tf.nn.relu(conv4_3, name = 'relu4_3')
    pool4_pad       = tf.pad(relu4_3, paddings = [[0, 0], [0, 1], [0, 1], [0, 0]], constant_values=float('-Inf'))
    pool4           = tf.nn.max_pool(pool4_pad, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool4')
    conv5_1_pad     = tf.pad(pool4, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_1         = convolution(conv5_1_pad, group=1, strides=[1, 1], padding='VALID', name='conv5_1')
    relu5_1         = tf.nn.relu(conv5_1, name = 'relu5_1')
    conv5_2_pad     = tf.pad(relu5_1, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_2         = convolution(conv5_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv5_2')
    relu5_2         = tf.nn.relu(conv5_2, name = 'relu5_2')
    conv5_3_pad     = tf.pad(relu5_2, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv5_3         = convolution(conv5_3_pad, group=1, strides=[1, 1], padding='VALID', name='conv5_3')
    relu5_3         = tf.nn.relu(conv5_3, name = 'relu5_3')
    pool5_pad       = tf.pad(relu5_3, paddings = [[0, 0], [0, 1], [0, 1], [0, 0]], constant_values=float('-Inf'))
    pool5           = tf.nn.max_pool(pool5_pad, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool5')
    fc6_0           = tf.contrib.layers.flatten(pool5)
    fc6_1           = tf.layers.dense(fc6_0, 4096, kernel_initializer = tf.constant_initializer(__weights_dict['fc6_1']['weights']), bias_initializer = tf.constant_initializer(__weights_dict['fc6_1']['bias']), use_bias = True)
    relu6           = tf.nn.relu(fc6_1, name = 'relu6')
    fc7_0           = tf.contrib.layers.flatten(relu6)
    fc7_1           = tf.layers.dense(fc7_0, 4096, kernel_initializer = tf.constant_initializer(__weights_dict['fc7_1']['weights']), bias_initializer = tf.constant_initializer(__weights_dict['fc7_1']['bias']), use_bias = True)
    relu7           = tf.nn.relu(fc7_1, name = 'relu7')
    fc8_0           = tf.contrib.layers.flatten(relu7)
    fc8_1           = tf.layers.dense(fc8_0, 2622, kernel_initializer = tf.constant_initializer(__weights_dict['fc8_1']['weights']), bias_initializer = tf.constant_initializer(__weights_dict['fc8_1']['bias']), use_bias = True)
    prob            = tf.nn.softmax(fc8_1, name = 'prob')
    return data, prob


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer
