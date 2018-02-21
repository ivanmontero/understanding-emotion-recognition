import keras
from keras.models import Model
from keras import layers
import keras.backend as K

def load_weights(model, weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    for layer in model.layers:
        if layer.name in weights_dict:
            cur_dict = weights_dict[layer.name]
            current_layer_parameters = list()
            if layer.__class__.__name__ == "BatchNormalization":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
                current_layer_parameters.extend([cur_dict['mean'], cur_dict['var']])
            elif layer.__class__.__name__ == "SeparableConv2D":
                current_layer_parameters = [cur_dict['depthwise_filter'], cur_dict['pointwise_filter']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            else:
                # rot weights
                current_layer_parameters = [cur_dict['weights']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            model.get_layer(layer.name).set_weights(current_layer_parameters)

    return model

def KitModel(weight_file = None):
        
    data            = layers.Input(name = 'data', shape = (224, 224, 3,) )
    conv1           = layers.Conv2D(name='conv1', filters = 96, kernel_size=((7, 7)), strides=((2, 2)), padding='valid', use_bias=True)(data)
    relu1           = layers.Activation(name = 'relu1', activation = 'relu')(conv1)
    norm1           = LRN(size = 3, alpha = 0.0005000000237487257, beta = 0.75, k = 1.0, name = 'norm1')(relu1)
    pool1_input     = layers.ZeroPadding2D(padding = ((0, 2), (0, 2)))(norm1)
    pool1           = layers.MaxPooling2D(name = 'pool1', pool_size = (3, 3), strides = (3, 3), padding = 'valid')(pool1_input)
    conv2_input     = layers.ZeroPadding2D(padding = ((2, 2), (2, 2)))(pool1)
    conv2           = layers.Conv2D(name='conv2', filters = 256, kernel_size=((5, 5)), strides=((1, 1)), padding='valid', use_bias=True)(conv2_input)
    relu2           = layers.Activation(name = 'relu2', activation = 'relu')(conv2)
    pool2_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu2)
    pool2           = layers.MaxPooling2D(name = 'pool2', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool2_input)
    conv3_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool2)
    conv3           = layers.Conv2D(name='conv3', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv3_input)
    relu3           = layers.Activation(name = 'relu3', activation = 'relu')(conv3)
    conv4_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu3)
    conv4           = layers.Conv2D(name='conv4', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv4_input)
    relu4           = layers.Activation(name = 'relu4', activation = 'relu')(conv4)
    conv5_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu4)
    conv5           = layers.Conv2D(name='conv5', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv5_input)
    relu5           = layers.Activation(name = 'relu5', activation = 'relu')(conv5)
    pool5_input     = layers.ZeroPadding2D(padding = ((0, 2), (0, 2)))(relu5)
    pool5           = layers.MaxPooling2D(name = 'pool5', pool_size = (3, 3), strides = (3, 3), padding = 'valid')(pool5_input)
    fc6_0           = __flatten(name = 'fc6_0', input = pool5)
    fc6_1           = layers.Dense(name = 'fc6_1', units = 4048, use_bias = True)(fc6_0)
    relu6           = layers.Activation(name = 'relu6', activation = 'relu')(fc6_1)
    drop6           = layers.Dropout(name = 'drop6', rate = 0.5, seed = None)(relu6)
    fc7_0           = __flatten(name = 'fc7_0', input = drop6)
    fc7_1           = layers.Dense(name = 'fc7_1', units = 4048, use_bias = True)(fc7_0)
    relu7           = layers.Activation(name = 'relu7', activation = 'relu')(fc7_1)
    drop7           = layers.Dropout(name = 'drop7', rate = 0.5, seed = None)(relu7)
    fc8_cat_0       = __flatten(name = 'fc8_cat_0', input = drop7)
    fc8_cat_1       = layers.Dense(name = 'fc8_cat_1', units = 7, use_bias = True)(fc8_cat_0)
    prob            = layers.Activation(name = 'prob', activation = 'softmax')(fc8_cat_1)
    model           = Model(inputs = [data], outputs = [prob])
    load_weights(model, weight_file)
    return model

from keras.layers.core import Layer
class LRN(Layer):

    def __init__(self, size=5, alpha=0.0005, beta=0.75, k=2, **kwargs):
        self.n = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LRN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LRN, self).build(input_shape)

    def call(self, x, mask=None):
        half_n = self.n - 1
        squared = K.square(x)
        scale = self.k
        norm_alpha = self.alpha / (2 * half_n + 1)
        if K.image_dim_ordering() == "th":
            b, f, r, c = self.shape
            squared = K.expand_dims(squared, 0)
            squared = K.spatial_3d_padding(squared, padding=((half_n, half_n), (0, 0), (0,0)))
            squared = K.squeeze(squared, 0)
            for i in range(half_n*2+1):
                scale += norm_alpha * squared[:, i:i+f, :, :]
        else:
            b, r, c, f = self.shape
            squared = K.expand_dims(squared, -1)
            squared = K.spatial_3d_padding(squared, padding=((0, 0), (0,0), (half_n, half_n)))
            squared = K.squeeze(squared, -1)
            for i in range(half_n*2+1):
                scale += norm_alpha * squared[:, :, :, i:i+f]

        scale = K.pow(scale, self.beta)
        return x / scale

    def compute_output_shape(self, input_shape):
        return input_shape

def __flatten(name, input):
    if input.shape.ndims > 2: return layers.Flatten(name = name)(input)
    else: return input

