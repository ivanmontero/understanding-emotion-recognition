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
    conv1_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(data)
    conv1_1         = layers.Conv2D(name='conv1_1', filters = 64, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv1_1_input)
    relu1_1         = layers.Activation(name = 'relu1_1', activation = 'relu')(conv1_1)
    conv1_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu1_1)
    conv1_2         = layers.Conv2D(name='conv1_2', filters = 64, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv1_2_input)
    relu1_2         = layers.Activation(name = 'relu1_2', activation = 'relu')(conv1_2)
    pool1_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu1_2)
    pool1           = layers.MaxPooling2D(name = 'pool1', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool1_input)
    conv2_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool1)
    conv2_1         = layers.Conv2D(name='conv2_1', filters = 128, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv2_1_input)
    relu2_1         = layers.Activation(name = 'relu2_1', activation = 'relu')(conv2_1)
    conv2_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu2_1)
    conv2_2         = layers.Conv2D(name='conv2_2', filters = 128, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv2_2_input)
    relu2_2         = layers.Activation(name = 'relu2_2', activation = 'relu')(conv2_2)
    pool2_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu2_2)
    pool2           = layers.MaxPooling2D(name = 'pool2', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool2_input)
    conv3_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool2)
    conv3_1         = layers.Conv2D(name='conv3_1', filters = 256, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv3_1_input)
    relu3_1         = layers.Activation(name = 'relu3_1', activation = 'relu')(conv3_1)
    conv3_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu3_1)
    conv3_2         = layers.Conv2D(name='conv3_2', filters = 256, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv3_2_input)
    relu3_2         = layers.Activation(name = 'relu3_2', activation = 'relu')(conv3_2)
    conv3_3_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu3_2)
    conv3_3         = layers.Conv2D(name='conv3_3', filters = 256, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv3_3_input)
    relu3_3         = layers.Activation(name = 'relu3_3', activation = 'relu')(conv3_3)
    pool3_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu3_3)
    pool3           = layers.MaxPooling2D(name = 'pool3', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool3_input)
    conv4_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool3)
    conv4_1         = layers.Conv2D(name='conv4_1', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv4_1_input)
    relu4_1         = layers.Activation(name = 'relu4_1', activation = 'relu')(conv4_1)
    conv4_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu4_1)
    conv4_2         = layers.Conv2D(name='conv4_2', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv4_2_input)
    relu4_2         = layers.Activation(name = 'relu4_2', activation = 'relu')(conv4_2)
    conv4_3_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu4_2)
    conv4_3         = layers.Conv2D(name='conv4_3', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv4_3_input)
    relu4_3         = layers.Activation(name = 'relu4_3', activation = 'relu')(conv4_3)
    pool4_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu4_3)
    pool4           = layers.MaxPooling2D(name = 'pool4', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool4_input)
    conv5_1_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool4)
    conv5_1         = layers.Conv2D(name='conv5_1', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv5_1_input)
    relu5_1         = layers.Activation(name = 'relu5_1', activation = 'relu')(conv5_1)
    conv5_2_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu5_1)
    conv5_2         = layers.Conv2D(name='conv5_2', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv5_2_input)
    relu5_2         = layers.Activation(name = 'relu5_2', activation = 'relu')(conv5_2)
    conv5_3_input   = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu5_2)
    conv5_3         = layers.Conv2D(name='conv5_3', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv5_3_input)
    relu5_3         = layers.Activation(name = 'relu5_3', activation = 'relu')(conv5_3)
    pool5_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu5_3)
    pool5           = layers.MaxPooling2D(name = 'pool5', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool5_input)
    fc6_0           = __flatten(name = 'fc6_0', input = pool5)
    fc6_1           = layers.Dense(name = 'fc6_1', units = 4096, use_bias = True)(fc6_0)
    relu6           = layers.Activation(name = 'relu6', activation = 'relu')(fc6_1)
    drop6           = layers.Dropout(name = 'drop6', rate = 0.5, seed = None)(relu6)
    fc7_0           = __flatten(name = 'fc7_0', input = drop6)
    fc7_1           = layers.Dense(name = 'fc7_1', units = 4096, use_bias = True)(fc7_0)
    relu7           = layers.Activation(name = 'relu7', activation = 'relu')(fc7_1)
    drop7           = layers.Dropout(name = 'drop7', rate = 0.5, seed = None)(relu7)
    fc8_0           = __flatten(name = 'fc8_0', input = drop7)
    fc8_1           = layers.Dense(name = 'fc8_1', units = 2622, use_bias = True)(fc8_0)
    prob            = layers.Activation(name = 'prob', activation = 'softmax')(fc8_1)
    model           = Model(inputs = [data], outputs = [prob])
    load_weights(model, weight_file)
    return model

def __flatten(name, input):
    if input.shape.ndims > 2: return layers.Flatten(name = name)(input)
    else: return input

