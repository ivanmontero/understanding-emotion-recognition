import keras
from keras.models import Model
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K

# classes = 7

# def ITCModel():
#     model = Sequential()
#     model.add(Conv2D(96, (7, 7), activation='relu', input_shape=(224, 224, 3,)))
#     model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
#     model.add(Conv2D(256, (5, 5), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(Conv2D(512, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
#     model.add(Flatten())
#     model.add(Dense(4048, activation='relu'))
#     model.add(Dense(classes, activation ='softmax', name='predictions'))
#     return model

# def ITCModel1():
#     data            = layers.Input(name = 'data', shape = (224, 224, 3,) )
#     conv1           = layers.Conv2D(name='conv1', filters = 96, kernel_size=((7, 7)), strides=((2, 2)), padding='valid', use_bias=True)(data)
#     relu1           = layers.Activation(name = 'relu1', activation = 'relu')(conv1)
#     pool1_input     = layers.ZeroPadding2D(padding = ((0, 2), (0, 2)))(relu1)
#     pool1           = layers.MaxPooling2D(name = 'pool1', pool_size = (3, 3), strides = (3, 3), padding = 'valid')(pool1_input)
#     conv2_input     = layers.ZeroPadding2D(padding = ((2, 2), (2, 2)))(pool1)
#     conv2           = layers.Conv2D(name='conv2', filters = 256, kernel_size=((5, 5)), strides=((1, 1)), padding='valid', use_bias=True)(conv2_input)
#     relu2           = layers.Activation(name = 'relu2', activation = 'relu')(conv2)
#     pool2_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu2)
#     pool2           = layers.MaxPooling2D(name = 'pool2', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool2_input)
#     conv3_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool2)
#     conv3           = layers.Conv2D(name='conv3', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv3_input)
#     relu3           = layers.Activation(name = 'relu3', activation = 'relu')(conv3)
#     conv4_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu3)
#     conv4           = layers.Conv2D(name='conv4', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv4_input)
#     relu4           = layers.Activation(name = 'relu4', activation = 'relu')(conv4)
#     conv5_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(relu4)
#     conv5           = layers.Conv2D(name='conv5', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv5_input)
#     relu5           = layers.Activation(name = 'relu5', activation = 'relu')(conv5)
#     pool5_input     = layers.ZeroPadding2D(padding = ((0, 2), (0, 2)))(relu5)
#     pool5           = layers.MaxPooling2D(name = 'pool5', pool_size = (3, 3), strides = (3, 3), padding = 'valid')(pool5_input)
#     fc6_0           = layers.Dense(name = 'fc6_0', units = 4048, use_bias = True)(pool5)
#     relu6           = layers.Activation(name = 'relu6', activation = 'relu')(fc6_0)
#     fc6_cat_0       = __flatten(name = 'fc6_cat_0', input = relu6)
#     fc6_cat_1       = layers.Dense(name = 'fc6_cat_1', units = 7, use_bias = True)(fc6_cat_0)
#     prob            = layers.Activation(name = 'predictions', activation = 'softmax')(fc6_cat_1)
#     model           = Model(inputs = [data], outputs = [prob])
#     return model
# def __flatten(name, input):
#     if input.shape.ndims > 2: return layers.Flatten(name = name)(input)
#     else: return input

def ITCModel():
    data            = layers.Input(name = 'data', shape = (224, 224, 3,) )
    conv1           = layers.Conv2D(name='conv1', filters = 96, kernel_size=((7, 7)), strides=((2, 2)), padding='valid', use_bias=True)(data)
    relu1           = layers.Activation(name = 'relu1', activation = 'relu')(conv1)
    pool1_input     = layers.ZeroPadding2D(padding = ((0, 2), (0, 2)))(relu1)
    pool1           = layers.MaxPooling2D(name = 'pool1', pool_size = (3, 3), strides = (3, 3), padding = 'valid')(pool1_input)
    conv2_input     = layers.ZeroPadding2D(padding = ((2, 2), (2, 2)))(pool1)
    conv2           = layers.Conv2D(name='conv2', filters = 256, kernel_size=((5, 5)), strides=((1, 1)), padding='valid', use_bias=True)(conv2_input)
    relu2           = layers.Activation(name = 'relu2', activation = 'relu')(conv2)
    pool2_input     = layers.ZeroPadding2D(padding = ((0, 1), (0, 1)))(relu2)
    pool2           = layers.MaxPooling2D(name = 'pool2', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(pool2_input)
    conv3_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(pool2)
    conv3           = layers.Conv2D(name='conv3', filters = 512, kernel_size=((3, 3)), strides=((1, 1)), padding='valid', use_bias=True)(conv3_input)
    relu3           = layers.Activation(name = 'relu3', activation = 'relu')(conv3)
    pool4_input     = layers.ZeroPadding2D(padding = ((0, 2), (0, 2)))(relu3)
    pool4           = layers.MaxPooling2D(name = 'pool4', pool_size = (3, 3), strides = (3, 3), padding = 'valid')(pool4_input)
    fc5_0           = layers.Dense(name = 'fc5_0', units = 4048, use_bias = True)(pool4)
    relu5           = layers.Activation(name = 'relu5', activation = 'relu')(fc5_0)
    fc5_cat_0       = layers.Flatten(name = 'fc5_cat_0')(relu5)
    fc5_cat_1       = layers.Dense(name = 'fc5_cat_1', units = 7, use_bias = True)(fc5_cat_0)
    prob            = layers.Activation(name = 'predictions', activation = 'softmax')(fc5_cat_1)
    model           = Model(inputs = [data], outputs = [prob])
    return model



model = ITCModel()
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(shear_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

DATASET_DIR = "../datasets/ferg_db_12_emotions/"

training_set = train_datagen.flow_from_directory(DATASET_DIR,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(DATASET_DIR,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)