import keras
from keras.models import Model
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K

def ITCModel():
    model = Sequential()
    model.add(Conv2D(64, (7, 7), activation='relu', input_shape=(224, 224, 3,)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(7, activation='softmax', name='predictions'))

    return model

# def ITCModel():
#     model = Sequential()
#     model.add(Conv2D(64, (7, 7), padding='same', activation='relu', input_shape=(224, 224, 3,)))
#     model.add(Conv2D(64, (7, 7), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    
#     model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
#     model.add(Conv2D(128, (5, 5), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

#     model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(256, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))

#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dense(7, activation='softmax', name='predictions'))

#     return model

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

model = ITCModel()
model.summary()

optimizer = Adam(lr=.1)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])

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