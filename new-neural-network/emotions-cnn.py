### Retrain last layer of emotions in the wild
###
# Dataset
import os
from os.path import isdir, isfile, join
from vis.utils import utils

# Model setup
import sys
sys.path.insert(0, '../training/emoti-wild/')
from emoti_wild_keras import KitModel
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

###################### GETTING DATASET ######################
DATASET_DIR = "../datasets/ferg_db_12_emotions/"
ALREADY_TRAINED = "./already_trained.dat"
class_dirs = [f for f in os.listdir(DATASET_DIR) if isdir(join(DATASET_DIR, f))]
for i in range(len(class_dirs)):
    print(str(i) + ": " + class_dirs[i])

already_trained = []
if os.path.isfile(ALREADY_TRAINED):
    already_trained = open(ALREADY_TRAINED).readlines()

train_data = {}
for class_dir in class_dirs:
    file_names = [f for f in os.listdir(DATASET_DIR + class_dir) if isfile(join(DATASET_DIR + class_dir, f))]
    train_data[class_dir] = file_names

# print(train_data)
###################### MODEL CREATION ######################
# create the base pre-trained model
base_model = KitModel("../training/emoti-wild/emoti-wild-weights.npy")

# add a global spcial average pooling layer
x = base_model.output
# x = GlobalAveragePooling2D()(x)
# a fully connected layer for the almost last layer
x = Dense(1024, activation="relu")(x)
# add a logistic layer for predictions (only 7 emotions)
predictions = Dense(7, activation="softmax")(x)

# The model to train
model = Model(inputs=base_model.input, outputs=predictions)

# Train only the last layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

###################### MODEL TRAINING ######################
def generate_training_batch(size):
    training_batch = []
    pred = []
    names = []
    for class_dir in class_dirs:
        for file_name in train_data[class_dir]:
            if len(training_batch) == size:
                break
            # print(DATASET_DIR + class_dir + "/" + file_name)
            if not file_name in already_trained:
                training_batch.append(utils.load_img(DATASET_DIR + class_dir + "/" + file_name, target_size=(224, 224)))
                pred.append(class_dirs.index(class_dir))
                names.append(file_name)
    # print(training_batch)
    return names, training_batch, pred

EPOCHS = 250
names, batch, pred = generate_training_batch(100)
while len(batch) != 0:
    print("training: " + ''.join(names))
    model.fit(batch, pred, 10, epochs=EPOCHS, verbose=2)
    trained_file = open(ALREADY_TRAINED, "a+")
    for name in names:
        trained_file.write(name)
    trained_file.close()
    names, batch, pred = generate_training_batch(100)


