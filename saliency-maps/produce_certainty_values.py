#from keras.applications import VGG16
import sys
from emoti_wild_keras import KitModelLinear, KitModel
import os
from vis.utils import utils
sys.path.insert(0, '../')
from keras_vis_fixed.visualization import visualize_saliency, visualize_cam, overlay
from keras import activations
from matplotlib import pyplot as plt
from keras.models import load_model
import matplotlib.cm as cm
import numpy as np
from scipy.misc import imsave

# Load the model
model = KitModel("emoti-wild-weights.npy")
model.summary()

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'prob')

# Loading all test data
class_dirs = [f for f in os.listdir("datasets/")]
train_data = {}
for class_dir in class_dirs:
    file_names = [f for f in os.listdir("datasets/" + class_dir)]
    train_data[class_dir] = file_names

# Emotion set
EMOTIONS = {
    "angry" : 0,
    "disgust" : 1,
    "fear" : 2,
    "joy" : 3,
    "neutral" : 4,
    "sad" : 5,
    "surprise" : 6,
}


# Produce certainties
for class_name in train_data:
    class_idx = EMOTIONS[class_name]
    input_data = []
    for img_name in train_data[class_name]:
        input_data.append(utils.load_img("datasets/" + class_name + "/" + img_name, target_size=(224, 224)))
    result = model.predict(np.array(input_data), batch_size=32)
    i = 0
    print(result)
    lines_of_text = []
    for img_name in train_data[class_name]:
        line = img_name + "\n"
        for e in EMOTIONS:
            tabbing = "\t"
            if not e == "surprise":
                tabbing += "\t"
            line += "\t" + e + tabbing + ("%.10f" % result[i][EMOTIONS[e]]) + "\n"
        lines_of_text.append(line + "\n")
        i += 1
    f = open("certainties/" + class_name + ".txt", "w")
    f.writelines(lines_of_text)
    f.close()
