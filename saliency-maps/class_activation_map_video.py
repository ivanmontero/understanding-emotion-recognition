# The directory containing all the frames of the video
DATA_DIRECTORY = "video_input/"

# The target emotion to track
TARGET_EMOTION = "joy"

# The directory to output all the frames of the video, with class
# activation maps
OUTPUT_DIRECTORY = "video_output/"

#from keras.applications import VGG16
import sys
sys.path.insert(0, '../training/emoti-wild/')
from emoti_wild_keras import KitModelLinear
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
model = KitModelLinear("../training/emoti-wild/emoti-wild-weights.npy")
model.summary()

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'prob')

# Loading all images
input_file_names = [f for f in os.listdir(DATA_DIRECTORY)]

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

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
for img_name in input_file_names:
    img = utils.load_img(DATA_DIRECTORY + img_name, target_size=(224, 224))
    grads = visualize_cam(model, layer_idx, filter_indices=EMOTIONS[TARGET_EMOTION],
        seed_input=img, backprop_modifier=None)
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    
    blended = np.array(img) * .4 + np.array(jet_heatmap) * .6

    imsave(OUTPUT_DIRECTORY + img_name[:-4] + "_cam.png", blended)

