#from keras.applications import VGG16
import sys
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
model = KitModelLinear("emoti-wild-weights.npy")
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

# Helpful toString to deal with None
toStr = lambda s: s or "none"

# Save the images
def save(grads, img_name, class_name, vis_type, modifier):
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    directory = "output/%s_%s_%s/" % (class_name, vis_type, toStr(modifier))
    if not os.path.exists(directory):
        os.makedirs(directory)
    imsave(directory + "%s_%s_%s.png" % (img_name, vis_type, toStr(modifier)), 
           jet_heatmap)

def exists(img_name, class_name, vis_type, modifier):
    return os.path.exists("output/%s_%s_%s/%s_%s_%s.png" 
                % (class_name, vis_type, toStr(modifier), 
                    img_name, vis_type, toStr(modifier)))


# Modifiers
MODIFIERS = [None, 'guided', 'relu']

# Produce output
for class_name in train_data:
    class_idx = EMOTIONS[class_name]
    for img_name in train_data[class_name]:
        img = utils.load_img("datasets/" + class_name + "/" + img_name, target_size=(224, 224))
        for modifier in MODIFIERS:
            # Visualize saliency
            if not exists(img_name, class_name, "saliency", modifier):
                print("%s %s %s %s" % (class_name, img_name, "saliency", toStr(modifier)))
                grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
                    seed_input=img, backprop_modifier=modifier)
                save(grads, img_name, class_name, "saliency", modifier)
 
            if not exists(img_name, class_name, "heatmap", modifier):
                print("%s %s %s %s" % (class_name, img_name, "heatmap", toStr(modifier)))
                grads = visualize_cam(model, layer_idx, filter_indices=class_idx,
                    seed_input=img, backprop_modifier=modifier)
                save(grads, img_name, class_name, "heatmap", modifier)
