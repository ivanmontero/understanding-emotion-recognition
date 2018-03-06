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
train_data_total = {}
for class_dir in class_dirs:
    file_names = [f for f in os.listdir("datasets/" + class_dir)]
    train_data_total[class_dir] = file_names

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

OUTPUT_DIR = "output_per_class/"

# Save the images
def save(grads, class_name, img_name, vis_type, emotion):
    jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    directory = OUTPUT_DIR + "%s_%s/" % (class_name, vis_type)
    if not os.path.exists(directory):
        os.makedirs(directory)
    imsave(directory + "%s_%s_%s.png" % (img_name, vis_type, emotion), 
           jet_heatmap)

def exists(class_name, img_name, vis_type, emotion):
    return os.path.exists(OUTPUT_DIR + "%s_%s/%s_%s_%s.png" 
                % (class_name, vis_type, 
                    img_name, vis_type, emotion))

# Run in batches
batch_size = 10

batch_index = 0
def generate_batch():
    global batch_index, batch_size, train_data_total
    train_data = {}
    for class_name in train_data_total:
        imgs = train_data_total[class_name]
        if batch_index * batch_size >= len(imgs):
            continue
        if (batch_index + 1) * batch_size > len(imgs):
            train_data[class_name] = imgs[batch_index * batch_size :]
        else:
            train_data[class_name] = imgs[batch_index * batch_size : (batch_index + 1) * batch_size]
    batch_index += 1
    return train_data

train_data = generate_batch()
while len(train_data) > 0:
    # Produce output
    for class_name in train_data:
        for img_name in train_data[class_name]:
            img = utils.load_img("datasets/" + class_name + "/" + img_name, target_size=(224, 224))
            for emotion in EMOTIONS:
                if not exists(class_name, img_name, "saliency", emotion):
                    print("%s %s created from %s" % (img_name, "saliency", emotion))
                    grads = visualize_saliency(model, layer_idx, filter_indices=EMOTIONS[emotion],
                        seed_input=img, backprop_modifier="guided")
                    save(grads, class_name, img_name, "saliency", emotion)

                if not exists(class_name, img_name, "heatmap", emotion):
                    print("%s %s created from %s" % (img_name, "heatmap", emotion))
                    grads = visualize_cam(model, layer_idx, filter_indices=EMOTIONS[emotion],
                        seed_input=img, backprop_modifier=None)
                    save(grads, class_name, img_name, "heatmap", emotion)
    train_data = generate_batch()