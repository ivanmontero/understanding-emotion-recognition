from keras.applications import VGG16
import os
from vis.utils import utils
from vis.visualization import visualize_saliency, visualize_cam, overlay
from keras import activations
from matplotlib import pyplot as plt
from keras.models import load_model
import matplotlib.cm as cm
import numpy as np

# Build the VGG
model = VGG16(input_shape=(224, 224, 3),weights='imagenet', include_top=True)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear

# Apply modifications
model.save('temp.h5')
model = load_model('temp.h5')
os.remove('temp.h5')

# Load image
img = utils.load_img('zebra.jpg', target_size=(224, 224))

plt.rcParams['figure.figsize'] = (6, 6)

# Zebra
index = 340

modifier = [None, 'guided', 'relu']

grads = visualize_saliency(model, layer_idx, filter_indices=index,
        seed_input=img, backprop_modifier=modifier[0])
#grads = visualize_cam(model, layer_idx, filter_indices=index,
#        seed_input=img, backprop_modifier=modifier[0])     

#jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
#plt.imshow(overlay(jet_heatmap, img))

plt.imshow(grads, cmap='jet')
plt.show()
