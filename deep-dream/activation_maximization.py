# Model setup
from keras.applications import VGG16
from vis.utils import utils
from keras import activations
# apply modifications
import os
from keras.models import load_model
# Visualization
from vis.visualization import visualize_activation
from matplotlib import pyplot as plt

# Build the network
model = VGG16(weights='imagenet', include_top=True)

# Ability to search for layer index by name
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear

# Replacing the next line:
# model = utils.apply_modifications(model)
model.save('temp.h5')
model = load_model('temp.h5')
os.remove('temp.h5')

plt.rcParams['figure.figsize'] = (18, 6)

# Choose label
img = visualize_activation(model, layer_idx, filter_indices=850)
plt.imshow(img)
plt.show()
