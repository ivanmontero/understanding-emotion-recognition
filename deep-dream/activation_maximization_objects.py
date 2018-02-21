'''
Latest attempt at visualizing a class. Uses keras-vis library
Observations:
 - Output better resembles the specified class
 - The "Jitter" feature makes sure features aren't 
   confined to the edges of the image.
 - Try tweaking so it can work with InceptionV3
 - Explore source code of keras-vis for more fine-tuned control
'''
# Model setup
from keras.applications import VGG16
from keras.applications import InceptionV3
from vis.utils import utils
from keras import activations
# apply modifications
import os
from keras.models import load_model
# Visualization
from vis.visualization import visualize_activation
from matplotlib import pyplot as plt
from vis.input_modifiers import Jitter

# Build the network
model = VGG16(weights='imagenet', include_top=True)
#model = InceptionV3(weights='imagenet', include_top=True)

model.summary()

# Ability to search for layer index by name
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear

# Replacing the next line:
# model = utils.apply_modifications(model)
model.save('temp.h5')
model = load_model('temp.h5')
os.remove('temp.h5')

plt.rcParams['figure.figsize'] = (6, 6)

# Choose label
label = 850 #20# 850 #309 # 20

# Jitter 16 pix along all dim. during optimization
img = visualize_activation(model, layer_idx, filter_indices=label, 
        max_iter=500, verbose=True, input_modifiers=[Jitter(128)])
plt.imshow(img)
plt.show()
