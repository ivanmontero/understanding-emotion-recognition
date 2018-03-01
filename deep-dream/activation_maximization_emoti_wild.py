'''
Visualizing the classes of the Emotions in The Wild model
'''
# Model setup
import sys
sys.path.insert(0, '../training/emoti-wild/')
from emoti_wild_keras import KitModelLinear
# Keras and visualization
from vis.utils import utils
from keras import activations
# apply modifications
import os
from keras.models import load_model
# Visualization
from vis.visualization import visualize_activation
from matplotlib import pyplot as plt
from vis.input_modifiers import Jitter

# Image path
img_path = "./input/Female1_Stage2.png"

# Build the network
model = KitModelLinear("../training/emoti-wild/emoti-wild-weights.npy")
model.summary()

layer_idx = utils.find_layer_idx(model, 'prob')

#### Activation will be switched in model definition ####
## * KitModel is to be used for predictions
## * KitModelLinear is to be used for visualizations

# Load image 
img = utils.load_img(img_path, target_size=(224, 224))

#plt.imshow(img)
#plt.show()

plt.rcParams['figure.figsize'] = (6, 6)

# Choose label
label = 3

# Jitter 16 pix along all dim. during optimization
img = visualize_activation(model, layer_idx, filter_indices=label, 
        max_iter=20, verbose=True, seed_input=img)
plt.imshow(img)
plt.show()
