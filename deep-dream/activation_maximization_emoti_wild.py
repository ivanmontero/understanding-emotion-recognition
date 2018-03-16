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
## Since the library will try recomipiling the model by 
## saving the file in a "tempfile" location, which causes
## an exception, KitModelLinear manually sets the activation
## of the last layer from softmax to linear for better 
## results.

# Load image 
img = utils.load_img(img_path, target_size=(224, 224))

#plt.imshow(img)
#plt.show()

plt.rcParams['figure.figsize'] = (6, 6)

EMOTIONS = {
    "angry" : 0,
    "disgust" : 1,
    "fear" : 2,
    "joy" : 3,
    "neutral" : 4,
    "sad" : 5,
    "surprise" : 6,
}

# Choose label
label = 0

# Jitter 16 pix along all dim. during optimization
img = visualize_activation(model, layer_idx, filter_indices=label, 
        max_iter=1000, verbose=True, seed_input=img)
plt.imshow(img)
plt.show()
