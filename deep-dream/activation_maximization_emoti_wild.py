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

# Build the network
model = KitModelLinear("../training/emoti-wild/emoti-wild-weights.npy")
model.summary()

layer_idx = utils.find_layer_idx(model, 'prob')

#### Activation will be switched in model definition ####
## * KitModel is to be used for predictions
## * KitModelLinear is to be used for visualizations
# Ability to search for layer index by name

# Swap softmax with linear
# model.layers[layer_idx].activation = activations.linear

# Replacing the next line:
# model = utils.apply_modifications(model)
# model.save('temp.h5')
# model = load_model('temp.h5')
# os.remove('temp.h5')

plt.rcParams['figure.figsize'] = (6, 6)

# Choose label
label = 2

# Jitter 16 pix along all dim. during optimization
img = visualize_activation(model, layer_idx, filter_indices=label, 
        max_iter=10000, verbose=True, input_modifiers=[Jitter(16)])
plt.imshow(img)
plt.show()
