'''
Same code as activation_maximization.py, but adapted to
produces of all classes in the VGG16 classification 
network, and saves them in the output/ directory.
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
# save
from scipy.misc import imsave

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
#label = 833 #20# 850 #309 # 20
for i in range(1000):
    filename = 'output/label_%d.png' % (i)
    if not os.path.isfile(filename):
        print('processing label ' + str(i))
        # Jitter 16 pix along all dim. during optimization
        img = visualize_activation(model, layer_idx, filter_indices=i, 
                max_iter=500, verbose=True, input_modifiers=[Jitter(128)])
        imsave(filename, img)
        print('processed label ' + str(i))

