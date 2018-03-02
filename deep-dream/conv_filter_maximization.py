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
from vis.visualization import get_num_filters
import numpy as np
# Saving images
from scipy.misc import imsave

img_width = 224
img_height = 224

model = KitModelLinear("../training/emoti-wild/emoti-wild-weights.npy")

print('Model loaded.')

model.summary()

output_dir = "filter_max_output_lib/"

layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

for layer_name in layer_dict:
    if not 'conv' in layer_name:
        continue
    if os.path.isfile(output_dir + "stitched_filters_%s.png" % (layer_name)):
        continue
    layer_idx = utils.find_layer_idx(model, layer_name)

    filters = np.arange(get_num_filters(model.layers[layer_idx]))

    kept_filters = []
    for idx in filters:
        print("\n\n================\n" + layer_name + " filter #" + str(idx) + "\n================\n\n")
        img = visualize_activation(model, layer_idx, filter_indices=idx, verbose=True)
        kept_filters.append(img)


    # we will stich the best 64 filters on a 8 x 8 grid.
    n = 1
    while len(kept_filters) > (n*n):
        n += 1

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    # kept_filters.sort(key=lambda x: x[1], reverse=True)
    # kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our n x n filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            if i * n + j < len(kept_filters):
                img = kept_filters[i * n + j]
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                                (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    imsave(output_dir + 'stitched_filters_%s.png' % (layer_name), stitched_filters)
