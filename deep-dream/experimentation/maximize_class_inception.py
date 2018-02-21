'''
WIP: Visualizing a single class. First attempt at visualizing a class.
Observations:
 - Output is fairly noisy
 - Features seem to be condensed in the edges of the image
 - Method works with InceptionV3, but needs tweaking
'''

from keras import applications
from keras import backend as K
import numpy as np
from scipy.misc import imsave

img_width = 512
img_height = 512

# Build  inception model
model = applications.inception_v3.InceptionV3(include_top=True, weights='imagenet')

model.summary()

# placeholder for input images
input_img = model.input

# get the symbolic outputs of each "key" layer
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# the output to maximize
output_index = 1

# maximize the output index
loss = K.mean(model.output[:, output_index])

# compute the gradient of the input picture
grads = K.gradients(loss, input_img)[0]

# Normalization: normalize gradients
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# returns the loss and grads of the gien input
iterate = K.function([input_img], [loss, grads])

# gray w/ some noise
# input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128
input_img_data = np.random.random((1, img_width, img_height, 3)) * 20 + 128

# step size
step = 1

# gradient ascent
for i in range(200):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

def deprocess_image(x):
    # normalize tensor: centor on 0
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

img = input_img_data[0]
img = deprocess_image(img)
imsave('output_%d_img.png' % (output_index), img)
