import keras as K
from scipy import misc
import numpy as np

import sys
# sys.path.insert(0, 'vgg-face/')
# from vgg_face_keras import KitModel
sys.path.insert(0, 'emoti-wild/')
from emoti_wild_keras import KitModel

# Loads the converted model
# The vgg-faces model
# model = KitModel("./vgg-face/vgg-face-weights.npy")
# The emoti-wild model
model = KitModel("./emoti-wild/emoti-wild-weights.npy")

# Prints a summary of the network
model.summary()

# Loads in an image of elon musk, which is prediction #427
img = misc.imread("elon_musk.jpg")

# Image size before resizing
print(img.shape)

# Resize image
img = misc.imresize(img, [224, 224])

# New image size
print(img.shape)

# Run the image throught the model, and make a prediction
pred = model.predict(np.array([img]))
# print(model.predict_classes(img))

# Print out the prediction index, which should be 427
print(np.argmax(pred))
