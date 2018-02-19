import keras as K
import sys
sys.path.insert(0, 'model/')
from vgg_face_keras import KitModel

model = KitModel("./model/vgg-face-weights.npy")

model.summary()
