from __future__ import print_function
import os
from keras.applications import VGG16
from keras.applications import InceptionV3
from keras import activations
from keras.models import load_model
from matplotlib import pyplot as plt
from scipy.misc import imsave
import numpy as np
import time
from keras.applications import *
from keras import backend as K
from scipy.misc import imsave
from vis.utils import utils
from vis.visualization import visualize_activation
from vis.input_modifiers import Jitter