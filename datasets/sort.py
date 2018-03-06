# For sorting
import os
from os.path import isfile

DATA_DIR = "train_data/"

files = [f for f in os.listdir(DATA_DIR) if isfile(DATA_DIR + f)]

classes = {}

for f in files:
    underscore = [pos for pos, char in enumerate(f) if char == '_']
    
    cls = f[underscore[0] + 1 : underscore[1]]
    if not cls in classes:
        classes[cls] = []

    classes[cls].append(f)

for c in classes:
    print(c)
    for f in classes[c]:
        os.rename(DATA_DIR + f, c + "/" + f)
