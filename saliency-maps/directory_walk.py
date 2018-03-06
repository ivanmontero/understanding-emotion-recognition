import os

fs = []
for root, dirs, files in os.walk("output/"):
    for f in files:
        # RENAME FILES
        f = f[:f.index(".png")] + f[f.index(".png") + 4:]
        fs.append(root + "/" + f)

# print(fs)

imgs = []
for root, dirs, files in os.walk("datasets/"):
    for f in files:
        imgs.append(f[:f.index(".png")])

img_dict = {}
for img in imgs:
    img_dict[img] = []
    for f in fs:
        if img in f:
            img_dict[img].append(f)

# print(img_dict)

freq_dict = {}
for img in img_dict:
    freq = len(img_dict[img])
    if not freq in freq_dict:
        freq_dict[freq] = {}
    freq_dict[freq][img] = img_dict[img]

del freq_dict[0]
print(freq_dict)