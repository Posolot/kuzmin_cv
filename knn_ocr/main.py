import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops
def extractor(region):
    area = np.sum(region.image) / region.image.size
    perimeter = region.perimeter / region.image.size
    wh = np.sum(region.image[:region.image.shape[0] // 2]) / np.sum(region.image[region.image.shape[0] // 2:])
    cy, cx = region.local_centroid
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    euler = region.euler_number
    px, py = region.image.shape[0] // 5, region.image.shape[1] // 5
    kl = 3 * np.sum(region.image[px:-px, py:-py]) / region.image.size
    kls = 2 * np.sum(region.image[int(region.image.shape[0] * 0.45):-int(region.image.shape[0] * 0.45),
                                  int(region.image.shape[1] * 0.45):-int(region.image.shape[1] * 0.45)]) / region.image.size
    pm = region.image.shape[0] / region.image.shape[1]
    eccentricity = region.eccentricity * 8
    have_v1 = (np.sum(np.mean(region.image, 0) > 0.87) > 2) * 3
    have_g1 = (np.sum(np.mean(region.image, 1) > 0.85) > 2) * 4
    have_g2 = (np.sum(np.mean(region.image, 1) > 0.5) > 2) * 5
    hole_size = np.sum(region.image) / region.filled_area
    solidity = region.solidity * 2
    return np.array([area, perimeter, cy, cx, euler, eccentricity, have_v1, hole_size, have_g1, have_g2, kl, pm, kls, solidity, wh])

labels = os.listdir("task/train/")
x, y = [], []

for labelind, label_name in enumerate(labels):
    for filename in os.listdir(f"task/train/{label_name}"):
        template = plt.imread(f"task/train/{label_name}/{filename}")[:, :, :3].mean(2)
        template = (template > 0).astype(int)
        template_labeled = label(template)
        regions = regionprops(template_labeled)
        chosen_region = regions[0] if np.sum(regions[0].image) > 250 else regions[1]
        x.append(extractor(chosen_region))
        y.append(labelind)

x = np.array(x)
y = np.array(y)
knn = cv2.ml.KNearest_create()
knn.train(x.astype("f4"), cv2.ml.ROW_SAMPLE, y.reshape(-1, 1).astype("f4"))
for test in os.listdir("task/"):
    if test != "train":
        template = plt.imread(f"task/{test}")[:, :, :3].mean(2)
        template = (template > 0.1).astype(int)
        template_labeled = label(template)
        regions = regionprops(template_labeled)
        regions = sorted(regions, key=lambda x: x.centroid[1])

        lasta = [0, 0]
        print("\n",test,"Text on picture -- ",end="")
        for i, region in enumerate(regions):
            if np.sum(region.image) > 250:
                new_point = extractor(region)
                ret, _, _, _ = knn.findNearest(np.array(new_point).reshape(1, -1).astype("f4"), 2)
                a = region.bbox
                if i != 0 and a[1] - lasta[-1] > 30:
                    print(" ", end="")
                lasta = a
                print(labels[int(ret)][-1], end="")

