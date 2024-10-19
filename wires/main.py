import matplotlib.pyplot as plt
import numpy as np
from scipy.datasets import face
from skimage.draw import disk
from sys import getrecursionlimit, setrecursionlimit
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, binary_opening
import scipy.ndimage.morphology


def neighbours2(y, x):
    return (y, x - 1), (y - 1, x)


def exist(B, nbs):
    nbs1 = []
    for i in nbs:
        if (i[0] >= 0 and i[0] < B.shape[0] and i[1] >= 0 and i[1] < B.shape[1]):
            if B[i] == 0:
                i = None
        else:
            i = None
        nbs1.append(i)
    return nbs1[0], nbs1[1]


def find(label, linked):
    j = label
    while linked[j] != 0:
        j = linked[j]
    return j


def union(label1, label2, linked):
    j = find(label1, linked)
    k = find(label2, linked)
    if j != k:
        linked[k] = j


def two_pass(B):
    LB = np.zeros_like(B)
    linked = np.zeros(B.size // 2 + 1, dtype="uint")
    label = 1
    for y in range(LB.shape[0]):
        for x in range(LB.shape[1]):
            if B[y, x] != 0:
                nbs = neighbours2(y, x)
                existed = exist(B, nbs)
                if existed[0] is None and existed[1] is None:
                    m = label
                    label += 1
                else:
                    lbs = [LB[n] for n in existed if n is not None]
                    m = min(lbs)
                LB[y, x] = m
                for n in existed:
                    if n is not None:
                        lb = LB[n]
                        if lb != m:
                            union(m, lb, linked)
    for y in range(LB.shape[0]):
        for x in range(LB.shape[1]):
            if B[y, x] != 0:
                new_label = find(LB[y, x], linked)
                if new_label != LB[y, x]:
                    LB[y, x] = new_label

    unique_labels = np.unique(LB)
    unique_labels = unique_labels[unique_labels != 0]
    for i, label in enumerate(unique_labels):
        LB[LB == label] = i + 1
    return LB


struct = np.ones((3, 2))
for i in [1, 2, 3, 4, 5, 6]:
    print("\n")
    image = np.load(f"wires{i}npy.txt").astype(int)
    t_image = two_pass(image)
    izm_image = binary_opening(t_image, struct).astype(int)
    t_izm_image = two_pass(izm_image)
    unique_labels = np.unique(t_image)
    print(f"на рисунке {i} \nкол-проводов:{len(unique_labels)-1}")
    for pr in range(1, t_image.max() + 1):
        pr1 = (t_image == pr)
        new_labels = t_izm_image[pr1]
        unique_new_labels = np.unique(new_labels)
        unique_new_labels = unique_new_labels[unique_new_labels != 0]
        num_parts = len(unique_new_labels)
        if num_parts == 0:
            print(f"{pr} провод полностью разорван")
        elif num_parts == 1:
            print(f"{pr} провод не порван")
        else:
            print(f"{pr} провод порван на {num_parts} части")

    plt.subplot(121)
    plt.imshow(t_image)
    plt.subplot(122)
    plt.imshow(t_izm_image)
    plt.show()
