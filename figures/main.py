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


struct2 = np.ones((4, 6))
struct3 = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
]
struct4 = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
]
struct6 = [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 1]
]
struct5 = [
    [1, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1]
]

image = np.load(f"ps.npy.txt.").astype(int)
struct_one = two_pass(binary_erosion(image, struct2).astype(int))
struct_two = two_pass(binary_erosion(image, struct3).astype(int))
struct_three = two_pass(binary_erosion(image, struct4).astype(int))
struct_four = two_pass(binary_erosion(image, struct5).astype(int))
struct_five = two_pass(binary_erosion(image, struct6).astype(int))

print(f"Кол-во полных горизонтальных объектов ", struct_one.max())
print(f"Кол-во вертикальных объектов(отсутстствует квадрат справа в середине) ", struct_two.max())
print(f"Кол-во вертикальных объектов (отсутстствует квадрат слева в середине)", struct_three.max())
print(f"Кол-во горизонтальных объектов(отсутстствует квадрат снизу в середине) ", struct_four.max())
print(f"Кол-во горизонтальных объектов (отсутстствует квадрат сверху в середине)", struct_five.max())
print(f"Кол-во всех объектов ", struct_two.max()+struct_three.max()+struct_four.max()+struct_five.max()+struct_one.max())

plt.imshow(image)
plt.show()
