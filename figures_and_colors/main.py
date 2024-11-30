import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
import numpy as np


def colors_counter(figure_colors):
    for fig_color in set(figure_colors):
        cnt = figure_colors.count(fig_color)
        print(f"{fig_color}: {cnt}")


def color_to_colorname(hue):
    if hue < 0.05 or hue >= 0.95:
        return "Red"
    elif 0.05 <= hue < 0.15:
        return "Orange"
    elif 0.15 <= hue < 0.35:
        return "Yellow"
    elif 0.35 <= hue < 0.45:
        return "Green"
    elif 0.45 <= hue < 0.65:
        return "Cyan"
    elif 0.65 <= hue < 0.75:
        return "Blue"
    elif 0.75 <= hue < 0.85:
        return "Purple"
    elif 0.85 <= hue < 0.95:
        return "Pink"
    else:
        return "Unknown"


im = plt.imread("balls_and_rects.png")
im_hsv = rgb2hsv(im)

binary = im.mean(2)
binary[binary > 0] = 1
labeled = label(binary)
regions = regionprops(labeled)

circle_colors = []
rect_colors = []
figures = {
    "circle": 0,
    "rects": 0
}

for region in regions:
    recognized = 'circle' if region.eccentricity < 0.5 else 'rects'
    figures[recognized] += 1

    cy, cx = region.centroid
    color = im_hsv[int(cy), int(cx)][0]
    if recognized == 'circle':
        circle_colors.append(color_to_colorname(color))
    else:
        rect_colors.append(color_to_colorname(color))

print(f"Count figures -- {np.max(labeled)}")
print(f"Circle -- {figures['circle']}\nRectangles -- {figures['rects']}\n")
print("BALLS COLORS:")
colors_counter(circle_colors)
print("\nRECTANGLE COLORS:")
colors_counter(rect_colors)
