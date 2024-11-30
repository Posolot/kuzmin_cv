
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.spatial.distance import cdist


def track_objects(centroids_list):
    trajectories = {}
    for img_num, centroid in enumerate(centroids_list):
        if img_num == 0:
            for fig_num, (centroid, label) in enumerate(centroid):
                trajectories[fig_num] = [centroid]
        else:
            last_center = []
            for center in trajectories.values():
                last_center.append(center[-1])
            last_center = np.array(last_center)
            curr_center = []
            for c in centroid:
                curr_center.append(c[0])
            curr_centroids = np.array(curr_center)
            distances = cdist(last_center, curr_center)
            for fig_num, path in trajectories.items():
                min_distance = np.argmin(distances[fig_num])
                path.append(curr_centroids[min_distance])
    return trajectories


centroids_list = []
for i in range(100):
    data = np.load(f'out/h_{i}.npy')
    labeled = label(data)
    regions = regionprops(labeled)
    current_centroid = []
    for region in regions:
        current_centroid.append([region.centroid, region.label])
    centroids_list.append(current_centroid)


trajectories = track_objects(centroids_list)
for fig_num, trajectory in trajectories.items():
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 1], trajectory[:, 0], marker='o')
plt.show()

