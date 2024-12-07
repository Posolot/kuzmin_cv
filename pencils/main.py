import cv2
import numpy as np
import os


def calculate_statistics(contours, min_area_threshold=1000, min_ratio=2):
    areas = []
    aspect_ratios = []

    for c in contours:
        if len(c) > 2:
            contour_area = cv2.contourArea(c)
            if contour_area > min_area_threshold:
                rect = cv2.minAreaRect(c)
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height)

                if aspect_ratio >= min_ratio:
                    areas.append(contour_area)
                    aspect_ratios.append(aspect_ratio)

    median_area = np.median(areas) if areas else 0
    median_aspect_ratio = np.median(aspect_ratios) if aspect_ratios else 0

    return median_area, median_aspect_ratio


def search_pencil(contours, min_area, min_ratio):
    count = 0
    for c in contours:
        if len(c) > 2:
            rect = cv2.minAreaRect(c)
            contour_area = cv2.contourArea(c)

            width, height = rect[1]
            aspect_ratio = max(width, height) / min(width, height)

            if contour_area >= min_area and aspect_ratio >= min_ratio:
                count += 1
    return count


colors = {'orange': ((5, 100, 100), (20, 255, 255)),
          'green': ((30, 50, 50), (85, 255, 255)),
          'blue': ((90, 50, 50), (140, 255, 255))}


total_count = 0
all_contours = []

for i in range(1, 13):
    image_path = f'images/img ({i}).jpg'
    print(f"Processing image: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error loading image {image_path}")
        continue

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
        cons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(cons)

min_area, min_aspect_ratio = calculate_statistics(all_contours, 5000, 2)

print(f"Min_area: {min_area}, Min_aspect_ratio: {min_aspect_ratio}")

for i in range(1, 13):
    image_path = f'images/img ({i}).jpg'
    print(f"Image {i} search")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error loading image {image_path}")
        continue

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
        cons, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count_color_pencil = search_pencil(cons, min_area, min_aspect_ratio)
        total_count += count_color_pencil
        print(f"{color} pencils found: {count_color_pencil}")

print(f"Total pencils found: {total_count}")

