import zmq
import cv2
import numpy as np
from skimage.measure import label,regionprops

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
port = 5555
socket.connect("tcp://192.168.0.100:%s" % port)
cv2.namedWindow("Client recv", cv2.WINDOW_GUI_NORMAL)
count = 0


def count_objects(img):
    count_circle = 0
    count_square = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]
    ret, tresh1 = cv2.threshold(saturation, 100, 255, cv2.THRESH_BINARY)
    ret, tresh2 = cv2.threshold(brightness, 225, 255, cv2.THRESH_BINARY)
    combined_thresh = cv2.bitwise_or(tresh1, tresh2)
    kernel = np.ones((3, 3), np.uint8)
    erosed = cv2.erode(combined_thresh, kernel, iterations=2)
    cv2.imshow("binary", erosed)

    labeled = label(combined_thresh)
    regions = regionprops(labeled)
    for region in regions:
        circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
        if circularity > 0.77 and region.area > 50:
            count_circle += 1
        elif region.area > 50:
            count_square += 1
    return count_circle,count_square


while True:
    msg = socket.recv()
    frame = cv2.imdecode(np.frombuffer(msg, np.uint8), -1)
    count = count_objects(frame)
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("screenshot.png", frame)
    cv2.putText(frame, f"Count - {count[0]+count[1]},circle - {count[0]}, square - {count[1]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0),3)
    count_objects(frame)
    cv2.imshow("Client recv", frame)
