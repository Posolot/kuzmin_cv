import cv2
import pyautogui
import numpy as np
import time
from mss import mss

mon = {'top': 280, 'left': 600, 'width': 740, 'height': 200}
pyautogui.PAUSE = 0
right, left = 140, 76
in_right = 0.0

sct = mss()
time.sleep(3)

initial_time = time.time()
elapsed_time = 0


def calculate_timer(start_tick, elapsed):
    time_delta = time.time() - start_tick
    elapsed = time_delta * 6
    return elapsed


while True:
    img = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2GRAY)
    img = cv2.resize(img, (500, 120))

    cropped_img = img[88, left:right]
    if 83 in cropped_img:
        pyautogui.keyUp('down')
        pyautogui.keyDown('up')
    else:
        pyautogui.keyUp('up')
        pyautogui.keyDown('down')

    elapsed_time = calculate_timer(initial_time, elapsed_time)
    in_right += elapsed_time * 0.00000001
    right += int(in_right/1)

    cv2.rectangle(img, (left, 92), (right, 102), (255, 0, 0), 2)

    cv2.imshow("Dino BOT", img)
    key = cv2.waitKey(25)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
