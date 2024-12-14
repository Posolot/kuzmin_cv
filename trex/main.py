import time
import cv2
import numpy as np
import pyautogui
import mss

MONITOR = {'top': 280, 'left': 600, 'width': 700, 'height': 200}

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    return dilated

def find_obstacles(processed_image):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            obstacles.append((x, y, w, h))
    return sorted(obstacles, key=lambda obs: obs[0])

def update_timer(start_time, tm):
    elapsed_time = time.time() - start_time
    tm = elapsed_time*6
    return tm

def make_decision(obstacles, tm):
    danger_zone = 140*1.25

    if tm > 200:
        danger_zone = 170*1.25
    if tm > 250:
        danger_zone = 190*1.25
    if tm > 300:
        danger_zone = 210*1.25
    if tm > 350:
        danger_zone = 220*1.25
    if tm > 400:
        danger_zone = 230*1.25
    if tm > 500:
        danger_zone = 275*1.25

    if tm > 800:
        danger_zone = 300
        if tm > 1200:
            danger_zone *= 1 + (tm//1200-tm/1200)

    for i in range(len(obstacles) - 1):
        x1, y1, w1, h1 = obstacles[i]
        x2, y2, w2, h2 = obstacles[i + 1]

        if w1 < 50 < w2:
            if 1000 > tm > 800:
                danger_zone = int(danger_zone * 1.35)
                break
            elif 600 < tm < 800:
                danger_zone = int(danger_zone * 1.3)
                break
            elif 400 < tm < 600:
                danger_zone = int(danger_zone * 1.2)
                break
            elif 200 < tm < 400:
                danger_zone = int(danger_zone * 1.15)
                break
            else:
                danger_zone = int(danger_zone * 1.1)
                break

    for obstacle in obstacles:
        x, y, w, h = obstacle
        mid_x = x + w // 2

        if 50 < mid_x < danger_zone:
            if h > 50:
                pyautogui.press("space")
                break
            elif 10 < h <= 50:
                if y < 120:
                    pyautogui.keyDown("down")
                    time.sleep(0.3)
                    pyautogui.keyUp("down")
                else:
                    pyautogui.press("space")
                break

def main():
    start_time = time.time()
    tm = 0

    with mss.mss() as sct:
        while True:
            screen = np.array(sct.grab(MONITOR))
            processed = process_image(screen)
            obstacles = find_obstacles(processed)
            tm = update_timer(start_time, tm)
            print(f"Текущее значение таймера: {tm}")
            make_decision(obstacles, tm)

            cv2.imshow("Game View", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    print("Начинаем игру через 3 секунды...")
    time.sleep(3)
    main()
