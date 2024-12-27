import threading
import cv2
import numpy as np
import pygame
import pymunk
import time


def find_largest_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    if largest_contour is None:
        return None, None

    mask = np.zeros_like(frame)
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    return mask, largest_contour


def process_masked_image(frame, mask):
    masked_image = cv2.bitwise_and(frame, mask)
    gray_masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, binary_masked_image = cv2.threshold(gray_masked_image, 130, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_masked_image, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 800:
            continue
        if cv2.contourArea(contour) < 10000:
            approx = cv2.approxPolyDP(contour, epsilon=0.001, closed=True)
            for i in range(len(approx) - 1):
                x1, y1 = approx[i][0]
                x2, y2 = approx[i + 1][0]
                new_lines.append(((x1, y1), (x2, y2)))

    return new_lines


def process_video(video_source, lines, lock, data_ready_event, ball_position):
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mask, largest_contour = find_largest_contour(frame)
        if mask is None:
            continue

        new_lines = process_masked_image(frame, mask)

        with lock:
            lines.clear()
            lines.extend(new_lines)

        cv2.circle(frame, (int(ball_position[0]), int(ball_position[1])), 10, (0, 0, 255), -1)

        data_ready_event.set()

        cv2.imshow('Original Frame with Ball', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def simulate(lines, video_size, lock, data_ready_event, ball_position):
    pygame.init()
    screen = pygame.display.set_mode(video_size)
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0, 5000)

    ball_body = pymunk.Body(1, pymunk.moment_for_circle(2, 0, 10))
    ball_body.position = video_size[0] // 2, 60
    ball_shape = pymunk.Circle(ball_body, 10)
    ball_shape.elasticity = 0.8
    ball_shape.friction = 0.7
    space.add(ball_body, ball_shape)

    line_segments = []

    while True:
        data_ready_event.wait()
        data_ready_event.clear()

        with lock:
            current_lines = list(lines)
        if not current_lines:
            continue

        for line in line_segments:
            space.remove(line)

        line_segments = []
        for (x1, y1), (x2, y2) in current_lines:
            line = pymunk.Segment(space.static_body, (x1, y1), (x2, y2), 1)
            line.elasticity = 0.8
            line.friction = 1.0
            line_segments.append(line)
            space.add(line)

        space.step(1 / 60.0)

        ball_position[0] = int(ball_body.position.x)
        ball_position[1] = int(ball_body.position.y)

        screen.fill((255, 255, 255))

        for (x1, y1), (x2, y2) in current_lines:
            pygame.draw.line(screen, (0, 0, 0), (x1, y1), (x2, y2), 2)

        pygame.draw.circle(screen, (255, 0, 0), (ball_position[0], ball_position[1]), 10)

        pygame.display.flip()
        clock.tick(60)


video_source = "video.mp4"
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Не удалось открыть видео.")
    exit()

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_size = (video_width, video_height)
cap.release()

lines = []
lock = threading.Lock()
data_ready_event = threading.Event()
ball_position = [video_size[0] // 2, 60]

video_thread = threading.Thread(target=process_video, args=(video_source, lines, lock, data_ready_event, ball_position))
simulation_thread = threading.Thread(target=simulate, args=(lines, video_size, lock, data_ready_event, ball_position))

video_thread.start()
simulation_thread.start()

video_thread.join()
simulation_thread.join()
