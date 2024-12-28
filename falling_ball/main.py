import cv2
import numpy as np
import pygame
import pymunk

def find_largest_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
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
        if cv2.contourArea(contour) < 2000:
            continue
        if cv2.contourArea(contour) < 8000:
            approx = cv2.approxPolyDP(contour, epsilon=0.001, closed=True)
            for i in range(len(approx) - 1):
                x1, y1 = approx[i][0]
                x2, y2 = approx[i + 1][0]
                new_lines.append(((x1, y1), (x2, y2)))

    return new_lines

def main(video_url, resize_factor=0.5):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("Не удалось открыть видео по URL.")
        exit()

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_size = (video_width, video_height)
    resized_video_size = (int(video_width * resize_factor), int(video_height * resize_factor))

    pygame.init()
    screen = pygame.display.set_mode(resized_video_size)
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0, 5000)

    line_segments = []
    balls = []

    running = True
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, resized_video_size)

        mask, largest_contour = find_largest_contour(frame_resized)
        if mask is not None:
            new_lines = process_masked_image(frame_resized, mask)
            for line in line_segments:
                space.remove(line)
            line_segments = []
            for (x1, y1), (x2, y2) in new_lines:
                line = pymunk.Segment(space.static_body, (x1, y1), (x2, y2), 5)
                line.elasticity = 0.8  
                line.friction = 1.5    
                line_segments.append(line)
                space.add(line)

            space.step(1 / 60.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                ball_body = pymunk.Body(10, pymunk.moment_for_circle(10, 0, 10)) 
                ball_body.position = x, y
                ball_shape = pymunk.Circle(ball_body, 10)
                ball_shape.elasticity = 0.9
                ball_shape.friction = 1.0 
                space.add(ball_body, ball_shape)
                balls.append((ball_body, ball_shape))
        screen.fill((0, 0, 0))
        for (x1, y1), (x2, y2) in new_lines:
            pygame.draw.line(screen, (255, 255, 255), (x1, y1), (x2, y2), 2)

        for ball_body, ball_shape in balls:
            ball_position = [int(ball_body.position.x), int(ball_body.position.y)]
            pygame.draw.circle(screen, (255, 0, 0), ball_position, 10)

        for ball_body, ball_shape in balls:
            ball_position = [int(ball_body.position.x), int(ball_body.position.y)]
            cv2.circle(frame_resized, (ball_position[0], ball_position[1]), 10, (0, 0, 255), -1)

        cv2.imshow("Resized Video with Balls", frame_resized)

        pygame.display.flip()
        clock.tick(60)

        space.step(1 / 60.0)

    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()

video_url = "http://192.168.43.1:8080/video"
resize_factor = 0.5

main(video_url, resize_factor)
