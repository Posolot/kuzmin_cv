import cv2

capture = cv2.VideoCapture('output.avi')
count_image = 0

ret, frame = capture.read()
while frame is not None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect = 0
    circle = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            rect += 1
        elif len(approx) > 4:
            circle += 1

    if circle == 5 and rect == 3:
        count_image += 1
        print(f"{count_image}) Найдено изображение)")
    ret, frame = capture.read()

print("Число моих изображений = ", count_image)
