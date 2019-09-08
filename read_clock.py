from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

d_img_count = 0
warped_w = 1000
warped_h = 1200


def d_imwrite(name, img):
    global d_img_count
    cv2.imwrite(f"tmp/{d_img_count}_{name}", img)
    d_img_count += 1


def find_circle(img):
    circles = cv2.HoughCircles(img,
                               cv2.HOUGH_GRADIENT,
                               dp=1,
                               minDist=300,
                               param1=400,
                               param2=100,
                               minRadius=0,
                               maxRadius=0)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img_color, (i[0], i[1]), 2, (0, 0, 255), 3)
    d_imwrite("circles.jpg", img_color)

    circle = circles[0, 0]
    return circle


def warp_polar(img, circle):
    warped = cv2.warpPolar(img, dsize=(1000, warped_h), center=(circle[0], circle[1]), maxRadius=circle[2],
                           flags=cv2.WARP_POLAR_LINEAR)
    d_imwrite("warp_polar.jpg", warped)
    return warped


def dilate(img):
    kernel = np.ones((1, 200), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    d_imwrite("dilation.jpg", dilation)
    return dilation


def threshold(img):
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    d_imwrite("threshold.jpg", th)
    return th


def crop_hand_area(img):
    cropped = img[:, warped_w // 5:]
    d_imwrite("cropped.jpg", cropped)
    return cropped


def get_3_hands_by_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangle = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    min_h = warped_h
    min_h_idx = 0
    min_w = warped_w
    min_w_idx = 0
    values = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rectangle = cv2.rectangle(rectangle, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if x != 0:  # not on center
            continue

        value = y + h // 2  # center of hand
        value += warped_h // 4  # default value starts at 3 o'clock
        if value > warped_h:
            value -= warped_h
        values.append(value)

        idx = len(values) - 1

        if h < min_h:
            min_h = h
            min_h_idx = idx

        if w < min_w:
            min_w = w
            min_w_idx = idx

    d_imwrite("rectangle.jpg", rectangle)

    second_hand_idx = min_h_idx  # thinnest -> second
    hour_hand_idx = min_w_idx  # shortest -> hour
    minute_hand_idx = 3 - min_w_idx - min_h_idx  # another one -> minute

    hour_value = values[hour_hand_idx]
    minute_value = values[minute_hand_idx]
    second_value = values[second_hand_idx]

    hour = hour_value // 100
    minute = minute_value // 20
    second = second_value // 20
    return hour, minute, second


def imwrite(file_name, title, img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.title(title)
    plt.imshow(img)
    # plt.figure(figsize=(500, 500))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.9, hspace=0)
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    bottom=False,
                    left=False)
    plt.savefig("result/"+file_name.replace("jpg", "png"))


def read_clock():
    input_dir = Path("img")
    file_name = "img4.jpg"

    file_path = input_dir / file_name

    img = cv2.imread(str(file_path))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d_imwrite("gray.jpg", gray_img)

    circle = find_circle(gray_img)
    warped = warp_polar(gray_img, circle)
    dilated = dilate(warped)
    th = threshold(dilated)
    cropped = crop_hand_area(th)

    hour, minute, second = get_3_hands_by_contours(cropped)
    print(f"hour:{hour}, minute:{minute}, second:{second}")
    title = f"{hour:02}:{minute:02}:{second:02}"
    imwrite(file_name, title, img)


def main():
    read_clock()


if __name__ == "__main__":
    main()
