import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Tuple

### GLOBAL VARIABLES ###########################################################

# Path to the project root, i.e. "a1"
A1_ROOT = Path(__file__).parent.parent.resolve()

### TYPEDEFS ###################################################################

# Define an "image" type, which is more intuitive than "np.ndarray"
Image = np.ndarray

# Define a "rectangle" type, describing a top-left corner and width and height
Rectangle = Tuple[int, int, int, int]

### IMAGE PROCESSING ###########################################################

# def find_signs(img: Image) -> List[Rectangle]:
#     ''''''
#     img = cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 1]
#     img = cv.threshold(img, thresh=150, maxval=255, type=cv.THRESH_BINARY)[1]
#     # img = cv.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv.INTER_CUBIC)
#     # img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=np.ones((3, 3)))
#     # img = cv.erode(img, kernel=np.ones((5, 5)))

#     plt.imshow(img, cmap='gray')
#     plt.show()

#     # contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     # print('Contours detected:', len(contours))

#     # contours = sorted(contours, key=lambda c: cv.contourArea(c), reverse=True)

#     # for contour in contours:
#     #     img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
#     #     cv.drawContours(img_rgb, [contour,], -1, color=(255, 0, 0))
#     #     plt.imshow(img_rgb)
#     #     plt.show()
#     #     break

def template_diamond() -> Image:
    ''''''
    template = np.zeros((100, 100), dtype=np.uint8)

    for i in range(template.shape[0]):

        for j in range(template.shape[1]):

            if i <= 50 and i > 50 - j and i > j - 50:
                template[i][j] = 1

            elif i > 50 and i < j + 50 and i < 150 - j:
                template[i][j] = 1

    return template

def template_rect_h(height: int) -> Image:
    ''''''
    template = np.zeros((height + 2, 100), dtype=np.uint8)
    template[1:-1, :] = 1
    return template

def template_rect_v(width: int) -> Image:
    ''''''
    template = np.zeros((100, 100), dtype=np.uint8)
    template[:, (50 - int(width / 2)):(50 + int(width / 2))] = 1
    return template

def template_rectangle(aspect_ratio: float) -> Image:
    ''''''
    if aspect_ratio == 1:
        return np.ones((100, 100), dtype=np.uint8)

    if aspect_ratio > 1:
        return template_rect_h(int(100 / aspect_ratio))

    if aspect_ratio < 1:
        return template_rect_v(int(100 * aspect_ratio))

def rect_overlap(rect1: Rectangle, rect2: Rectangle) -> float:
    '''
    Returns the number of overlapping pixels as a percentage of the total
    number of pixels in the smaller rectangle.
    '''
    x1, y1, w1, h1 = rect1; area1 = w1 * h1
    x2, y2, w2, h2 = rect2; area2 = w2 * h2

    bounding_w = max(x1 + w1, x2 + w2) - min(x1, x2)
    bounding_h = max(y1 + h1, y2 + h2) - min(y1, y2)

    bounding_array_1 = np.zeros((bounding_w, bounding_h), dtype=np.uint8)
    bounding_array_2 = np.zeros((bounding_w, bounding_h), dtype=np.uint8)

    x_offset = min(x1, x2)
    y_offset = min(y1, y2)

    x1_offset = x1 - x_offset; y1_offset = y1 - y_offset
    x2_offset = x2 - x_offset; y2_offset = y2 - y_offset

    bounding_array_1[x1_offset:(x1_offset + w1), y1_offset:(y1_offset + h1)] = 1
    bounding_array_2[x2_offset:(x2_offset + w2), y2_offset:(y2_offset + h2)] = 1

    overlap = np.sum(bounding_array_1 & bounding_array_2)

    return overlap / min(area1, area2)

def remove_overlapping(ordered_rects, threshold: float):
    ''''''
    # Sort list by ordering key
    ordered_rects = sorted(ordered_rects, key=lambda t: t[1], reverse=True)

    non_overlapping: List[Rectangle] = [ordered_rects[0][0],]

    for rect, _ in ordered_rects[1:]:
        for existing_rect in non_overlapping:
            if rect_overlap(rect, existing_rect) > threshold:
                break
        else:
            non_overlapping.append(rect)

    return non_overlapping

def find_signs(img: Image, templates: List[Image],
        template_threshold: float = 0.8) -> List[Rectangle]:
    ''''''
    img_red = img[:, :, 2]
    img_sat = cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 1]

    img = cv.addWeighted(img_red, 0.4, img_sat, 0.6, 0)
    img = cv.threshold(img, 185, 255, cv.THRESH_BINARY)[1]
    img = cv.dilate(img, kernel=np.ones((3, 3)))

    contours, _ = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contour_convex_ratio = \
        lambda c: cv.contourArea(cv.convexHull(c)) / cv.contourArea(c)
    contours = [c for c in contours if 0.8 < contour_convex_ratio(c) < 1.2]
    cv.drawContours(img, contours, -1, 255, -1)

    ordered_rects = []

    for template, scale_range in templates:

        for scale in scale_range:
            scaled_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
            scaled_template = cv.resize(template, scaled_size, interpolation=cv.INTER_AREA)

            try:
                match = cv.matchTemplate(img, scaled_template, cv.TM_CCORR_NORMED)
            except cv.error:
                continue

            h, w = scaled_template.shape[:2]

            loc = np.where(match >= template_threshold)
            for y, x in zip(*loc):
                ordered_rects.append(((x, y, w, h), match[y, x]))

    # ordered_rects = sorted(ordered_rects, key=lambda t: t[1], reverse=True)
    # for rect, key in ordered_rects:
    #     img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    #     x, y, w, h = rect
    #     if 1.4 < w / h < 1.6:
    #     # if w != h:
    #         continue
    #     print('Match:', key)
    #     cv.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0))
    #     plt.imshow(img_rgb)
    #     plt.show()

    if ordered_rects == []:
        return ordered_rects

    return remove_overlapping(ordered_rects, threshold=0.15)

### ENTRYPOINT #################################################################

def main():

    # Get the filepath of the sample images
    imgsrc_fp = str(Path(A1_ROOT, 'data', 'street_signs'))
    for _, _, files in os.walk(imgsrc_fp):
        imgs_fp = sorted(files, key=lambda f: int(f[6:-4]))
        break

    # Read in the sample images
    imgs = [cv.imread(str(Path(imgsrc_fp, img_fp))) for img_fp in imgs_fp]

    # Create templates for each sign shape
    templates = [
        (template_diamond(), np.linspace(0.35, 1.25, 24)),
        (template_rectangle(aspect_ratio=1.5), np.linspace(1.00, 1.20, 5)),
        (template_rectangle(aspect_ratio=5), np.linspace(1.20, 1.30, 3)),
    ]

    # # Identify the sign/s in each image, and draw a rectangle around each
    # for img in imgs[2:3]:
    #     sign_boxes = find_signs(img, templates)
    #     # for x, y, w, h in sign_boxes:
    #     #     cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))

    #     # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    #     # plt.show()

    fig, axs = plt.subplots(3, 4)

    fig.tight_layout()

    for i, ax in enumerate(axs.flat):
        if i < 11:
            sign_boxes = find_signs(imgs[i], templates)
            img_rgb = cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB)
            for x, y, w, h in sign_boxes:
                cv.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0))
            ax.imshow(img_rgb)

    plt.show()


if __name__ == '__main__':
    main()