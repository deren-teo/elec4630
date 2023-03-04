import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Path to the project root, i.e. "a1"
A1_ROOT = Path(__file__).parent.parent.resolve()


def display_colour_channels():
    '''TODO: documentation'''
    
    for j in range(11):

        img_fp = Path(A1_ROOT, 'data', 'street_signs', f'images{j}.jpg')
        img_raw = cv.imread(str(img_fp), cv.IMREAD_COLOR)
        img_gry = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)
        img_rgb = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)
        img_hsv = cv.cvtColor(img_raw, cv.COLOR_BGR2HSV)

        _, axs = plt.subplots(3, 3)

        axs[0][0].imshow(img_rgb)
        axs[0][0].axis('off')
        axs[0][0].set_title('Original')

        axs[0][1].imshow(img_gry, cmap='gray')
        axs[0][1].axis('off')
        axs[0][0].set_title('Grayscale')

        axs[0][2].axis('off')

        for i in range(3):
            axs[1][i].imshow(img_rgb[:, :, i])
            axs[1][i].axis('off')
            axs[1][i].set_title(f'RGB[{i}]')

        for i in range(3):
            axs[2][i].imshow(img_hsv[:, :, i])
            axs[2][i].axis('off')
            axs[2][i].set_title(f'HSV[{i}]')

        plt.show()

def process_image(img, thresh=0):
    '''
    Attempt to process an image containing one or more street signs into a
    binary image, where the area of any street signs is white, and the
    remaining area is black.
    '''
    
    # Yellow signs tend to have high saturation relative to background
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_sat = img_hsv[:, :, 1]

    # Blur saturation image
    # img_blurred = cv.blur(img_sat, ksize=(7, 7))
    # img_blurred = cv.GaussianBlur(img_sat, ksize=(11, 11), sigmaX=0)
    # img_blurred = cv.medianBlur(img_sat, ksize=7)
    img_blurred = cv.bilateralFilter(img_sat, d=9, sigmaColor=75, sigmaSpace=75)

    # Binarize saturation image
    maxval = np.amax(img_blurred)
    _, img_binarized = cv.threshold(img_blurred, thresh, maxval, cv.THRESH_BINARY)

    img_blurred = cv.blur(img_binarized, ksize=(7, 7))

    # maxval = np.amax(img_blurred)
    # _, img_binarized = cv.threshold(img_blurred, 100, maxval, cv.THRESH_BINARY)

    # Filter image to reduce noise
    # img_filtered = cv.blur(img_binarized, ksize=(11, 11))
    # img_filtered = cv.GaussianBlur(img_binarized, ksize=(11, 11), sigmaX=0)
    # img_filtered = cv.medianBlur(img_binarized, ksize=3)
    # img_filtered = cv.bilateralFilter(img_binarized, d=9, sigmaColor=75, sigmaSpace=75)

    # Binarize again after blurring
    # _, img_filtered = cv.threshold(img_filtered, 127, 255, cv.THRESH_BINARY)

    # # Upscale image before morphological transforms
    # scaled_h = img_binarized.shape[0] * 4
    # scaled_w = img_binarized.shape[1] * 4
    # img_upscaled = cv.resize(img_binarized, (scaled_w, scaled_h), interpolation=cv.INTER_CUBIC)

    # Morphologically erode image to remove noise
    # kernel = np.ones((3, 3), np.uint8)
    # img_eroded = cv.morphologyEx(img_binarized, cv.MORPH_ERODE, kernel)

    return img_blurred

# def segment_image(img):
#     '''TODO: documentation'''

#     kernel = np.ones((71, 71), np.uint8)
#     sure_bg = cv.dilate(img, kernel)

#     kernel = np.ones((71, 71), np.uint8)
#     sure_fg = cv.erode(img, kernel)

#     sure_fg = np.uint8(sure_fg)
#     unknown = cv.subtract(sure_bg, sure_fg)

#     _, markers = cv.connectedComponents(sure_fg)
#     markers = markers + 1
#     markers[unknown==255] = 0

#     img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#     markers = cv.watershed(img, markers)
#     img[markers==-1] = [255, 0, 0]

#     return markers

def create_template(r):
    '''TODO: documentation'''
    template = np.zeros((r, r, 3), np.uint8)
    template = cv.circle(template, (r // 2, r // 2), r // 6, (255, 0, 0), r // 3)
    template = cv.GaussianBlur(template, ksize=(11, 11), sigmaX=0)
    return template[:, :, 0]

def template_match(img, dst, threshold=0.5, template_size=51):
    '''TODO: documentation'''
    template = create_template(template_size)
    w, h = template.shape

    match_result = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(match_result >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(dst, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 1)

    return dst

def main():
    '''TODO: documentation'''

    img_fp = Path(A1_ROOT, 'data', 'street_signs', 'images0.jpg')
    img_raw = cv.imread(str(img_fp))
    img_rgb = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)
    img_processed = process_image(img_raw, thresh=200)
    img_segmented = template_match(img_processed, img_rgb)

    plt.imshow(img_segmented)
    plt.show()


if __name__ == '__main__':
    main()