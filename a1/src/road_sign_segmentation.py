import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Path to the project root, i.e. "a1"
A1_ROOT = Path(__file__).parent.parent.resolve()


def process_image(img, alpha=0.4, beta=0.6, d=15, sigmaColor=25, sigmaSpace=75, thresh=185):
    '''
    Processes an image to remove all image details except street signs.
    This is done by taking the red channel and saturation channel of img,
    then blurring and binarizing the resulting image.

    The red channel and saturation channel are chosen as most street signs
    have relatively high red intensity, such as those with yellow or red
    features, and high saturation compared to the background.

    Parameters:
        img - image to process
        alpha - fractional contribution of red channel
        beta - fractional contribution of saturation channel
        d - blurring kernel diameter (using a bilateral filter)
        sigmaColor - blurring sigma for dissimilar colours
        sigmaSpace - blurring sigma for dissimilar spatial regions
        thresh - binarization threshold (out of 255)

    Returns:
        Black and white image with (ideally) most or all elements removed
        from the original image except street signs.
    '''
    img_red = img[:, :, 2]

    # Yellow signs tend to have high saturation relative to background
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_sat = img_hsv[:, :, 1]

    # Combine red channel and saturation channel to reduce intensity of 
    # saturated but non-reddish (incl. non-yellow) background
    img_prod = cv.addWeighted(img_red, alpha, img_sat, beta, 0)

    # Blur saturation image using bilateral filter to preserve edges
    img_blurred = cv.bilateralFilter(img_prod, d, sigmaColor, sigmaSpace)

    # Binarize saturation image
    maxval = np.amax(img_blurred)
    _, img_binarized = cv.threshold(img_blurred, thresh, maxval, cv.THRESH_BINARY)

    return img_binarized


def bounding_rect(img, dst, colour=(255, 0, 0), lw=1):
    '''
    Draws a bounding rectangle around all white points in the provided image.

    Parameters:
        img - image used to find bounding box; assumed to be single-channel
        dst - image that bounding box is applied to; assumed to be RGB
        colour - colour of bounding box (default red)
        lw - width of bounding box in pixels

    Returns:
        Image dst overlayed with a rectangular bounding box over all white
        ares in image img.
    '''
    x, y, w, h = cv.boundingRect(img)
    cv.rectangle(dst, (x, y), (x + w, y + h), colour, lw)
    return dst


def main():
    '''
    Visualise all 11 street sign samples with segmentation algorithm applied.
    Identified street signs are marked inside red rectangles.
    '''
    _, axs = plt.subplots(3, 4)
    
    for i, ax in enumerate(axs.flat):
        ax.axis('off')
        if i < 11:
            img_fp = Path(A1_ROOT, 'data', 'street_signs', f'images{i}.jpg')

            img_raw = cv.imread(str(img_fp))
            img_rgb = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)

            img_processed = process_image(img_raw)
            img_segmented = bounding_rect(img_processed, img_rgb)

            ax.imshow(img_segmented, cmap='gray')
    
    plt.show()


if __name__ == '__main__':
    main()