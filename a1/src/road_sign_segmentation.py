import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

BGR_RED = (0, 0, 255)


def detect_edges(img, threshold1, threshold2):
    '''TODO: documentation'''
    return cv.Canny(img, threshold1, threshold2)

def main():
    '''TODO: documentation'''

    img0 = cv.imread(str(ROOT)+'/data/street_signs/images0.jpg', cv.IMREAD_GRAYSCALE)
    img1 = detect_edges(img0, 250, 300)
    img2 = cv.imread(str(ROOT)+'/data/street_signs/templates/diamond1.png', cv.IMREAD_GRAYSCALE)
    # img2 = cv.GaussianBlur(img2, ksize=(5, 5), sigmaX=0, borderType=cv.BORDER_ISOLATED)

    orb = cv.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3)
    plt.show()

    # read_fp = str(Path(ROOT, 'data', 'street_signs', 'images0.jpg'))
    # img = cv.imread(read_fp, cv.IMREAD_COLOR)
    # edges = detect_edges(img, 150, 250)

    # # Convert segmented edges to 3-channel image to enable colour overlay
    # edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # # Load templates
    # templates = []
    # for template_name in ('diamond0.png', 'diamond1.png', 'diamond2.png'):
    #     fp = str(Path(ROOT, 'data', 'street_signs', 'templates', template_name))
    #     template = cv.imread(fp, 0)

    #     # Convert black-and-white templates to 3-channel to enable colour overlay
    #     if len(template.shape) == 2:
    #         template = cv.cvtColor(template, cv.COLOR_GRAY2BGR)

    #     # Apply Gaussian blur to template
    #     template = cv.GaussianBlur(template, ksize=(5, 5), sigmaX=0, borderType=cv.BORDER_ISOLATED)

    #     templates.append(template)

    # for template in templates:

    #     # Scale template
    #     for scale in np.arange(0.1, 1.0, 0.05):
    #         # scale = 1.0
    #         scaled_h = int(template.shape[0] * scale)
    #         scaled_w = int(template.shape[1] * scale)
    #         scaled_template = cv.resize(template, (scaled_w, scaled_h))
    #         h, w, _ = scaled_template.shape

    #         if h > edges.shape[0] or w > edges.shape[1]:
    #             continue

    #         res = cv.matchTemplate(edges, scaled_template, cv.TM_CCORR_NORMED)
            
    #         threshold = 0.4
    #         _, max_val, _, max_loc = cv.minMaxLoc(res)
    #         if max_val > threshold:
    #             top_left = max_loc
    #             bottom_right = (top_left[0] + w, top_left[1] + h)
    #             cv.rectangle(edges, top_left, bottom_right, BGR_RED, 1)
    #         # loc = np.where(res >= threshold)
    #         # for pt in zip(*loc[::-1]):
    #         #     cv.rectangle(edges, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    # # Show template overlay in the top left corner
    # # overlay = cv.addWeighted(edges[0:h, 0:w], 1.0, scaled_template, 1.0, 0)
    # # edges[0:h, 0:w] = overlay

    # write_fp = str(Path(ROOT, 'output', 'street_signs', 'result0.png'))
    # cv.imwrite(write_fp, edges)

if __name__ == '__main__':
    main()