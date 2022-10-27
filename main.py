import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from imageio import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu, threshold_triangle
from skimage.morphology import binary_opening, binary_closing
from skimage.measure import regionprops
from skimage import measure


def do_threshold(image, sizes, thresh):
    gray = rgb2gray(gaussian(image, sigma=1.5, channel_axis=True))
    if thresh:
        threshold = threshold_triangle(gray)
    else:
        threshold = threshold_otsu(gray)
    mask = gray <= threshold
    mask = binary_closing(binary_opening(mask, footprint=np.ones((sizes[0], sizes[0]))),
                          footprint=np.ones((sizes[1], sizes[1])))
    return mask


def masks_for_primitives(image, mask, n, coeffs):
    labels = measure.label(mask)
    props = regionprops(labels)
    # find
    object_position = ((image.shape[0] / 2) * coeffs[0], (image.shape[1] / 2) * coeffs[1])
    mask = (labels == (np.array([np.linalg.norm(object_position- np.array(prop.centroid)) for prop in props]).argmin() + 1))
    visMask = (mask * 255).astype("uint8")
    res = cv2.bitwise_and(image, image, mask=visMask)
    plt.imsave(fname=str(n) + ".jpg", arr=res)

def get_primitives_images():
    for n in range(0, 10):
        imagep = os.path.join("./Primitives", str(n) + '.JPG')
        # sizes of footprint parametr in morphological operations
        sizes = [10, 20]
        # type of threshold best fitting images: 0 for otsu, 1 for triangle
        threshold = 0
        # coefficient of object relative to image center
        place = [1, 1]
        if n in [6, 7]:
            threshold = 1
        if n not in [2, 3]:
            place = [1.1, 1.1]
            if n == 5:
                place = [1.0, 1.2]
        if n in [8]:
            sizes[1] = 10
        image = imread(imagep)
        mask = do_threshold(image, sizes, threshold)
        masks_for_primitives(image, mask, n, place)


get_primitives_images()

