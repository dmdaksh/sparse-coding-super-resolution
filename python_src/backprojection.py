import numpy as np 
from skimage.transform import resize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2

def gauss2D(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

import numpy as np

def bilateral_filter(image, diameter, sigma_color, sigma_space):
    """
    Apply bilateral filtering to an image.

    Parameters:
    - image: 2D numpy array, the grayscale image to filter.
    - diameter: int, the diameter of the pixel neighborhood.
    - sigma_color: float, the filter sigma in the color space.
    - sigma_space: float, the filter sigma in the coordinate space.

    Returns:
    - filtered_image: 2D numpy array, the filtered image.
    """
    m, n = image.shape
    r = diameter // 2
    filtered_image = np.zeros_like(image)

    # Iterate over each pixel in the image
    for i in range(r, m-r):
        for j in range(r, n-r):
            total_weight = 0
            filtered_value = 0

            # Neighborhood pixels
            for k in range(-r, r+1):
                for l in range(-r, r+1):
                    x, y = i + k, j + l

                    # Spatial weight
                    spatial_weight = np.exp(-(k**2 + l**2) / (2 * sigma_space**2))

                    # Range weight
                    range_weight = np.exp(-((image[i, j] - image[x, y]) ** 2) / (2 * sigma_color**2))

                    # Bilateral weight
                    weight = spatial_weight * range_weight

                    filtered_value += image[x, y] * weight
                    total_weight += weight

            filtered_image[i, j] = filtered_value / total_weight if total_weight > 0 else image[i, j]

    return filtered_image


def backprojection(img_sr, img_lr, c, maxIter):
    p = gauss2D((5, 5), 1)
    p = np.multiply(p, p)
    p = np.divide(p, np.sum(p))

    img_initial_sr = img_sr.copy()

    for i in range(maxIter):
        img_lr_ds = resize(img_sr, img_lr.shape, anti_aliasing=1)
        img_diff = img_lr - img_lr_ds

        img_diff = resize(img_diff, img_sr.shape)
        img_sr += convolve2d(img_diff, p, 'same') + c * (img_sr - img_initial_sr)
    return img_sr

def bilateral_filter_opencv(image, d, sigma_color, sigma_space):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# def backprojection(img_sr, img_lr, c, maxIter, diameter=5, sigma_color=12.0, sigma_space=16.0):
#     """
#     Iterative backprojection algorithm using bilateral filtering.

#     Parameters:
#     - img_sr: High-resolution image to be refined.
#     - img_lr: Low-resolution original image.
#     - c: Regularization constant.
#     - maxIter: Maximum number of iterations.
#     - diameter, sigma_color, sigma_space: Parameters for the bilateral filter.

#     Returns:
#     - img_sr: Refined high-resolution image.
#     """
#     # Initialize the filter weights for bilateral filtering
#     # The parameters can be adjusted or passed to the function
#     img_initial_sr = img_sr

#     for i in range(maxIter):
#         # Downscale img_sr to the size of img_lr using resize
#         img_lr_ds = resize(img_sr, img_lr.shape, anti_aliasing=True)
        
#         # Compute the difference between the original LR image and downscaled SR image
#         img_diff = img_lr - img_lr_ds

#         # Upscale the difference back to the high-resolution grid
#         img_diff_us = resize(img_diff, img_sr.shape, anti_aliasing=True)

#         # Apply bilateral filter to the upscaled difference
#         filtered_diff = bilateral_filter(img_diff_us, diameter, sigma_color, sigma_space)

#         # Update the super-resolution image using the filtered difference
#         img_sr += c * (filtered_diff + (img_sr - img_initial_sr))

#     return img_sr