import sys
from typing import List
import math

import numpy as np
import pygame as pygame
from scipy import signal
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208580076

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    kernel_x = np.array([[-1, 0, 1]])
    kernel_y = kernel_x.T

    # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    w_ = int(win_size / 2)

    # Implement Lucas Kanade Algorithm
    # for each point, calculate I_x, I_y, I_t
    fx_drive = cv2.filter2D(im2, -1, kernel_x, borderType=cv2.BORDER_REPLICATE)
    fy_drive = cv2.filter2D(im2, -1, kernel_y, borderType=cv2.BORDER_REPLICATE)
    ft_drive = im2 - im1

    originalPoints = []
    dU_dV = []
    for i in range(w_, im1.shape[0] - w_ + 1, step_size):
        for j in range(w_, im1.shape[1] - w_ + 1, step_size):
            Ix = fx_drive[i - w_:i + w_, j - w_:j + w_].flatten()
            Iy = fy_drive[i - w_:i + w_, j - w_:j + w_].flatten()
            It = ft_drive[i - w_:i + w_, j - w_:j + w_].flatten()
            AtA_ = [[(Ix * Ix).sum(), (Ix * Iy).sum()],
                    [(Ix * Iy).sum(), (Iy * Iy).sum()]]
            lam_ = np.linalg.eigvals(AtA_)
            lam2_ = np.min(lam_)
            lam1_ = np.max(lam_)
            if lam2_ <= 1 or lam1_ / lam2_ >= 100:
                continue
            Atb_ = [[-(Ix * It).sum()], [-(Iy * It).sum()]]
            u_v = np.linalg.inv(AtA_) @ Atb_
            dU_dV.append(u_v)
            originalPoints.append([j, i])
    return np.array(originalPoints), np.array(dU_dV)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    # Build image pyramids
    pyramid1 = [img1]
    pyramid2 = [img2]
    for _ in range(k - 1):
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        pyramid1.append(img1)
        pyramid2.append(img2)

    # Initialize the final optical flow array
    optical_flow = np.zeros_like(pyramid1[0], dtype=np.float32)

    # Initialize the original points and dU_dV arrays
    original_points = []
    dU_dV = []

    # Iterate through each pyramid level, starting from the smallest
    for level in range(k):
        current_img1 = pyramid1[level]
        current_img2 = pyramid2[level]

        # Scale the optical flow from the previous level
        optical_flow = cv2.resize(optical_flow, (current_img1.shape[1], current_img1.shape[0])) * 2

        # Compute optical flow for the current level
        level_points, level_dU_dV = opticalFlow(current_img1, current_img2, stepSize, winSize)

        # Update the optical flow based on the current level
        for i in range(len(level_points)):
            u, v = level_points[i]
            du, dv = level_dU_dV[i]
            optical_flow[v, u] = du
            optical_flow[v, u + 1] = dv

            # Add the points and dU_dV to the final arrays
            original_points.append(level_points[i])
            dU_dV.append(level_dU_dV[i])

    # Convert original_points and dU_dV to NumPy arrays
    original_points = np.array(original_points)
    dU_dV = np.array(dU_dV)

    return original_points, dU_dV


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------
def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    xy, uv = opticalFlow(im1, im2, 20, 5)
    u = []
    v = []
    for a in uv:
        u.append(a[0] * 2)
        v.append(a[1] * 2)

    return np.array([[1, 0, np.median(u)], [0, 1, np.median(v)], [0, 0, 1]])


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """

    xy, uv = opticalFlowPyrLK(im1, im2, stepSize=20, winSize=5, k=5)

    degrees = []

    for point, uvi in zip(xy, uv):
        point = np.flip(point)
        point2 = (point[0] + uvi[1], point[1] + uvi[0])
        ang = find_angle(point, point2)
        degrees.append(abs(ang))
    degrees = [x for x in degrees if x >= 0.1]

    angle = np.mean(degrees) * 2 if np.mean(degrees) * 2 > 0.61 else np.median(degrees) * 2

    xy, uv = opticalFlowPyrLK(im1, im2, stepSize=20, winSize=5, k=5)
    u = []
    v = []
    for a in uv:
        u.append(a[0])
        v.append(a[1])
    u = [x * 2 for x in u if abs(x) >= 0.1]
    v = [x for x in v if abs(x) >= 0.1]

    return np.array([[math.cos(math.radians(angle)), -math.sin(math.radians(angle)), np.median(u)],
                     [math.sin(math.radians(angle)), math.cos(math.radians(angle)), np.median(v)],
                     [0, 0, 1]])


def find_angle(p1, p2):
    v1 = pygame.math.Vector2(p1[0] - 0, p1[1] - 0)
    v2 = pygame.math.Vector2(p2[0] - 0, p2[1] - 0)
    ang = v1.angle_to(v2)
    return ang


def correlation(im1, im2):
    shape = np.max(im1.shape) // 2
    fft1 = np.fft.fft2(np.pad(im1, shape))
    fft2 = np.fft.fft2(np.pad(im2, shape))
    prod = fft1 * fft2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    corr = result_full.real[1 + shape:-shape + 1, 1 + shape:-shape + 1]
    y1, x1 = np.unravel_index(np.argmax(corr), corr.shape)
    y2, x2 = np.array(im2.shape) // 2
    return x1, x2, y1, y2


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    x1, x2, y1, y2 = correlation(im1, im2)
    x = x2 - x1 - 1
    y = y2 - y1 - 1
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float32)


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """

    x1, x2, y1, y2 = correlation(im1, im2)
    angle = find_angle((x1, y1), (x2, y2))

    matrix = np.float32([
        [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
        [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
        [0, 0, 1]
    ])
    matrix = np.linalg.inv(matrix)
    reverse = cv2.warpPerspective(im2, matrix, im2.shape[::-1])

    x1, x2, y1, y2 = correlation(im1, reverse)

    x = x2 - x1 - 1
    y = y2 - y1 - 1

    return np.float32([
        [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), x],
        [math.sin(math.radians(angle)), math.cos(math.radians(angle)), y],
        [0, 0, 1]
    ])

def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    if len(im1.shape) > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    if len(im2.shape) > 2:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    warped_img = np.zeros(im2.shape)
    for row in range(1, im2.shape[0]):
        for col in range(1, im2.shape[1]):
            # we need to find the coordinates of the point in image 1
            # by using the inverse transformation.
            current_vec = np.array([row, col, 1]) @ np.linalg.inv(T)  # calculating the vector
            dx, dy = int(round(current_vec[0])), int(round(current_vec[1]))
            # if the point has a valid coordinates in image 1
            # we put the pixel in the warped image.
            if 0 <= dx < im1.shape[0] and 0 <= dy < im1.shape[1]:
                warped_img[row, col] = im1[dx, dy]  # putting the pixel in the warped image.

    return warped_img


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------

def gaussianKernel():
    """
    Kernel size : 5 X 5
    sigma : 0.3 * ((kernelSize - 1) * 0.5 - 1) + 0.8
    :return: 2D Gaussian kernel
    """
    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    ax = np.linspace(-2., 2., 5)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / kernel.sum()


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """

    gaussianPyramid = []

    # creat the first level (original img) for the pyramid
    # crop the initial image to: (2^x) * (int)(IMG_size/2^x)
    n = 2 ** levels
    width, height = n * int(img.shape[1] / n), n * int(img.shape[0] / n)
    img = cv2.resize(img.copy(), (width, height))
    img = img.astype(np.float64)

    img_i = img.copy()
    gaussianPyramid.append(img_i)

    # Creates a Gaussian Pyramid
    for i in range(1, levels):
        blurImg = cv2.filter2D(img_i, -1, gaussianKernel(), borderType=cv2.BORDER_REPLICATE)
        img_i = blurImg[::2, ::2]
        # img_i = cv2.pyrDown(img_i)
        gaussianPyramid.append(img_i)
    return gaussianPyramid


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    gaussian_pyr = gaussianPyr(img, levels)
    pyramid = [gaussian_pyr[-1]]

    for i in range(levels - 1, 0, -1):
        x, y = gaussian_pyr[i].shape[0], gaussian_pyr[i].shape[1]
        if len(gaussian_pyr[i].shape) == 3:  # RGB
            shape = (x * 2, y * 2, 3)
        else:  # GRAY
            shape = (x * 2, y * 2)
        preImg = np.zeros(shape)
        preImg[::2, ::2] = gaussian_pyr[i]
        expanded = cv2.filter2D(preImg, -1, gaussianKernel()*4, borderType=cv2.BORDER_REPLICATE)
        laplacian = cv2.subtract(gaussian_pyr[i-1], expanded)
        pyramid.append(laplacian)

    pyramid.reverse()
    return pyramid


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    levels = len(lap_pyr)
    img = lap_pyr[-1]
    for i in range(levels - 1, 0, -1):
        expanded = cv2.pyrUp(img)
        img = cv2.add(lap_pyr[i - 1], expanded)
    return img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    # Resize images to mask size
    height, width = mask.shape[0], mask.shape[1]
    img_1 = cv2.resize(img_1, (width, height))
    img_2 = cv2.resize(img_2, (width, height))

    # Pyramid Blending
    lapImg_1 = laplaceianReduce(img_1, levels)
    lapImg_2 = laplaceianReduce(img_2, levels)
    gaussMask = gaussianPyr(mask, levels)

    n = levels - 1
    pyramidBlend = gaussMask[n] * lapImg_1[n] + (1 - gaussMask[n]) * lapImg_2[n]
    for i in range(n - 1, -1, -1):
        upScale = cv2.pyrUp(pyramidBlend)
        pyramidBlend = upScale + gaussMask[i] * lapImg_1[i] + (1 - gaussMask[i]) * lapImg_2[i]

    # Naive Blending
    naiveBlend = mask * img_1 + (1 - mask) * img_2
    naiveBlend = cv2.resize(naiveBlend, (pyramidBlend.shape[1], pyramidBlend.shape[0]))

    return naiveBlend, pyramidBlend

