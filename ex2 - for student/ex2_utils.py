import math
import numpy as np
import cv2
from math import sin, cos, radians


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 208580076


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    signal_length = len(in_signal)
    kernel_length = len(k_size)
    conv_length = signal_length + kernel_length - 1
    padded_signal = np.pad(in_signal, (kernel_length - 1, kernel_length - 1), mode='constant')
    convolved_signal = np.zeros(conv_length)

    for i in range(conv_length):
        convolved_signal[i] = np.sum(padded_signal[i:i + kernel_length] * k_size[::-1])

    return convolved_signal


def conv2D(in_signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel, replicating border pixels
    :param in_signal: 2-D array
    :param kernel: 2-D array as a kernel
    :return: The convolved array
    """
    signal_height, signal_width = in_signal.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding sizes
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the input signal by replicating border pixels
    padded_signal = np.pad(in_signal, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
    flipped_kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    # Perform convolution using the same kernel
    convolved_signal = np.zeros_like(in_signal)
    for i in range(signal_height):
        for j in range(signal_width):
            convolved_signal[i, j] = np.sum(padded_signal[i:i+kernel_height, j:j+kernel_width] * flipped_kernel)
    return convolved_signal


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Gray scale image
    :return: (directions, magnitude)
    """
    kernel = np.array([[1, 0, -1]])
    dx = conv2D(in_image, kernel)
    dy = conv2D(in_image, kernel.T)

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    direction = np.arctan2(dy, dx)

    return direction, magnitude


def getGaussianKernel(k_size: int) -> np.ndarray:
    # Create 1D Gaussian kernel using binomial coefficients
    kernel1D = np.ones((1, k_size), dtype=np.float32)
    for i in range(1, k_size):
        kernel1D[0, i] = kernel1D[0, i - 1] * (k_size - i) / i
    kernel1D /= np.sum(kernel1D)

    # Create 2D Gaussian kernel by convolving 1D kernel with its transpose
    kernel2D = conv2D(kernel1D, kernel1D.T)

    return kernel2D


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    kernel = getGaussianKernel(k_size)
    blurred_image = conv2D(in_image, kernel)

    return blurred_image


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    kernel = cv2.getGaussianKernel(k_size, sigma=0.3 * ((k_size - 1) * 0.5 - 1) + 0.8)
    blurred_image = cv2.filter2D(in_image, -1, kernel)

    return blurred_image


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    return edgeDetectionZeroCrossingLOG(img)


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    img = np.float32(img)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    # blurred = blurImage1(img, 9)
    laplacian = cv2.Laplacian(blurred, cv2.CV_32F)
    rows, cols = laplacian.shape

    zero_crossings = np.zeros_like(laplacian)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [
                laplacian[i - 1, j - 1], laplacian[i - 1, j], laplacian[i - 1, j + 1],
                laplacian[i, j - 1], laplacian[i, j + 1],
                laplacian[i + 1, j - 1], laplacian[i + 1, j], laplacian[i + 1, j + 1]
            ]
            positive_crossings = np.sum(np.array(neighbors) > 0)
            negative_crossings = np.sum(np.array(neighbors) < 0)
            if positive_crossings > 0 and negative_crossings > 0:
                zero_crossings[i, j] = 255

    return zero_crossings.astype(np.uint8)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    if img.max() <= 1:
        img = (cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)).astype('uint8')

    radius_by_shape = min(img.shape[0], img.shape[1]) // 2
    max_radius = min(radius_by_shape, max_radius)

    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    direction = np.arctan2(gradient_y, gradient_x) - radians(90)

    canny_edges = cv2.Canny(img, 75, 150)
    accumulator = np.zeros((img.shape[0], img.shape[1], max_radius + 1))

    for (x, y), edge in np.ndenumerate(canny_edges):
        if edge == 255:
            for rad in range(min_radius, max_radius + 1):
                angle = direction[x, y]
                x1, x2 = x - int(rad * cos(angle)), x + int(rad * cos(angle))
                y1, y2 = y + int(rad * sin(angle)), y - int(rad * sin(angle))
                if 0 < x1 < accumulator.shape[0] and 0 < y1 < accumulator.shape[1]:
                    accumulator[x1, y1, rad] += 1
                if 0 < x2 < accumulator.shape[0] and 0 < y2 < accumulator.shape[1]:
                    accumulator[x2, y2, rad] += 1

    threshold = np.max(accumulator) * 0.45 + 1
    x, y, rad = np.where(accumulator >= threshold)
    detected_circles = [(y[i], x[i], rad[i]) for i in range(len(x)) if (x[i], y[i], rad[i]) != (0, 0, 0)]

    return detected_circles


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    in_image = np.float32(in_image)

    # Apply bilateral filter using OpenCV
    cv2_output = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)

    # My implementation
    # Create an empty output image with the same shape as the input image
    out_image = np.zeros_like(in_image)
    k2_size = k_size // 2
    # padded_image = np.pad(in_image, k2_size, mode='replicate')
    padded_image = cv2.copyMakeBorder(in_image, k2_size, k2_size, k2_size, k2_size, cv2.BORDER_REPLICATE)

    # calculate the spatial weights only one time (same one for all)
    spatial_weights = np.zeros((k_size, k_size))
    for k in range(k_size):
        for l in range(k_size):
            d = ((k - k2_size) ** 2 + (l - k2_size) ** 2) ** 0.5
            spatial_weights[k, l] = np.exp(-d ** 2 / (2 * sigma_space ** 2))

    # Iterate over each pixel in the image
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            # find neighborhood
            neighborhood = padded_image[i:i + k_size, j:j + k_size]
            color_diffs = np.abs(in_image[i, j] - neighborhood)
            color_weights = np.exp(-(color_diffs ** 2) / (2 * sigma_color ** 2))

            # combine the space and color weights
            combo = spatial_weights * color_weights

            # Apply the bilateral filter to the current pixel
            out_image[i, j] = np.sum(neighborhood * combo) / np.sum(combo)

    return cv2_output, out_image
