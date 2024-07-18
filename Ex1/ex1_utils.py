"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 208580076


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Read the image from file
    img = cv2.imread(filename)

    # Convert to the requested representation
    if representation == 1:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == 2:  # RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Representation must be either 1 (grayscale) or 2 (RGB)")

    # Normalize pixel intensities to [0, 1] range
    img = img.astype(np.float32) / 255.0

    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    # Load the image in the requested representation
    img = imReadAndConvert(filename, representation)

    # Determine the title and color map based on the requested representation
    if representation == 1:  # grayscale
        cmap = 'gray'
        title = 'Grayscale Image'
    elif representation == 2:  # RGB
        cmap = None
        title = 'RGB Image'
    else:
        raise ValueError("Representation must be either 1 (grayscale) or 2 (RGB)")

    # Display the image using matplotlib
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # Convert the image to YIQ color space
    T = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.275, -0.321],
                  [0.212, -0.523, 0.311]])
    imYIQ = np.dot(imgRGB, T.T)
    return imYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # Convert the image to RGB color space
    T = np.array([[1.0, 0.956, 0.621],
                  [1.0, -0.272, -0.647],
                  [1.0, -1.107, 1.704]])
    imRGB = np.dot(imgYIQ, T.T)
    return imRGB


def hsitogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imOrig: Original Histogram
        :ret
    """
    if imOrig.ndim == 2:  # grayscale image
        # Normalize the image pixels
        imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the image to uint8 type
        imOrig = imOrig.astype('uint8')

        histOrig, bins = np.histogram(imOrig.flatten(), 256, [0, 256])
        cumSum = histOrig.cumsum()
        normalized_cumSum = cumSum / cumSum[-1]
        lut = np.ceil(normalized_cumSum * 255).astype(np.uint8)
        imEq = lut[imOrig]
        histEq, bins = np.histogram(imEq.flatten(), 256, [0, 256])
        imEq = imEq / 255
    elif imOrig.ndim == 3:  # RGB image
        # convert to YIQ
        imYIQ = transformRGB2YIQ(imOrig)
        # equalize Y channel
        y = imYIQ[:, :, 0]
        # Normalize the image pixels
        y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the image to uint8 type
        y = y.astype('uint8')

        histOrig, bins = np.histogram(y.flatten(), 256, [0, 256])
        cumSum = histOrig.cumsum()
        normalized_cumSum = cumSum / cumSum[-1]
        lut = np.ceil(normalized_cumSum * 255).astype(np.uint8)
        yEq = lut[y]
        histEq, bins = np.histogram(yEq.flatten(), 256, [0, 256])
        yEq = yEq / 255
        imYIQ[:, :, 0] = yEq
        # convert back to RGB
        imEq = transformYIQ2RGB(imYIQ)

    # plot images and histograms
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    ax[0, 0].imshow(imOrig, cmap='gray')
    ax[0, 0].set_title('Original Image')
    ax[0, 1].imshow(imEq, cmap='gray')
    ax[0, 1].set_title('Equalized Image')
    ax[1, 0].plot(histOrig)
    ax[1, 0].set_title('Original Histogram')
    ax[1, 1].plot(histEq)
    ax[1, 1].set_title('Equalized Histogram')
    plt.show()
    return imEq, histOrig, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    rgbFlag = False
    if imOrig.ndim == 3:  # RGB image
        rgbFlag = True
        # Convert the input image to YIQ color space and take only Y channel
        yiqImg = transformRGB2YIQ(imOrig)
        imOrig = yiqImg[:, :, 0]

    quantized_images = []
    error = []
    copy = np.copy(imOrig)
    # Normalize the image pixels and Convert it to uint8 type
    imOrig = cv2.normalize(imOrig, None, 0, 255, cv2.NORM_MINMAX)
    imOrig = imOrig.astype('uint8')

    histOrig, bins = np.histogram(imOrig.flatten(), 256, [0, 256])
    bins = bins[:-1]
    cumSum = np.cumsum(histOrig)
    pixels_per_segment = cumSum[-1] // nQuant
    z = np.zeros(nQuant+1)
    z[0] = 0
    z[-1] = 255
    # Find all borders
    for i in range(1, nQuant):
        border = np.argmin(np.abs(cumSum - pixels_per_segment))
        pixels_per_segment += pixels_per_segment
        z[i] = border
        i += 1
    z[-1] = 255
    z = z.astype(np.int32)

    # # Initialize the segment borders with equal sizes
    # z = np.linspace(0, 255, nQuant + 1, dtype=int)

    # Quantize the image iteratively
    for iter in range(nIter):
        # Compute the values of q based on the current segment borders
        q = np.zeros(nQuant)
        for i in range(nQuant):
            q[i] = sum(histOrig[z[i]:z[i+1]] * bins[z[i]:z[i+1]]) / sum(histOrig[z[i]:z[i+1]])
        # Assign the quantized values to each pixel
        qImg = np.zeros_like(imOrig)
        for i in range(nQuant):
            if i == 0:
                qImg[imOrig <= z[i + 1]] = q[i]
            elif i == nQuant - 1:
                qImg[imOrig > z[i]] = q[i]
            else:
                qImg[(imOrig > z[i]) & (imOrig <= z[i + 1])] = q[i]

        # Compute the error between the original and quantized images
        MSEerror = mse(copy*255, qImg)
        error.append(MSEerror)

        # Update the segment borders for the next iteration
        z[1:-1] = (q[:-1] + q[1:]) / 2

        # Convert the quantized Y channel back to RGB color space
        if rgbFlag:  # RGB image
            yiqImg[:, :, 0] = qImg / 255
            qImg = transformYIQ2RGB(yiqImg)
        else:
            qImg = qImg / 255
        quantized_images.append((qImg))

    # Plot the error as a function of the iteration number
    plt.plot(error)
    plt.xlabel('Iteration number')
    plt.ylabel('MSE error')
    plt.title('MSE as a function of the iteration number')

    # Show the plot
    plt.show()

    return quantized_images, error
