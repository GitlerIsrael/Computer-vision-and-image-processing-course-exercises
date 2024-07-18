# Ex1 - Image Processing 
This code contains a collection of functions for basic image processing tasks, implemented using the OpenCV and Matplotlib libraries in Python. The functions included in this repository are:

Python version: 3.10
## ex1_utils.py
#### 1. myID():
returns the ID of the user who wrote the code.
#### 2. imReadAndConvert(filename: str, representation: int) -> np.ndarray:
reads an image from the specified file path and converts it to grayscale or RGB representation as requested.
#### 3. imDisplay(filename: str, representation: int):
displays an image in grayscale or RGB representation as requested.
#### 4. transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
converts an RGB image to the YIQ color space.
#### 5. transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
converts an YIQ image to the RGB color space.
#### 6. hsitogramEqualize(imOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
performs histogram equalization on a grayscale or RGB image.
#### 7. quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (np.ndarray, np.ndarray):
performs optimal color quantization on a grayscale or RGB image.

## gamma.py
####  gammaDisplay(im: np.ndarray, gamma: float) -> np.ndarray:
GUI for gamma correction - applies gamma correction on a grayscale or RGB image for display purposes.


#### I also provided 2 images for testing. (in addition to your images).