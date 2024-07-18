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
from ex1_utils import LOAD_GRAY_SCALE
import cv2


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # Load the image
    img = cv2.imread(img_path, rep)

    # Normalize pixel intensities to [0, 1] range
    img = img / 255.0

    # Define the function to be called when the trackbar is moved
    def gamma_correction(gamma):
        # Convert the integer trackbar value to a float gamma value
        gamma = gamma / 100.0

        # Apply gamma correction to the image
        img_corrected = cv2.pow(img, gamma)
        img_corrected = cv2.normalize(img_corrected, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


        # Display the corrected image
        cv2.imshow('Gamma Correction', img_corrected)

    # Create a named window for the GUI
    cv2.namedWindow('Gamma Correction')

    # Create a trackbar for the gamma value
    cv2.createTrackbar('Gamma', 'Gamma Correction', 100, 200, gamma_correction)

    # Show the original image
    cv2.imshow('Gamma Correction', img)

    # Wait for a key press to close the window
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
