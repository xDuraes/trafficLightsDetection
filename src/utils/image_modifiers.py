import cv2 as cv
from cv2.typing import MatLike

def applyMask(image:MatLike, boundaries:tuple) -> MatLike:
    """
    Function to apply a mask to isolate colors in an image

    :param image: OpenCV processed image
    :type image: MatLike
    :param boundaries: tuple of two elements containing the lower and upper boundary, respectively
    :type boundaries: tuple
    :return: Returns an binarized masked version of the original image
    :rtype: MatLike
    """
    lower_boundary, upper_boundary = boundaries

    masked_image = cv.inRange(image, lower_boundary, upper_boundary)

    return masked_image