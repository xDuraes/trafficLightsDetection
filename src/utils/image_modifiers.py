import cv2 as cv
from cv2.typing import MatLike

def applyMask(image:MatLike, boundaries:list) -> MatLike:
    """
    Function to apply a mask to isolate colors in an image

    :param image: OpenCV processed image
    :type image: MatLike
    :param boundaries: List of the n boundaries to be applied in the image sturctured as follows: 
    [(lower boundary, upper boundary)]
    :type boundaries: list
    :return: Description
    :rtype: MatLike
    """
    img = image
    for lower_boundary, upper_boundary in boundaries:
        img = cv.inRange(img, lower_boundary, upper_boundary)
    return img