from cv2 import inRange, findContours, drawContours, RETR_TREE, CHAIN_APPROX_SIMPLE
from cv2.typing import MatLike

def applyMask(image:MatLike, boundaries:tuple) -> MatLike:
    """
    Function to apply a mask to isolate colors in an image.

    :param image: OpenCV processed image
    :type image: MatLike
    :param boundaries: tuple of two elements containing the lower and upper boundary, respectively
    :type boundaries: tuple
    :return: Returns an binarized masked version of the original image
    :rtype: MatLike
    """
    lower_boundary, upper_boundary = boundaries

    masked_image = inRange(image, lower_boundary, upper_boundary)

    return masked_image

def findAndDrawContours(image:MatLike, masked_image:MatLike, color:tuple):
    """
    Function to find and draw contours in an specific color. 
    Contours recognition are made in a binarized masked image and drawn in any image. 
    
    :param image: Image in which the contours will be writen
    :type image: MatLike
    :param masked_image: Binarized masked image
    :type masked_image: MatLike
    :param color: BGR tuple containing the color in which the contours will be writen
    :type color: tuple
    """
    contour, _ = findContours(
        image=masked_image,
        mode=RETR_TREE,
        method=CHAIN_APPROX_SIMPLE
    )
    
    img_contour = drawContours(image, contour, -1, color, 3)

    return img_contour 
