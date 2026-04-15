# local modules
from constants.contours import MIN_AREA_CONTOUR, CIRCULARITY_INTERVAL
# non-native modules
from cv2 import (
    inRange, findContours, drawContours, HoughCircles, circle, contourArea, arcLength,
    RETR_TREE, CHAIN_APPROX_SIMPLE, HOUGH_GRADIENT
)
from cv2.typing import MatLike
from numpy import(
    around,
    uint16, 
    pi
)

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

def _validate_contours(contour, minArea:int=50, circularityInterval:tuple=(0.7, 1.2)):
    """
    Function to validate the area and circularity of a contour
    
    :param contour: A contour of the contours returned by findContours()
    :param minArea: Minimum area required for a contour to be considered
    :type minArea: int
    :param circularityInterval: Tuple containing the minimum and maximum circularity for a contour to be validated, respectivelly
    :type circularityInterval: tuple
    """
    # Small noise filtering
    area = contourArea(contour)
    if area < minArea: 
        return False

    # circularity calculation
    perimeter = arcLength(contour, True)
    # safety check for zero division error
    if perimeter == 0:
        return False

    circularity = (4 * pi * area) / (perimeter ** 2)

    # circularity validation
    min_circularity, max_circularity = circularityInterval[0], circularityInterval[1]
    if circularity < min_circularity or circularity > max_circularity:
        return False
    
    return True

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
    contours, _ = findContours(
        image=masked_image,
        mode=RETR_TREE,
        method=CHAIN_APPROX_SIMPLE
    )

    valid_contours = []
    for contour in contours:
        if _validate_contours(contour, MIN_AREA_CONTOUR, CIRCULARITY_INTERVAL):
            valid_contours.append(contour)

    img_contour = drawContours(image, valid_contours, -1, color, 3)

    return img_contour

def findCircles(
        image:MatLike,
        dp:float,
        minDist:float,
        param1:float,
        param2:float,
        minRadius:int,
        maxRadius:int,
):
    """
    
    """  
    
    # circle detection
    circles = HoughCircles(
        image=image,
        method=HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    return circles

def drawCircles(image:MatLike, circles:MatLike, color:tuple, drawCenter:bool):
    
    if color not in [(0, 255, 0), (0, 255, 255), (0, 0, 255)]:
        raise ValueError(f"Color {color} does not exist in a traffic light")

    # circle drawing in image
    if circles is not None:
        circles = uint16(around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            if drawCenter:
                circle(image, center, 1, color, 3)
            # circle radius
            radius = i[2]
            circle(image, center, radius, color, 3)

    return image