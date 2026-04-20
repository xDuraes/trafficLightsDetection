# local modules
from constants.contours import COLOR2POSITION
# non-native modules
from cv2 import (
    inRange, findContours, drawContours, HoughCircles, circle, contourArea, arcLength, boundingRect, cvtColor, 
    bilateralFilter, getStructuringElement, morphologyEx, rectangle, createCLAHE, adaptiveThreshold,
    CHAIN_APPROX_SIMPLE, HOUGH_GRADIENT, COLOR_BGR2GRAY, MORPH_RECT, MORPH_CLOSE, 
    RETR_EXTERNAL, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV
)
from cv2.typing import MatLike
from numpy import(
    around,
    uint16, ndarray,
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

def validateContours(contour, minArea:int=50, circularityInterval:tuple=(0.7, 1.2)) -> bool:
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

def returnContours(masked_image:MatLike):
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
        mode=RETR_EXTERNAL,
        method=CHAIN_APPROX_SIMPLE
    )

    return contours

def drawContoursInImage(image:MatLike, contours:ndarray, color:tuple[int,int,int]):
    """
    Function to draw given contours into an image

    :param image: Image in which the contours will be drawn
    :type image: MatLike
    :param contours: Contours to be drawn
    :type contours: ndarray
    :param color: Color to draw the contours
    :type color: tuple[int, int, int]
    """

    img_contour = drawContours(image, contours, -1, color, 3)

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
    Function to find circles in an image, better used in masked images
    
    :param image: Image in which the circles will be searched
    :type image: MatLike
    :param dp: dp parameter for HoughCircles function
    :type dp: float
    :param minDist: Min distance in between circles
    :type minDist: float
    :param param1: param1 parameter for HoughCircles function
    :type param1: float
    :param param2: param2 parameter for HoughCircles function
    :type param2: float
    :param minRadius: Min radius for a circle to be considered
    :type minRadius: int
    :param maxRadius: Max radius for a circle to be considered
    :type maxRadius: int
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

def drawCircles(image:MatLike, circles:MatLike, color:tuple[int,int,int], drawCenter:bool):
    """
    Function to draw circles in an image
    
    :param image: Image in which the circles will be drawn
    :type image: MatLike
    :param circles: Contours of the circles to be drawn
    :type circles: MatLike
    :param color: Color in which the circles will be drawn
    :type color: tuple[int, int, int]
    :param drawCenter: Boolean to determine if a dot will be drawn in the center of the circle
    :type drawCenter: bool
    """
    
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

def findTrafficLightContour(image:MatLike, circleContour:ndarray, color:tuple[int,int,int] = None):
    """
    Function to find the retangular contour of the traffic light
    
    :param image: Image in which the retangular contour will be searched
    :type image: MatLike
    :param circleContour: Contours of the circles inside the traffic light
    :type circleContour: ndarray
    :param color: Color to draw the contour
    :type color: tuple[int, int, int]
    """

    x_top_left_circle, y_top_left_circle, width_circle, height_circle = boundingRect(circleContour)

    # traffic lights have approximately 3 lights of height and 1 light of width
    # we are using this as a base for estimating its proportions
    estimated_height = int(height_circle * 3.5)
    estimated_width = int(width_circle * 1.6)
    
    # centralizing the searching area
    circle_x = x_top_left_circle + width_circle // 2
    circle_y = y_top_left_circle + height_circle // 2
    
    if color and color in COLOR2POSITION:
        # calculating a relative position in which the circle must be based in its color
        light_position = COLOR2POSITION[color]
        anchor_y = circle_y - int(light_position * estimated_height) + estimated_height // 2
    else:
        # fallback to centralizate on the circle not based on color
        anchor_y = circle_y

    # selecting the area to crop
    area_x = max(0, circle_x - estimated_width // 2)
    area_y = max(0, anchor_y - estimated_height // 2)
    area_w = min(image.shape[1] - area_x, estimated_width)
    area_h = min(image.shape[0] - area_y, estimated_height)
    
    if area_w <= 0 or area_h <= 0:
        return None

    area_to_search = image[area_y:area_y + area_h, area_x:area_x + area_w]
    
    gray_area = cvtColor(area_to_search, COLOR_BGR2GRAY)
    
    # enhancing local contrast
    clahe = createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray_area)
    
    blurred = bilateralFilter(enhanced, d=7, sigmaColor=40, sigmaSpace=40)
    
    # local binarization
    binary = adaptiveThreshold(
        blurred, maxValue=255,
        adaptiveMethod=ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=THRESH_BINARY_INV,
        blockSize=15,
        C=4
    )
    
    # closing gaps in the borders
    kernel = getStructuringElement(MORPH_RECT, (3, 3))
    closed = morphologyEx(binary, MORPH_CLOSE, kernel)
    
    contours, _ = findContours(closed, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    min_area = (width_circle * height_circle) * 0.1
    borders_candidates = []
    
    for contour in contours:
        hx, hy, hw, hh = boundingRect(contour)
        
        # if contour has an area smaller than a determined min area it is ignored
        if contourArea(contour) < min_area:
            continue
        
        aspect_ratio = hh / hw if hw > 0 else 0
        if aspect_ratio >= 1.5:
            borders_candidates.append((hx, hy, hw, hh))
            
    
    if borders_candidates:
        # gathering the union of the biggest and smallest contours to avoid cropping borders
        x_min = min(border[0] for border in borders_candidates)
        y_min = min(border[1] for border in borders_candidates)
        x_max = max(border[0] + border[2] for border in borders_candidates)
        y_max = max(border[1] + border[3] for border in borders_candidates)
        hx, hy, hw, hh = x_min, y_min, x_max - x_min, y_max - y_min
    else:
        # fallback to get all the cropped image
        hx, hy, hw, hh = 0, 0, area_w, area_h

    final_ratio = hh / hw if hw > 0 else 0
    if final_ratio < 1.2:
        return None
    
    return (area_x + hx, area_y + hy, hw, hh)

def drawBoxesInImage(image:MatLike, box:list[tuple[int,int,int,int]], color:tuple[int,int,int]):
    """
    Function to draw rectangles in an image
    
    :param image: Image in which the rectangles will be drawn
    :type image: MatLike
    :param box: A tuple containing the informations refering to the rectangle (x top left, y top left, width, height)
    :type box: list[tuple[int, int, int, int]]
    :param color: Color in which the rectangle will be drawn
    :type color: tuple[int, int, int]
    """
    
    if color not in [(0, 255, 0), (0, 255, 255), (0, 0, 255)]:
        raise ValueError(f"Color '{color}' not supported")
    
    x, y, w, h = box
    
    top_left = (x, y)
    bottom_right = (x + w, y + h)
    
    rectangle(image, top_left, bottom_right, color, 3)
    
    return image