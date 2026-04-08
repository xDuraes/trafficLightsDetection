from constants.hsv_constants import (
    IMAGE_PATH, 
    WINDOW_TRACKBAR_NAME, WINDOW_ORIGINAL_IMAGE_NAME, WINDOW_MASK_IMAGE_NAME,
    LOWER_HUE_NAME, LOWER_SATURATION_NAME, LOWER_VALUE_NAME,
    UPPER_HUE_NAME, UPPER_SATURATION_NAME, UPPER_VALUE_NAME
    )

import cv2 as cv
import numpy as np

def nothing(x):
    pass

# window for the trackbar
cv.namedWindow(WINDOW_TRACKBAR_NAME, cv.WINDOW_NORMAL)

# trackbars for lower and upper HSV
cv.createTrackbar(LOWER_HUE_NAME, WINDOW_TRACKBAR_NAME, 0, 179, nothing)
cv.createTrackbar(LOWER_SATURATION_NAME, WINDOW_TRACKBAR_NAME, 0, 255, nothing)
cv.createTrackbar(LOWER_VALUE_NAME, WINDOW_TRACKBAR_NAME, 0, 255, nothing)
cv.createTrackbar(UPPER_HUE_NAME, WINDOW_TRACKBAR_NAME, 179, 179, nothing)
cv.createTrackbar(UPPER_SATURATION_NAME, WINDOW_TRACKBAR_NAME, 255, 255, nothing)
cv.createTrackbar(UPPER_VALUE_NAME, WINDOW_TRACKBAR_NAME, 255, 255, nothing)

image = cv.imread(IMAGE_PATH)

while True:
    # frame = image # If using a static image
    # _, frame = cap.read() # If using webcam
    
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # current positions of the trackbars
    l_h = cv.getTrackbarPos(LOWER_HUE_NAME, WINDOW_TRACKBAR_NAME)
    l_s = cv.getTrackbarPos(LOWER_SATURATION_NAME, WINDOW_TRACKBAR_NAME)
    l_v = cv.getTrackbarPos(LOWER_VALUE_NAME, WINDOW_TRACKBAR_NAME)
    u_h = cv.getTrackbarPos(UPPER_HUE_NAME, WINDOW_TRACKBAR_NAME)
    u_s = cv.getTrackbarPos(UPPER_SATURATION_NAME, WINDOW_TRACKBAR_NAME)
    u_v = cv.getTrackbarPos(UPPER_VALUE_NAME, WINDOW_TRACKBAR_NAME)

    # bounds and mask
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])
    mask = cv.inRange(hsv, lower_bound, upper_bound)

    # results
    cv.imshow(WINDOW_ORIGINAL_IMAGE_NAME, image)
    cv.imshow(WINDOW_MASK_IMAGE_NAME, mask)

    key = cv.waitKey(1) & 0xFF

    # 'q' to exit
    if key == ord('q'):
        break

    # 'p' to print hsv values
    elif key == ord('p'):
        print(f"Lower: [{l_h}, {l_s}, {l_v}]")
        print(f"Upper: [{u_h}, {u_s}, {u_v}]")

cv.destroyAllWindows()