# local imports
from constants.hsv_boundaries import (
    LOWER_GREEN, LOWER_YELLOW, LOWER_RED_START, LOWER_RED_END,
    UPPER_GREEN, UPPER_YELLOW, UPPER_RED_START, UPPER_RED_END
)
from utils.arguments import parse_args
from utils.image_modifiers import applyMask
# non-native imports
import cv2 as cv

def detectTrafficLights():
    img = cv.imread

if __name__ == '__main__':
    detectTrafficLights(parse_args())