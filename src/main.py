# native imports
import os
# local imports
from constants.hsv_boundaries import (
    LOWER_GREEN, LOWER_YELLOW, LOWER_RED_START, LOWER_RED_END,
    UPPER_GREEN, UPPER_YELLOW, UPPER_RED_START, UPPER_RED_END
)
from utils.arguments import parse_args
from utils.image_modifiers import applyMask
# non-native imports
import cv2 as cv
import numpy as np

def detectTrafficLights(args):
    """
    Main function to detect traffic lights and identify which signal it is representing
    
    :param args: Parser containing the arguments that will be used in the pipeline
    """
    input_dir = args.input_dir
    images = os.listdir(input_dir)

    for image_name in images:
        input_image_path = os.path.join(input_dir, image_name)

        original_img = cv.imread(input_image_path)
        hsv_image = cv.cvtColor(original_img, cv.COLOR_BGR2HSV)

        masked_image_green = applyMask(
            image=hsv_image,
            boundaries=(np.array(LOWER_GREEN), np.array(UPPER_GREEN))
        )

        masked_image_yellow = applyMask(
            image=hsv_image,
            boundaries=(np.array(LOWER_YELLOW), np.array(UPPER_YELLOW))
        )

        masked_image_red = cv.bitwise_or(
            applyMask(
                image=hsv_image,
                boundaries=(np.array(LOWER_RED_START), np.array(UPPER_RED_START))
            ),
            applyMask(
                image=hsv_image,
                boundaries=(np.array(LOWER_RED_END), np.array(UPPER_RED_END))
            )
        )

if __name__ == '__main__':
    detectTrafficLights(parse_args())