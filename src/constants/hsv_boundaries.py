# values selected after empirical testing and analysis performed in app/find_hsv_channel.py

LOWER_GREEN = [55, 107, 158]
UPPER_GREEN = [157, 255, 255]

LOWER_YELLOW = [10, 120, 80]
UPPER_YELLOW = [35, 255, 255]

# yellow needs two variables because its color is end/start of the hue scale
LOWER_RED_START = [0, 120, 80]
UPPER_RED_START = [10, 255, 255]
LOWER_RED_END = [160, 120, 80]
UPPER_RED_END = [179, 255, 255]