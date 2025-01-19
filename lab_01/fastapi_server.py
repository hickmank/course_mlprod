"""Adaptation from Lab 1 - part 1.

In this lab we setup a server for a trained ML model. 

"""

import os
import io
import cv2
import requests
import numpy as np
from IPython.display import Image, display


# Some example images
image_files = [
    'apple.jpg',
    'clock.jpg',
    'oranges.jpg',
    'car.jpg'
    ]

for image_file in image_files:
    print(f'\nDisplaying image: {image_file}')
    display(Image(filename=f"./images/{image_file}"))


