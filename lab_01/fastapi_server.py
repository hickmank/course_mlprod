"""Adaptation from Lab 1 - part 1.

In this lab we setup a server for a trained ML model. 

"""

import os
import io
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import requests
import numpy as np
from IPython.display import Image, display

import uvicorn
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVE'] = '3'


# Some example images
image_files = [
    'apple.jpg',
    'clock.jpg',
    'oranges.jpg',
    'car.jpg'
    ]

# for image_file in image_files:
#     print(f'\nDisplaying image: {image_file}')
#     display(Image(filename=f"./images/{image_file}"))


# Now we create and draw boxes around objects in the images
dir_name = './images_with_boxes'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def detect_and_draw_box(filename, model='yolov3-tiny', confidence=0.5):
    """Detects common objects on an image and creates a new image with bounding boxes.

    Args:
        filename (str): Filename of the image.
        model (str): Either "yolov3" or "yolov3-tiny".
        confidence (float): Desired condidence level for bounding box.

    """

    # Images are stored under the './image' directory
    img_filepath = f'./images/{filename}'

    # Read the image into a numpy array
    img = cv2.imread(img_filepath)

    # Perform the object detection
    bbox, label, conf = cv.detect_common_objects(img, confidence=confidence, model=model)

    # Print current images filename
    print(f'==========================\nImage processed: {filename}\n')

    # Print detected objects with confidence level
    for l, c in zip(label, conf):
        print(f'Detected object: {l} with confidence level {c}\n')

    # Create a new image that includes the bounding boxes
    output_image = draw_bbox(img, bbox, label, conf)

    # Save the image
    cv2.imwrite(f'./images_with_boxes/{filename}', output_image)

    # Display the image with bounding boxes
    display(Image(f'./images_with_boxes/{filename}'))


# # Let's try out our function
# for image_file in image_files:
#     detect_and_draw_box(image_file)


# Now we set up the server instance

# Assign an instance of the FastAPI class to the variable 'app'.
# You will interact with your api using this instance.
app = FastAPI(title='Deploying an ML Model with FastAPI')

# List available models using Enum for convenience. This is useful when the
# options are pre-defined.
class Model(str, Enum):
    yolov3tiny = 'yolov3-tiny'
    yolov3 = 'yolov3'


# By using @app.get("/") you are allowing the GET method to work for the '/' endpoint.
@app.get('/')
def home():
    return 'Congratulations! Your API is working as expected.'


# This endpoint handles all the logic necessary for the object detection to work.
# It requires the deired model and the image in which to perform object detection.
@app.post('/predict')
def prediction(model: Model, file: UploadFile = File(...)):

    # 1. Validate input file
    filename = file.filename
    fileExtension = filename.split('.')[-1] in ('jpg', 'jpeg', 'png')
    if not fileExtension:
        raise HTTPException(status_code=415, detail='Unsupported file provided.')

    # 2. Transform raw image into cv2 image

    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())

    # Start the stream from the beginning
    image_stream.seek(0)

    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 3. Run object detection model

    # Run object detection
    bbox, label, conf = cv.detect_common_objects(image, model=model)

    # Create image that includes bounding boxxes and labels
    output_image = draw_bbox(image, bbox, label, conf)

    # Save it in a folder within the server
    cv2.imwrite(f'images_uploaded/{filename}', output_image)

    # 4. Stream the response back to the client

    # Open the saved image for reading in binary mode
    file_image = open(f'images_uploaded/{filename}', mode='rb')

    # return the image as a stream specifying media type
    return StreamingResponse(file_image, media_type='image/jpeg')


# Now you can spin up the server!!!

# Allows the server to be run in this interactive environment
nest_asyncio.apply()

# This is an alias for localhost which means this machine
host = '127.0.0.1'

# Spin up the server
uvicorn.run(app, host=host, port=8000, root_path='/')

# Now you can got to http://127.0.0.1:8000/docs and try out your model!!!
