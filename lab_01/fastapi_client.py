"""Adaptation from Lab 1 - part 2.

In part 1 we setup a server for a trained ML model. In this lab we create a
client to query the server.

"""

import os
import io
import cv2
import requests
import numpy as np
from IPython.display import Image, display


# Define information for the server
base_url = 'http://localhost:8000'
endpoint = '/predict'
model = 'yolov3-tiny'

url_with_endpoint_no_params = base_url + endpoint
print(f'Full endpoint URL: {url_with_endpoint_no_params}')

# Set parameters by adding a "?" followed by parameter name and value
full_url = url_with_endpoint_no_params + '?model=' + model
print(f'Full URL with model specified: {full_url}')

# Endpoint expects a model name and image. Image is handled via the `requests` package
def response_from_server(url, image_file, verbose=True):
    """Makes a POST request to the server and returns the response.

    Args:
        url (str): URL that the request is sent to
        image_file (_io.BufferedReader): File to upload, should be an image.
        verbose (bool): True if the status of the response should be printed.

    Returns:
        requests.models.Response: Response from the server.

    """

    files = {'file': image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code
    if verbose:
        if status_code == 200:
            msg = "Everything went well!!"
        else:
            msg = "There was an error when handling the request."

        print(msg)

    return response


# Now query the server
with open("./images/clock2.jpg", "rb") as image_file:
    prediction = response_from_server(full_url, image_file)

# Now we'll parse the response from the server to get are boxes-around-objects image
dir_name = "./images_predicted"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def display_image_from_response(response):
    """Display image within server's response.

    Args:
        response (requests.models.Response): The response from the server
                                             after object detection.

    """

    image_stream = io.BytesIO(response.content)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    filename = "image_with_objects.jpeg"
    cv2.imwrite(f'./images_predicted/{filename}', image)
    display(Image(f'./images_predicted/{filename}'))


# Now display the response image
display_image_from_response(prediction)

# Let's try some other images
image_files = [
    'car2.jpg',
    'clock3.jpg',
    'apples.jpg'
    ]

for image_file in image_files:
    with open(f'./images/{image_file}', 'rb') as image_file:
        prediction = response_from_server(full_url, image_file, verbose=False)

    display_image_from_response(prediction)

