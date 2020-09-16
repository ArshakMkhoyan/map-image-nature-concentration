import base64

import requests
from skimage.color import rgb2lab, deltaE_cie76
from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)


def get_coordinates_ru_corner(img):
    for y, arr in enumerate(img):
        for x, val in enumerate(arr):
            if x == 0:
                continue
            if x > len(arr) // 2:
                break
            if val != arr[x - 1]:
                return [y, x]


def get_mask(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return np.logical_or(r > 245, g < 10, b < 10)


def match_image_by_color(image, blue_color, green_color, threshold=15):
    selected_image = rgb2lab(np.uint8(np.asarray([image])))
    blue_color = rgb2lab(np.uint8(np.asarray([[blue_color]])))
    green_color = rgb2lab(np.uint8(np.asarray([[green_color]])))

    mask_blue = deltaE_cie76(selected_image, blue_color) < threshold
    mask_green = deltaE_cie76(selected_image, green_color) < threshold

    mask = np.bitwise_or(mask_blue, mask_green)
    mask = mask.reshape(mask.shape[1:])

    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 1), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return np.around(mask.mean(), 3), mask * 255


@app.route("/nature_concentration/", methods=['POST'])
def get_nature_concentration():
    if request.json and 'image' in request.json:
        green_rgb = [201.0, 240.0, 187.0]
        blue_rgb = [162.0, 206.0, 255.0]
        red_rgb = [253, 0, 0]

        image_bytes = requests.get(request.json['image']).content
        image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Add red borders to make things easier
        image_rgb = cv2.copyMakeBorder(image_rgb, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=red_rgb)

        # Remove red borders
        binary = get_mask(image_rgb)
        ru = get_coordinates_ru_corner(binary)
        ld = get_coordinates_ru_corner(binary[::-1, ::-1])
        image_rgb = image_rgb[ru[0]:-ld[0], ru[1]:-ld[1], :]

        # Get image mask and nature concentration value
        nature_concentration_value, image_mask = match_image_by_color(image_rgb, blue_rgb, green_rgb)

        # Encode mask to base64
        _, buffer = cv2.imencode('.jpg', image_mask)
        jpg_as_text = base64.b64encode(buffer)

        # TMP. Get image encoded
        # _, tmp = cv2.imencode('.jpg', image_rgb)
        # tmp_as_text = base64.b64encode(tmp)
        # TMP

        return jsonify(nature_concentration_value=nature_concentration_value, image_mask=jpg_as_text.decode("utf-8"))
    else:
        return {'nature_concentration_value': None, 'image_mask': None}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=12000)
