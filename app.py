from skimage.color import rgb2lab, deltaE_cie76
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import io

app = Flask(__name__)


def match_image_by_color(image, blue_color, green_color, threshold=10):
    selected_image = rgb2lab(np.uint8(np.asarray([image])))
    blue_color = rgb2lab(np.uint8(np.asarray([[blue_color]])))
    green_color = rgb2lab(np.uint8(np.asarray([[green_color]])))

    mask_blue = deltaE_cie76(selected_image, blue_color) < threshold
    mask_green = deltaE_cie76(selected_image, green_color) < threshold

    mask = np.bitwise_or(mask_blue, mask_green)
    mask = mask.reshape(mask.shape[1:])

    mask = mask.astype(np.uint8)
    kernel = np.ones((9, 1), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return np.around(mask.mean(), 3), mask


@app.route("/nature_concentration/", methods=['POST'])
def get_nature_concentration():
    if request.files:
        green_rgb = [201.0, 240.0, 187.0]
        blue_rgb = [162.0, 206.0, 255.0]

        image_bytes = request.files['file'].read()
        image = np.array(Image.open(io.BytesIO(image_bytes)))

        nature_concentration_value, image_mask = match_image_by_color(image, blue_rgb, green_rgb)

        print(image_mask.mean())
        # plt.imshow(image_mask, cmap='gray')
        # plt.savefig('DR.png')
        # plt.show()
        # im = Image.fromarray(image_mask)
        # im.save('test_images/output.jpeg')
        # cv2.imwrite('test_images/output.jpg', image_mask)
        return jsonify(nature_concentration_value=nature_concentration_value, image_mask=image_mask.tolist())
    else:
        return None


if __name__ == "__main__":
    app.run(host='0.0.0.0')
