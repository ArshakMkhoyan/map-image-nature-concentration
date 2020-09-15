import base64

import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt


def test(inp):
    # server = 'http://52.30.27.231:12000'
    server = 'http://0.0.0.0:12000'
    response = requests.post(server + '/nature_concentration', json=inp)
    print(response.ok)

    response = response.json()
    print(response['nature_concentration_value'])

    mask = response['image_mask']
    jpg_original = base64.b64decode(mask)
    image_grey = cv2.imdecode(np.frombuffer(jpg_original, np.uint8), -1)

    tmp_as_text = response['tmp_as_text']
    jpg_original = base64.b64decode(tmp_as_text)
    image_orig = cv2.imdecode(np.frombuffer(jpg_original, np.uint8), -1)

    plt.imshow(image_orig)
    # plt.imshow(image_grey)
    plt.imsave('test_images/output_new.jpg', image_grey, cmap='gray')
    plt.show()


# image_bytes = request.files['file'].read()
# image = np.array(Image.open(io.BytesIO(image_bytes)))

# link = 'test_images/green.jpeg'
# image_bytes = open(request.json['image'], 'rb').read()

# link_red = 'https://maps.googleapis.com/maps/api/staticmap?center=49.630672,' \
#            '%209.672367&zoom=6&size=1000x1000&maptype=roadmap&path=color:0xfe0000fe|fillcolor:0xfe0000fe|weight:1|85,' \
#            '180|85,90|85,0|85,-90|85,-180|0,-180|-85,-180|-85,-90|-85,-0|-85,90|-85,180|0,180|85,180|51.07788,' \
#            '4.19369|51.07788,15.151044|48.183464,15.151044|48.183464,4.19369|51.07788,' \
#            '4.19369&key=AIzaSyCf6NDAXlZ5E1PdqJrAqCqhxEOQ7vYhDWI'

link_red = 'https://maps.googleapis.com/maps/api/staticmap?center=53.577566149999996,%209.88381665&zoom=14&size=1000x1000&maptype=roadmap&path=color:0xfe0000fe%7Cfillcolor:0xfe0000fe%7Cweight:1%7C85,180%7C85,90%7C85,0%7C85,-90%7C85,-180%7C0,-180%7C-85,-180%7C-85,-90%7C-85,-0%7C-85,90%7C-85,180%7C0,180%7C85,180%7C53.59430665,9.84388335%7C53.59430665,9.92374995%7C53.56082565,9.92374995%7C53.56082565,9.84388335%7C53.59430665,9.84388335&key=AIzaSyCf6NDAXlZ5E1PdqJrAqCqhxEOQ7vYhDWI'

# link_red = 'https://maps.googleapis.com/maps/api/staticmap?center=54.7883333,9.4434716&zoom=14&size=1000x1000&maptype=roadmap&path=color:0xfe0000fe%7Cfillcolor:0xfe0000fe%7Cweight:1%7C85,180%7C85,90%7C85,0%7C85,-90%7C85,-180%7C0,-180%7C-85,-180%7C-85,-90%7C-85,-0%7C-85,90%7C-85,180%7C0,180%7C85,180%7C54.7897117,9.4415616%7C54.7897117,9.4453816%7C54.7869549,9.4453816%7C54.7869549,9.4415616%7C54.7897117,9.4415616&key=AIzaSyCf6NDAXlZ5E1PdqJrAqCqhxEOQ7vYhDWI'
# link_red = 'https://maps.googleapis.com/maps/api/staticmap?center=54.7883333,9.4434716&zoom=17&size=1000x1000&maptype=roadmap&path=color:0xfe0000fe%7Cfillcolor:0xfe0000fe%7Cweight:1%7C85,180%7C85,90%7C85,0%7C85,-90%7C85,-180%7C0,-180%7C-85,-180%7C-85,-90%7C-85,-0%7C-85,90%7C-85,180%7C0,180%7C85,180%7C54.7897117,9.4415616%7C54.7897117,9.4453816%7C54.7869549,9.4453816%7C54.7869549,9.4415616%7C54.7897117,9.4415616&key=AIzaSyCf6NDAXlZ5E1PdqJrAqCqhxEOQ7vYhDWI'

inp = {'image': link_red}
test(inp)
