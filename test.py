import base64

import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt


def test(inp):
    # files = {'file': open(inp, 'rb')}
    # response = requests.post('http://0.0.0.0:5000/nature_concentration', files=files)
    # server = 'http://172.31.32:5000'
    server = 'http://0.0.0.0:5000'
    response = requests.post(server + '/nature_concentration', json=inp)
    print(response.ok)

    response = response.json()
    print(response['nature_concentration_value'])

    mask = response['image_mask']
    jpg_original = base64.b64decode(mask)
    image_grey = cv2.imdecode(np.frombuffer(jpg_original, np.uint8), -1)
    plt.imshow(image_grey)
    plt.imsave('test_images/output_new.jpg', image_grey, cmap='gray')
    plt.show()


# image_bytes = request.files['file'].read()
# image = np.array(Image.open(io.BytesIO(image_bytes)))

# link = 'test_images/green.jpeg'
# image_bytes = open(request.json['image'], 'rb').read()

# link = 'https://maps.googleapis.com/maps/api/staticmap?center=54.7829996,' \
#        '%209.4562149&zoom=8&size=640x400&maptype=roadmap&markers=color:red%7C54.7829996,' \
#        '%209.4562149&key=AIzaSyCf6NDAXlZ5E1PdqJrAqCqhxEOQ7vYhDWI'
link_red = 'https://maps.googleapis.com/maps/api/staticmap?center=49.630672,' \
           '%209.672367&zoom=6&size=1000x1000&maptype=roadmap&path=color:0xfe0000fe|fillcolor:0xfe0000fe|weight:1|85,' \
           '180|85,90|85,0|85,-90|85,-180|0,-180|-85,-180|-85,-90|-85,-0|-85,90|-85,180|0,180|85,180|51.07788,' \
           '4.19369|51.07788,15.151044|48.183464,15.151044|48.183464,4.19369|51.07788,' \
           '4.19369&key=AIzaSyCf6NDAXlZ5E1PdqJrAqCqhxEOQ7vYhDWI '
inp = {'image': link_red}
test(inp)
