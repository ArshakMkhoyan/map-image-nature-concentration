import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt


def test(inp):
    # files = {'file': open(inp, 'rb')}
    # response = requests.post('http://0.0.0.0:5000/nature_concentration', files=files)
    response = requests.post('http://0.0.0.0:5000/nature_concentration', json=inp)
    print(response.ok)

    response = response.json()
    print(response['nature_concentration_value'])

    plt.imshow(response['image_mask'])
    plt.imsave('test_images/output_new.jpg', np.array(response['image_mask']), cmap='gray')
    plt.show()


# image_bytes = request.files['file'].read()
# image = np.array(Image.open(io.BytesIO(image_bytes)))

# link = 'test_images/green.jpeg'
# image_bytes = open(request.json['image'], 'rb').read()

link = 'https://maps.googleapis.com/maps/api/staticmap?center=54.7829996,' \
       '%209.4562149&zoom=8&size=640x400&maptype=roadmap&markers=color:red%7C54.7829996,' \
       '%209.4562149&key=AIzaSyCf6NDAXlZ5E1PdqJrAqCqhxEOQ7vYhDWI'
inp = {'image': link}
test(inp)
