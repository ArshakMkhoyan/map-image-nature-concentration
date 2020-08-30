import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt

files = {'file': open('test_images/green.jpeg', 'rb')}
response = requests.post('http://0.0.0.0:5000/nature_concentration', files=files)
print(response.ok)

response = response.json()
print(response['nature_concentration_value'])

plt.imshow(response['image_mask'])
plt.imsave('test_images/output.jpg', np.array(response['image_mask']),  cmap='gray')
plt.show()

