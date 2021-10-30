from __future__ import print_function
import requests
import cv2
import time 
import json

start = time.time()

addr = 'https://docface.azurewebsites.net'
test_url = addr + '/compare_faces'

img1 = '/media/souvik/Data/DocFace/Flask/data/jyoti.jpeg'
img2 = '/media/souvik/Data/DocFace/Flask/data/jyoti_id.jpeg'

# encode image as jpeg
files = [('image1', open(img1, 'rb')), ('image2', open(img2, 'rb'))]

# send http request with image and receive response
response = requests.post(test_url, files=files)
decode_response = json.loads(response.text)

end = time.time()
print(f"\nTime to read: {round(end-start,2)} seconds.")

# decode response
print(decode_response)
