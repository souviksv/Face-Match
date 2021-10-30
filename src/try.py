from __future__ import print_function
import requests
import json
import cv2
import numpy as np
from zipfile import ZipFile
addr = 'http://localhost:5001'
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}



def image(img1,img2):
    return img1, img2
    


# encode image as jpeg
def main():

    # img1 = cv2.imread('/media/souvik/Data/DocFace/doc_face/data/IMG_20191211_163410.jpg')
    # img2 = cv2.imread('/media/souvik/Data/DocFace/Flask/data/IMG_20191211_164516.jpg')

    with ZipFile('sample2.zip', 'w') as zipObj2:
        
        zipObj2.write('/media/souvik/Data/DocFace/doc_face/data/IMG_20191211_163410.jpg')
        zipObj2.write('/media/souvik/Data/DocFace/Flask/data/IMG_20191211_164516.jpg')
        

    # image_s = image(img1,img2)
    # ary_img = np.array(image_s)
    # # print(type(img1))
    # print(type(ary_img))
   
   # _, img_encoded = cv2.imencode('.jpg', ary_img)
    
    #print(img_encoded)
    # send http request with image and receive response
    response = requests.post(test_url, data=zipObj2, headers=headers)
    
    print(type(response))
    # decode response
    #print(json.loads(response.text))

    # expected output: {u'message': u'image received. size=124x124'}
if __name__=='__main__':
    main()        