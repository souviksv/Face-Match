import cv2
from src.face_match import compare2face
from src.detectface import detectFace


def _get_result(img1,img2,MODEL_PATH):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    img1 = detectFace(img1, MODEL_PATH)
    # cv2.imshow('image1', img1)
    # cv2.waitKey(1)

    img2 = detectFace(img2,MODEL_PATH)
    # cv2.imshow('Image2', img2)
    # cv2.waitKey(0)
    
    distance = 20
    if img1 is not None and img2 is not None:
        distance = compare2face(img1,img2)
    return distance, img1, img2
   
    # distance = compare2face(img1,img2)
    # return distance, img1, img2