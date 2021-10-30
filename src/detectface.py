import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
from src.face_detector import FaceDetector

def detectFace(img,MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0'):

    #image_array = cv2.imread(img)
    resize_frame = cv2.resize(img, (640,480))
    rgb_image = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
    face_detector = FaceDetector(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0')  
    boxes = face_detector(rgb_image, score_threshold=0.5)

    if boxes is not None:
        for box in boxes:
            a,b,c,d = box
            xmin = int(a)
            ymin = int(b)
            xmax = int(c)
            ymax = int(d)

            h = ymax - ymin
            w = xmax - xmin

            im_w, im_h = rgb_image.shape[:2]

            xmin = int(max(xmin - (w * 0.35), 0))
            xmax = int(min(xmax + (w * 0.35), im_w))

            ymin = int(max(ymin - (h * 0.35), 0))
            ymax = int(min(ymax + (h * 0.35), im_h))

            cropped_face = rgb_image[xmin:xmax, ymin:ymax]

            w, h = cropped_face.shape[:2]
            crp_img = cropped_face[:,:,::-1]
            img = cv2.resize(crp_img, (224,224))
            return img