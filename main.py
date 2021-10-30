from numpy.lib.ufunclike import _dispatcher
from src.passimage import _get_result
import numpy as np
from src.visualize import visualize
import cv2

def main():
    image1_path = "./data/dipika.jpg"
    image2_path = "./data/aishwarya_c.jpg"
    model_dir = "./model/model.pb"

    detect_face = _get_result(image1_path, image2_path, model_dir)
    distence = detect_face[0]

    detect_face1 = detect_face[1]
    detect_face2 = detect_face[2]

    concat = np.concatenate((detect_face1, detect_face2), axis=1)

    visual_image = visualize(concat, distence)

    cv2.imshow("Compared_face_image", visual_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()    

