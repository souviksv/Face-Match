import numpy as np
import cv2



def visualize(image, distance):
    print(distance)
    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 0.8
    org = (70, 199) 
    # fontScale 
    fontScale = 1
    # Blue color in BGR 
    color = (0, 255, 0)
    red_colr = (0,0,255) 
    # Line thickness of 2 px 
    thickness = 2
    
    if distance <= threshold :
        cv2.putText(image, 'Same Person', org, font,fontScale, color, thickness, cv2.LINE_AA)    
    else :
        cv2.putText(image, 'Not the Same Person', org, font,fontScale, red_colr, thickness, cv2.LINE_AA)
    return image    