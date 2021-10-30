# import tensorflow as tf
import tensorflow.compat.v1 as tf

import numpy as np
import src.facenet
from align import detect_face
import cv2
from scipy.spatial import distance

# some constants kept as default from facenet
minsize = 20
#threshold = [0.6, 0.7, 0.7]
threshold = [0.1, 0.1, 0.1]
factor = 0.709
margin = 44
input_image_size = 160

sess = tf.Session()
#Face-detector
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

#load Face-embedding model
src.facenet.load_model("./face_embedding_model/embedding.pb")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

def getFace(img):
    faces = []
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if not len(bounding_boxes) == 0:
        for face in bounding_boxes:
            if face[4] > 0.30:
                det = np.squeeze(face[0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                resized = cv2.resize(cropped, (input_image_size,input_image_size),interpolation=cv2.INTER_CUBIC)
                # cv2.imshow("img", resized)
                # cv2.waitKey(0)
                prewhitened = src.facenet.prewhiten(resized)
                faces.append({'face':resized,'rect':[bb[0],bb[1],bb[2],bb[3]],'embedding':getEmbedding(prewhitened)})
    return faces
def getEmbedding(resized):
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding

def compare2face(img1,img2):
    face1 = getFace(img1)
    face2 = getFace(img2)
    # for face in face1:
    #     print("Embeddings = "+str(face['embedding']))

    # for face in face2:
    #     print("Embeddings = "+str(face['embedding']))    

    if face1 and face2:
        # calculate Euclidean distance
        dist = distance.euclidean(face1[0]['embedding'],face2[0]['embedding'])
        #print(dist)
        return dist
    return -1
