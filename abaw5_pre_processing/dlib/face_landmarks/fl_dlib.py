import os
import sys
import time
from os.path import dirname, abspath, join
from typing import Optional, Union, Tuple


import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

# works with python 3.6. see pypi dlib.



def get_landmarks():
    pass



def test_alignment(img_path: str):
    detector_backends = ["opencv", "ssd", "mtcnn", "retinaface"]


    # extract faces
    for detector_backend in detector_backends:
        try:
            face_objs = DeepFace.extract_faces(
                img_path=img_path, detector_backend=detector_backend,
                target_size=(256, 256)
            )

            for face_obj in face_objs:
                face = face_obj["face"]
                print(detector_backend, len(face_objs), face_obj['confidence'])
                print(type(face), face.shape)
                plt.imshow(face)
                plt.axis("off")
                plt.show()
                print("-----------")
        except:
            print(detector_backend, 'didnt find a face.')


if __name__ == "__main__":

    # path_img = join(root_dir, 'data/debug/input/test_0006.jpg')
    path_img = join(root_dir, 'data/debug/input/test_0038.jpg')
    # path_img = join(root_dir, 'data/debug/input/test_0049.jpg')
    # path_img = join(root_dir, 'data/debug/input/test_0067.jpg')
    test_alignment(path_img)


