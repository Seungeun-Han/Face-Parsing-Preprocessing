import os
import cv2
import numpy as np

path = r"D:\Dataset\ETRI_MaskDB\ETRI_MaskDB_473\edges/"
label_list = os.listdir(path)
s_path = r"D:\Dataset\ETRI_MaskDB\ETRI_MaskDB_473\edges_1/"
if not os.path.exists(s_path):
    os.makedirs(s_path)

for i in label_list:
    image = cv2.imread(path + i, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    # print(i, image.shape)
    image = np.reshape(image, (1, -1))
    image = np.array(image[0])

    if np.any(image > 1):
        print(i)
        list = [i for i, v in enumerate(image) if v>1]
        image[list] = 1
        image = np.reshape(image, (h, -1))
        cv2.imwrite(s_path+i, image)
