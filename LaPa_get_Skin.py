# Copyright (c) 한승은. All rights reserved.

import os
import cv2
import numpy as np

path = r"D:\Dataset\LaPa\train\labels/"
label_list = os.listdir(path)
s_path = r"D:\Dataset\face_occlusion_dataset\LaPa-WO-Train_img_Train_mask/"
if not os.path.exists(s_path):
    os.makedirs(s_path)

for i in label_list:
    image = cv2.imread(path + i, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    # print(i, image.shape)
    image = np.reshape(image, (1, -1))
    image = np.array(image[0])

    skin = np.zeros_like(image)

    if np.any(image > 1):
        print(i)
        list = [i for i, v in enumerate(image) if v == 1]
        skin[list] = 255
        skin = np.reshape(skin, (h, -1))
        cv2.imwrite(s_path+i, skin)
