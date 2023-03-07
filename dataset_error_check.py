import os
import numpy as np
import cv2
import time

save_dir = "/insightface-master/parsing/dml_csr/datasets/labels/"
saved_list = os.listdir(save_dir)

check_dict = {}

for subject in saved_list:
    image = cv2.imread(save_dir + subject, cv2.IMREAD_GRAYSCALE)

    h, w = image.shape
    image = np.reshape(image, (1, -1))

    for i in range(len(image[0])):
        #check_dict[i] += 1
        if image[0][i] > 18:
            print(subject)
            break

