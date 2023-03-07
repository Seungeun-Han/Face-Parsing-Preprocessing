import os
import cv2
import numpy as np

path = "D:/Dataset/CelebAMask-HQ/CelebA-HQ_Aug_flip+crop_bgr_473/valid/cat_edges/"
label_list = os.listdir(path)

# print(label_list)

for i in label_list:
    image = cv2.imread(path + i, cv2.IMREAD_GRAYSCALE)
    image = np.reshape(image, (1, -1))
    image = np.array(image[0])

    if np.all(image == 0):
        print(i)
        # os.remove(path + i)
