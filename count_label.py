import os
import cv2
import numpy as np

path = "D:/Dataset/CelebAMask-HQ-mask/seg/"
label_list = os.listdir(path)
# label_list = [i for i in label_list if "nose" not in i or "lip" not in i or "mouth" not in i]

for i in label_list:
    image = cv2.imread(path + i, cv2.IMREAD_GRAYSCALE)
    cnt, labels = cv2.connectedComponents(image)
    print(cnt)
    """if i[:-9] in error_list:
        #
        print(i)
        os.remove(path + i)"""
