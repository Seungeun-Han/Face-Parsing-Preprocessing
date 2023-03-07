import os
import cv2
import numpy as np

one_path = os.path.join(r'D:\Dataset\CelebAMask-HQ\CelebA-HQ_112\edges')
one_list = os.listdir(one_path)

two_path = os.path.join(r'D:\Dataset\CelebAMask-HQ\CelebA-HQ_112\edges_1')
two_list = os.listdir(two_path)

remove_list = [i for i in one_list if i in two_list]

print(len(remove_list))
for subject in remove_list:
    os.remove(os.path.join(one_path, subject))