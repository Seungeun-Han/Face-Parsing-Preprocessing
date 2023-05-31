import os

import cv2

path = r"D:\Dataset\CelebAMask-HQ-mask\CelebAMask-HQ-maskRendering_256\test\labels/"
label_list = os.listdir(path)

save_path = r"D:\Dataset\CelebAMask-HQ-mask\CelebAMask-HQ-maskRendering_256\test\labels_x10/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in label_list:
    print(i)
    image = cv2.imread(path + i, cv2.IMREAD_GRAYSCALE)
    image *= 10


    cv2.imwrite(save_path + i, image)