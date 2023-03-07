import os
import cv2
import time

save_dir = "D:/Dataset/CelebAMask-HQ/CelebA-HQ_Aug_flip+crop_bgr_473/train/labels/"  # edges, images, cat_edges, labels
saved_list = os.listdir(save_dir)

for subject in saved_list:
    image = cv2.imread(save_dir+subject, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    if h != 473 or w != 473:
        print(subject)
        # os.remove(save_dir+subject)