import cv2
import os
import numpy as np
import random

def seperate():
    image_path = "D:/Dataset/CelebAMask-HQ-mask/CelebAMask-HQ-mask_256/mask_only/"
    ca_train_path = "D:/Dataset/CelebAMask-HQ/CelebA-HQ/train/images/"
    ca_valid_path = "D:/Dataset/CelebAMask-HQ/CelebA-HQ/valid/images/"
    ca_test_path = "D:/Dataset/CelebAMask-HQ/CelebA-HQ/test/images/"
    hq_train_path = "D:/Dataset/CelebAMask-HQ-mask/CelebAMask-HQ-mask_256/train/masks/"
    if not os.path.exists(hq_train_path):
        os.makedirs(hq_train_path)
    hq_valid_path = "D:/Dataset/CelebAMask-HQ-mask/CelebAMask-HQ-mask_256/valid/masks/"
    if not os.path.exists(hq_valid_path):
        os.makedirs(hq_valid_path)
    hq_test_path = "D:/Dataset/CelebAMask-HQ-mask/CelebAMask-HQ-mask_256/test/masks/"
    if not os.path.exists(hq_test_path):
        os.makedirs(hq_test_path)

    image_list = os.listdir(image_path)

    for images in image_list:
        # print(images)
        image = cv2.imread(image_path + images, cv2.IMREAD_GRAYSCALE)  # cv2.IMREAD_GRAYSCALE  / cv2.IMREAD_COLOR

        if images[:5] + ".jpg" in os.listdir(ca_train_path):
            cv2.imwrite(hq_train_path + images[:5] + "_kf94" + images[-4:], image)
        elif images[:5] + ".jpg" in os.listdir(ca_valid_path):
            cv2.imwrite(hq_valid_path + images[:5] + "_kf94" + images[-4:], image)
        elif images[:5] + ".jpg" in os.listdir(ca_test_path):
            cv2.imwrite(hq_test_path + images[:5] + "_kf94" + images[-4:], image)
        else:
            print(images)
            break

    print("train", len(os.listdir(hq_train_path)))
    print("valid", len(os.listdir(hq_valid_path)))
    print("test", len(os.listdir(hq_test_path)))
    print("sum", len(os.listdir(hq_train_path))+len(os.listdir(hq_valid_path))+len(os.listdir(hq_test_path)))


if __name__ == "__main__":
    seperate()