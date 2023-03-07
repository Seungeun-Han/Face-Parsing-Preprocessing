import cv2
import os
import numpy as np
import random

def seperate():
    image_path = "D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/"  # CelebAMask-HQ 이미지 디렉터리
    txt = "D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping_png.txt"
    ca_train_path = "C:/Users/USER/Downloads/misf-main/misf-main/celebA/train/"
    ca_valid_path = "C:/Users/USER/Downloads/misf-main/misf-main/celebA/valid/"
    ca_test_path = "C:/Users/USER/Downloads/misf-main/misf-main/celebA/test/"
    hq_train_path = "D:/Dataset/CelebAMask-HQ/train/"
    if not os.path.exists(hq_train_path):
        os.makedirs(hq_train_path)
    hq_valid_path = "D:/Dataset/CelebAMask-HQ/valid/"
    if not os.path.exists(hq_valid_path):
        os.makedirs(hq_valid_path)
    hq_test_path = "D:/Dataset/CelebAMask-HQ/test/"
    if not os.path.exists(hq_test_path):
        os.makedirs(hq_test_path)

    #print("celebA", len(os.listdir(image_path)))

    f = open(txt, "r")
    lines = f.readlines()[1:]
    for line in lines:
        idx, orig_idx, orig_file = line.strip().split()
        print(idx, orig_idx, orig_file)
        image = cv2.imread(image_path + idx + ".jpg", cv2.IMREAD_COLOR)

        if orig_file in os.listdir(ca_train_path):
            #cv2.imwrite(hq_train_path + idx.zfill(5) + ".jpg", image)
            continue
        elif orig_file in os.listdir(ca_valid_path):
            #cv2.imwrite(hq_valid_path + idx.zfill(5) + ".jpg", image)
            continue
        elif orig_file in os.listdir(ca_test_path):
            cv2.imwrite(hq_test_path + idx.zfill(5) + ".jpg", image)
        else:
            print(idx.zfill(5) + ".jpg")
            break

    print("train", len(os.listdir(hq_train_path)))
    print("valid", len(os.listdir(hq_valid_path)))
    print("test", len(os.listdir(hq_test_path)))
    print("sum", len(os.listdir(hq_train_path))+len(os.listdir(hq_valid_path))+len(os.listdir(hq_test_path)))

    f.close()

if __name__ == "__main__":
    seperate()