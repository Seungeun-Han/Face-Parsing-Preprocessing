# Copyright (c) 한승은. All rights reserved.

import os
import cv2
import numpy as np
import time

# original_path = "D:/Dataset/CelebAMask-HQ/noNecklace_CelebAMask-HQ_256/test/images"
# original_list = os.listdir(original_path)
#
#
# remove_path = "D:/Dataset/CelebAMask-HQ/necklace_CelebAMask-HQ_256/test/images"
# remove_list = os.listdir(remove_path)
#
#
# print(len(list(set(original_list) - set(remove_list))))
#
# for i in original_list:
#     if i in remove_list:
#         # image = cv2.imread(path + i, cv2.IMREAD_GRAYSCALE)
#         print(i)
#         os.remove(os.path.join(original_path,i))

def from_imagefolder_to_savefolder():
    path = "D:/Dataset/CelebAMask-HQ-mask/seg/"
    label_list = os.listdir(path)

    i_path = "D:/Dataset/CelebAMask-HQ_v2/CelebAMask-HQ_v2/CelebA-HQ-img_full_masked/"
    image_list = os.listdir(i_path)

    s_path = "D:/Dataset/CelebAMask-HQ-mask/images/"

    for i in label_list:
        if i in image_list:
            print(i)
            image = cv2.imread(i_path + i, cv2.IMREAD_COLOR)
            cv2.imwrite(s_path+i, image)


def from_celebLabelFolder_to_maskFolder():
    anno_path = "D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno_acc/"
    anno_list = os.listdir(anno_path)
    anno_list = [i for i in anno_list if "nose" not in i and "lip" not in i and "mouth" not in i]
    # anno_name_list = [i[:5] for i in anno_list]

    s_path = "D:/Dataset/CelebAMask-HQ-mask/seg_1024png/"
    save_list = os.listdir(s_path)
    save_name_list = [i[:5] for i in save_list]

    for i in anno_list:
        if i[:5] in save_name_list:
            print(i)
            image = cv2.imread(anno_path + i, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(s_path + i, image)

def from_celebLabelFolder_to_per_subject_folder():
    anno_path = "D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno_acc/"
    anno_list = os.listdir(anno_path)
    annotation_name_list = [i[:5] for i in anno_list]
    annotation_name_list = list(set(annotation_name_list))

    s_path = "D:/Dataset/CelebAMask-HQ/CelebAMask-HQ_perSubject"

    for i in annotation_name_list:
        to_save_path = os.path.join(s_path, i)
        if not os.path.exists(to_save_path):
            os.makedirs(to_save_path)
        to_save_list = [l for l in anno_list if i in l]
        print(to_save_list)

        for img in to_save_list:
            image = cv2.imread(os.path.join(anno_path, img), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(to_save_path, img), image)

def change_name_to_5blanks():
    path = "D:/Dataset/CelebAMask-HQ-mask/images/"
    list = os.listdir(path)

    s_path = "D:/Dataset/CelebAMask-HQ-mask/images_5blanks/"
    if not os.path.exists(s_path):
        os.makedirs(s_path)

    for i in list:
        image = cv2.imread(path + i, cv2.IMREAD_COLOR)
        print(i[:-9].zfill(5) + i[-9:])
        cv2.imwrite(s_path + i[:-9].zfill(5) + i[-9:], image)

def change_name_to_5blanks_and_toPng():
    path = "D:/Dataset/CelebAMask-HQ-mask/seg_1024jpg/"
    list = os.listdir(path)

    s_path = "D:/Dataset/CelebAMask-HQ-mask/seg_1024png/"
    if not os.path.exists(s_path):
        os.makedirs(s_path)

    for i in list:
        image = cv2.imread(path + i, cv2.IMREAD_GRAYSCALE)
        print(i[:-9].zfill(5) + i[-9:-3] + 'png')
        cv2.imwrite(s_path + i[:-9].zfill(5) + i[-9:-3] + 'png', image)

def opening_or_erode_or_closing():
    path = r'D:\Dataset\230703_tagging_samples\intern02_tagging_results\Cap/'
    list = os.listdir(path)
    # list = [i for i in list if "kf94" in i]

    s_path = r"D:\Dataset\230703_tagging_samples\CCTV_DB_Dataset\annotations/"
    if not os.path.exists(s_path):
        os.makedirs(s_path)

    kernel = np.ones((3, 3), np.uint8)

    for l in list:
        sub_path = os.path.join(path, l)
        sub_list = os.listdir(sub_path)
        for i in sub_list:
            # if ".jpg" in i:
            #     print(sub_path)
            print(i)
            image = cv2.imread(os.path.join(sub_path, i), cv2.IMREAD_GRAYSCALE)  # IMREAD_COLOR
            # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            # image = cv2.erode(image, kernel)
            # image *= 255

            cv2.imwrite(os.path.join(s_path, i), image)

def labeling():
    path = 'D:/Dataset/CelebAMask-HQ-mask/seg_1024png/'
    list = os.listdir(path)
    list = [i for i in list if "kf94" in i]

    s_path = "D:/Dataset/CelebAMask-HQ-mask/seg_1024png/"
    if not os.path.exists(s_path):
        os.makedirs(s_path)

    for i in list:
        start = time.time()
        # print(i)
        image = cv2.imread(path + i, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        count, map, stats, centroid = cv2.connectedComponentsWithStats(image)


        if stats[-1][-1] < 100:
            print(i, count, stats[-1][-1])
            image = np.reshape(image, (1, -1))
            map = np.reshape(map, (1, -1))

            label_list = [i for i, v in enumerate(map[0]) if v == count-1]

            image[0][label_list] = 0

            image = np.reshape(image, (h, -1))
            cv2.imwrite(s_path+i, image)
            # print("time :", time.time() - start)


opening_or_erode_or_closing()
