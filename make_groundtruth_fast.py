# Copyright (c) 한승은. All rights reserved.

import os
import numpy as np
import cv2
import time

IMAGE_PATH = 'D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/'
ANNOTATIOM_PATH = 'D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno_acc'
SAVE_PATH = "D:/Dataset/CelebAMask-HQ/CelebA-HQ_512/labels"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
INPUT_SIZE = 512

# Original
# LABELS = {'bg': 0, 'skin': 1, 'nose': 2, 'eye_g': 3, 'l_eye': 4, 'r_eye': 5,
#         'l_brow': 6, 'r_brow': 7, 'l_ear': 8, 'r_ear': 9, 'mouth': 10, 'u_lip': 11,
#         'l_lip': 12, 'hair': 13, 'hat': 14, 'ear_r': 15, 'neck_l': 16, 'neck': 17, 'cloth': 18}


# # Necklace Last
LABELS = {'bg': 0, 'skin': 1, 'nose': 2, 'eye_g': 3, 'l_eye': 4, 'r_eye': 5,
        'l_brow': 6, 'r_brow': 7, 'l_ear': 8, 'r_ear': 9, 'mouth': 10, 'u_lip': 11,
        'l_lip': 12, 'hair': 13, 'hat': 14, 'ear_r': 15, 'neck': 16, 'cloth': 17, 'neck_l': 18}



def make_groundTruth(label, seg, seg_name):
    """
    * 우선순위 중요함
    - cond1: 스킨 < 눈, 눈썹, 입술, 입, 안경
    - cond2: 귀 < 머리카락
    - cond3: 눈썹 < 머리카락
    - cond4: 눈 < 머리카락
    - cond5: 목 < 귀걸이
    - cond6: 머리카락 < 안경
    - cond7: 눈썹 < 모자
    - cond8: 코 < 머리카락
    - cond9: 머리카락 < 귀걸이
    - cond10: 귀 < 귀걸이
    - cond11: 귀걸이 < 스킨
    - cond12: 코 < 안경

    Args:
        label:
        seg:
        seg_name:

    Returns:

    """
    h, w = seg.shape
    label = np.reshape(label, (1, -1))
    seg = np.reshape(seg, (1, -1))

    seg_list = [i for i in range(len(seg[0])) if seg[0][i] == 255]

    assert_list = ['skin', 'l_ear', 'r_ear', 'l_brow', 'r_brow', 'neck', 'l_eye', 'r_eye', 'hair', 'nose']

    if seg_name not in assert_list:
        label[0][seg_list] = LABELS[seg_name]#*10

    elif seg_name == 'skin':  # cond1, 11
        skin_idx = [x for x in range(len(label[0])) if label[0][x] == 0 or label[0][x] == LABELS['ear_r']] #  or label[0][x] == LABELS['ear_r']
        intersection = list(set(seg_list) & set(skin_idx))
        label[0][intersection] = LABELS[seg_name]# * 10
    elif seg_name == 'l_ear' or seg_name == 'r_ear':  # cond2, 10
        ear_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['hair'] and label[0][x] != LABELS['ear_r']]  # *10
        intersection = list(set(seg_list) & set(ear_idx))
        label[0][intersection] = LABELS[seg_name]# * 10
    elif seg_name == 'l_brow' or seg_name == 'r_brow':  # cond3, 7
        brow_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['hair'] and label[0][x] != LABELS['hat']]  # *10
        intersection = list(set(seg_list) & set(brow_idx))
        label[0][intersection] = LABELS[seg_name]# * 10
    elif seg_name == 'l_eye' or seg_name == 'r_eye':  # cond4
        eye_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['hair']]  # *10
        intersection = list(set(seg_list) & set(eye_idx))
        label[0][intersection] = LABELS[seg_name] #* 10
    elif seg_name == 'neck':  # cond5
        neck_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['ear_r']]  # *10
        intersection = list(set(seg_list) & set(neck_idx))
        label[0][intersection] = LABELS[seg_name] #* 10
    elif seg_name == 'hair':  # cond6, 9
        hair_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['eye_g'] and label[0][x] != LABELS['ear_r']]  # *10
        intersection = list(set(seg_list) & set(hair_idx))
        label[0][intersection] = LABELS[seg_name] #* 10
    elif seg_name == 'nose':  # cond8
        hair_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['hair'] and label[0][x] != LABELS['eye_g']]  # *10
        intersection = list(set(seg_list) & set(hair_idx))
        label[0][intersection] = LABELS[seg_name] #* 10

    label = np.reshape(label, (h, -1))
    return label

# Image List
image_list = os.listdir(IMAGE_PATH)
# 일부만 만들고 싶을 때는 아래 코드 사용
# image_list = [x for x in image_list if int(x[:-4]) < 10000] #  >= 10000 and int(x[:-4])

# Annotation List
annotation_list = os.listdir(ANNOTATIOM_PATH)
annotation_name_list = [i[:5] for i in annotation_list]  # CelebAMask-HQ Only
annotation_name_list = list(set(annotation_name_list))

# If labels have already been created, exclude them from the task.
saved_list = os.listdir(SAVE_PATH)
image_list = [i for i in image_list if i[:-4].zfill(5) + ".png" not in saved_list]
print(len(image_list))


for im_list in image_list:
    start = time.time()
    parent_img_name = im_list[:-4].zfill(5)  # 0.jpg --> 00000.jpg
    print(parent_img_name)

    label = np.zeros((INPUT_SIZE, INPUT_SIZE))

    if parent_img_name in annotation_name_list:
        part_list = [i for i in annotation_list if parent_img_name in i]
        print(part_list)
        for p in part_list:
            annotation_path = os.path.join(ANNOTATIOM_PATH, p)

            parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            parsing_anno = cv2.resize(parsing_anno, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_NEAREST)

            label = make_groundTruth(label, parsing_anno, p[6:-4])  # CelebAMask-HQ Only


    cv2.imwrite(os.path.join(SAVE_PATH, parent_img_name + ".png"), label)
    print("time :", time.time() - start)


