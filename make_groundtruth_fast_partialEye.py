import os
import numpy as np
import cv2
import time


LABELS = {'bg': 0, 'skin': 1, 'hair': 2, 'l_eye': 3, 'r_eye': 4, 'eye_g': 5}


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

    assert_list = ['skin', 'l_eye', 'r_eye', 'hair']

    if seg_name not in assert_list:
        label[0][seg_list] = LABELS[seg_name]#*10

    elif seg_name == 'skin':  # cond1, 11
        skin_idx = [x for x in range(len(label[0])) if label[0][x] == 0]
        intersection = list(set(seg_list) & set(skin_idx))
        label[0][intersection] = LABELS[seg_name]# * 10
    elif seg_name == 'l_eye' or seg_name == 'r_eye':  # cond4
        eye_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['hair']]  # *10
        intersection = list(set(seg_list) & set(eye_idx))
        label[0][intersection] = LABELS[seg_name]  # * 10
    elif seg_name == 'hair':  # cond6, 9
        hair_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['eye_g']]  # *10
        intersection = list(set(seg_list) & set(hair_idx))
        label[0][intersection] = LABELS[seg_name] #* 10

    label = np.reshape(label, (h, -1))
    return label

im_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/')
image_list = os.listdir(im_path)
#im = cv2.imread(im_path, cv2.IMREAD_COLOR)
image_list = [x for x in image_list if int(x[:-4]) < 10000] #  >= 10000 and int(x[:-4])

parsing_anno_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno_acc/')
# annotation_list = os.listdir(parsing_anno_path)
annotation_list = [i for i in os.listdir(parsing_anno_path) if i[6:-4] in LABELS.keys()]
annotation_name_list = [i[:5] for i in annotation_list]
annotation_name_list = list(set(annotation_name_list))

save_dir = "D:/Dataset/CelebAMask-HQ/CelebA-HQ_partial_Eye_448/labels/" #fast_labels
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saved_list = os.listdir(save_dir)
image_list = [i for i in image_list if i[:-4].zfill(5) + ".png" not in saved_list]
print(len(image_list))

INPUT_SIZE = 448

for im_list in image_list:
    start = time.time()
    parent_img_name = im_list[:-4].zfill(5)
    print(parent_img_name)

    label = np.zeros((INPUT_SIZE, INPUT_SIZE))

    if parent_img_name in annotation_name_list:
        part_list = [i for i in annotation_list if parent_img_name in i]
        print(part_list)
        for p in part_list:
            annotation_path = parsing_anno_path + p

            parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            parsing_anno = cv2.resize(parsing_anno, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_NEAREST)

            label = make_groundTruth(label, parsing_anno, p[6:-4])

    cv2.imwrite(save_dir + parent_img_name + ".png", label)
    print("time :", time.time() - start)


