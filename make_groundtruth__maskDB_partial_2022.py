import os
import numpy as np
import cv2
import time

"""LABELS = {'bg': 0, 'skin': 1, 'nose': 2, 'eye_g': 3, 'l_eye': 4, 'r_eye': 5,
        'l_brow': 6, 'r_brow': 7, 'l_ear': 8, 'r_ear': 9, 'mouth': 10, 'u_lip': 11,
        'l_lip': 12, 'hair': 13, 'hat': 14, 'ear_r': 15, 'neck_l': 16, 'neck': 17, 'cloth': 18, 'mask': 19}"""

LABELS = {'bg': 0, 'skin': 1, 'hair': 2, 'l_ear': 3, 'r_ear': 4, 'eye_g': 5, 'hat': 6, 'mask': 8}

def make_groundTruth(label, seg, seg_name):
    """
    * 우선순위 중요함

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

    assert_list = ['skin', 'l_ear', 'r_ear', 'hair']

    if seg_name not in assert_list:
        label[0][seg_list] = LABELS[seg_name]#*10

    elif seg_name == 'skin':  # cond1
        skin_idx = [x for x in range(len(label[0])) if label[0][x] == 0]
        intersection = list(set(seg_list) & set(skin_idx))
        label[0][intersection] = LABELS[seg_name]# * 10
    elif seg_name == 'l_ear' or seg_name == 'r_ear':  # cond2, 10
        ear_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['hair']]  # *10
        intersection = list(set(seg_list) & set(ear_idx))
        label[0][intersection] = LABELS[seg_name]# * 10
    elif seg_name == 'hair':  # cond6, 9
        hair_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['eye_g']]  # *10
        intersection = list(set(seg_list) & set(hair_idx))
        label[0][intersection] = LABELS[seg_name] #* 10

    label = np.reshape(label, (h, -1))
    return label

im_path = os.path.join('D:/Dataset/ETRI_MaskDB/image/')
image_list = os.listdir(im_path)
#im = cv2.imread(im_path, cv2.IMREAD_COLOR)
# image_list = [x for x in image_list if "-00" in x]  # 750개
image_list = [x for x in image_list if '2022' in x]
# image_list = [x for x in image_list if 'IMG' in x]

parsing_anno_path = os.path.join('D:/Dataset/ETRI_MaskDB/label_split/')
# annotation_list = os.listdir(parsing_anno_path)
# annotation_list = [i for i in os.listdir(parsing_anno_path) if i[11:-4] in LABELS.keys()]
annotation_list = [i for i in os.listdir(parsing_anno_path) if i[16:-4] in LABELS.keys()]
# annotation_list = [i for i in os.listdir(parsing_anno_path) if i[9:-4] in LABELS.keys()]


save_dir = "D:/Dataset/ETRI_MaskDB/ETRI_MaskDB_partial_112/labels/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saved_list = os.listdir(save_dir)
image_list = [i for i in image_list if i[:-4] + ".png" not in saved_list]
print(len(image_list))

INPUT_SIZE = 112

for im_list in image_list:
    start = time.time()
    parent_img_name = im_list[:-4]
    print(parent_img_name)

    label = np.zeros((INPUT_SIZE, INPUT_SIZE))
    for idx, ann_list in enumerate(annotation_list):
        if parent_img_name in ann_list:
            annotation_path = parsing_anno_path + ann_list
            parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            parsing_anno = cv2.resize(parsing_anno, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_NEAREST)

            #print(ann_list[6:-4])
            # label = make_groundTruth(label, parsing_anno, ann_list[11:-4])
            label = make_groundTruth(label, parsing_anno, ann_list[16:-4])
            # label = make_groundTruth(label, parsing_anno, ann_list[9:-4])

    # label = cv2.resize(label, (473, 473), cv2.INTER_NEAREST)

    cv2.imwrite(save_dir + parent_img_name + ".png", label)
    print("time :", time.time() - start)

