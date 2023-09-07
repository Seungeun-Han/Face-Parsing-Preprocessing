import os
import numpy as np
import cv2
import time


# LABELS = {'bg': 0, 'skin': 1, 'hair': 2, 'l_ear': 3, 'r_ear': 4, 'eye_g': 5, 'hat': 6, 'hand': 7}
LABELS = {'bg': 0, 'skin': 1, 'hair': 2, 'l_ear': 3, 'r_ear': 4, 'eye_g': 5, 'hat': 6, 'kf94': 8}

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
        label[0][seg_list] = LABELS[seg_name]
    elif seg_name == 'hair':
        skin_idx = [x for x in range(len(label[0])) if label[0][x] == 0]
        intersection = list(set(seg_list) & set(skin_idx))
        label[0][intersection] = LABELS[seg_name]
    elif seg_name == 'skin':
        skin_idx = [x for x in range(len(label[0])) if label[0][x] == 0]
        intersection = list(set(seg_list) & set(skin_idx))
        label[0][intersection] = LABELS[seg_name]
    elif seg_name == 'l_ear' or seg_name == 'r_ear':
        ear_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['hair']]  # *10
        intersection = list(set(seg_list) & set(ear_idx))
        label[0][intersection] = LABELS[seg_name]

    label = np.reshape(label, (h, -1))
    return label

im_path = os.path.join('D:/Dataset/CelebAMask-HQ-mask/images_1024/')
image_list = os.listdir(im_path)
#im = cv2.imread(im_path, cv2.IMREAD_COLOR)
# image_list = [x for x in image_list if int(x[:5]) < 10000]  # _ 조건 추가

parsing_anno_path = os.path.join('D:/Dataset/CelebAMask-HQ-mask/seg_1024png/')
annotation_list = [i for i in os.listdir(parsing_anno_path) if i[6:-4] in LABELS.keys()]
# print(len(annotation_list))  # 90401
annotation_name_list = [i[:5] for i in annotation_list]
annotation_name_list = list(set(annotation_name_list))

save_dir = "D:/Dataset/CelebAMask-HQ-mask/CelebAMask-HQ-maskRendering_Partial_256/labels/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saved_list = os.listdir(save_dir)
image_list = [i for i in image_list if i[:5] + ".png" not in saved_list and i[:-4] + ".png" not in saved_list]
print(len(image_list))

INPUT_SIZE = 256

for im_list in image_list:
    start = time.time()
    parent_img_name = im_list[:5]
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

    cv2.imwrite(save_dir + parent_img_name + "_kf94.png", label)
    print("time :", time.time() - start)


