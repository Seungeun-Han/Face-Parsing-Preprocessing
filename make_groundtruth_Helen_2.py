import os
import numpy as np
import cv2
import time

"""LABELS = {'bg': 0, 'skin': 1, 'l_brow': 2, 'r_brow': 3, 'l_eye': 4, 'r_eye': 5,
        'nose': 6, 'u_lip': 7, 'mouth': 8, 'l_lip': 9, 'hair': 10}"""
LABELS = {'00': 0, '01': 1, '02': 2, '03': 3, '04': 4, '05': 5,
        '06': 6, '07': 7, '08': 8, '09': 9, '10': 10}

def make_groundTruth(label, seg, seg_name):
    """
    * 우선순위 중요함
    - cond1: 스킨 < 눈, 눈썹, 입술, 입
    - cond2: 눈썹 < 머리카락
    - cond3: 눈 < 머리카락
    - cond4: 코 < 머리카락

    Args:
        label:
        seg:
        seg_name:

    Returns:

    """
    h, w = seg.shape
    label = np.reshape(label, (1, -1))
    seg = np.reshape(seg, (1, -1))

    seg_list = [i for i in range(len(seg[0])) if seg[0][i] >= 128]

    assert_list = ['01', '02', '03', '04', '05', '06']

    if seg_name not in assert_list:
        label[0][seg_list] = LABELS[seg_name]#*10

    elif seg_name == '01':  # cond1
        skin_idx = [x for x in range(len(label[0])) if label[0][x] == 0]
        intersection = list(set(seg_list) & set(skin_idx))
        label[0][intersection] = LABELS[seg_name]# * 10
    elif seg_name == '02' or seg_name == '03':  # cond2
        brow_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['10']]  # *10
        intersection = list(set(seg_list) & set(brow_idx))
        label[0][intersection] = LABELS[seg_name]# * 10
    elif seg_name == '04' or seg_name == '05':  # cond3
        eye_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['10']]  # *10
        intersection = list(set(seg_list) & set(eye_idx))
        label[0][intersection] = LABELS[seg_name] #* 10
    elif seg_name == '06':  # cond4
        hair_idx = [x for x in range(len(label[0])) if label[0][x] != LABELS['10']]  # *10
        intersection = list(set(seg_list) & set(hair_idx))
        label[0][intersection] = LABELS[seg_name] #* 10

    label = np.reshape(label, (h, -1))
    return label

im_path = os.path.join('D:/Dataset/Helen/test/images/')
image_list = os.listdir(im_path)

parsing_anno_path = os.path.join('D:/Dataset/SmithCVPR2013_dataset_original/labels/')

save_dir = "D:/Dataset/Helen/test/labels/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saved_list = os.listdir(save_dir)
image_list = [i for i in image_list if i[:-4] + ".png" not in saved_list]
print(len(image_list))

for im_list in image_list:
    start = time.time()
    parent_img_name = im_list[:-4]
    print(parent_img_name)

    image = cv2.imread(im_path+im_list, cv2.IMREAD_COLOR)
    # print(image.shape[0], image.shape[1])  # hieght, width

    label = np.zeros((image.shape[1], image.shape[0]))
    child_label_path = parsing_anno_path+parent_img_name
    label_list = os.listdir(child_label_path)
    # print(child_label_list)

    for child_label_name in label_list:
        parsing_anno = cv2.imread(os.path.join(child_label_path, child_label_name), cv2.IMREAD_GRAYSCALE)

        label = make_groundTruth(label, parsing_anno, child_label_name[-6:-4])

    cv2.imwrite(save_dir + parent_img_name + ".png", label)
    print("time :", time.time() - start)


