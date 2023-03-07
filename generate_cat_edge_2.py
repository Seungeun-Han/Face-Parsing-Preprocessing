import os
import numpy as np
import cv2
import time

LABELS = {'bg': 0, 'skin': 1, 'nose': 2, 'eye_g': 3, 'l_eye': 4, 'r_eye': 5,
        'l_brow': 6, 'r_brow': 7, 'l_ear': 8, 'r_ear': 9, 'mouth': 10, 'u_lip': 11,
        'l_lip': 12, 'hair': 13, 'hat': 14, 'ear_r': 15, 'neck_l': 16, 'neck': 17, 'cloth': 18}


def generate_cat_edge(seg, seg_name, edge_width=3):
    h, w = seg.shape
    edge = np.zeros(seg.shape)

    """for i in range(h-1):  # 0열
        if seg[i][0] != 0 and seg[i][1] == 0:  # right
            edge[i][0] = LABELS[seg_name]
        elif seg[i][0] != 0 and seg[i - 1][0] == 0:  # up
            edge[i][0] = LABELS[seg_name]
        elif seg[i][0] != 0 and seg[i + 1][0] == 0:  # down
            edge[i][0] = LABELS[seg_name]

    for i in range(h-1):  # 마지막열
        if seg[i][w-1] != 0 and seg[i][w-2] == 0:  # left
            edge[i][w-1] = LABELS[seg_name]  # * 10
        elif seg[i][w-1] != 0 and seg[i - 1][w-1] == 0:  # up
            edge[i][w-1] = LABELS[seg_name]  # * 10
        elif seg[i][w-1] != 0 and seg[i + 1][w-1] == 0:  # down
            edge[i][w-1] = LABELS[seg_name]  # * 10

    for j in range(1, w-1):  # 0행
        if seg[0][j] != 0 and seg[0][j - 1] == 0:  # left
            edge[0][j] = LABELS[seg_name]  # * 10
        elif seg[0][j] != 0 and seg[0][j + 1] == 0:  # right
            edge[0][j] = LABELS[seg_name]  # * 10
        elif seg[0][j] != 0 and seg[1][j] == 0:  # down
            edge[0][j] = LABELS[seg_name]  # * 10

    for j in range(1, w-1):  # 마지막행
        if seg[h-1][j] != 0 and seg[h-1][j - 1] == 0:  # left
            edge[h-1][j] = LABELS[seg_name]  # * 10
        elif seg[h-1][j] != 0 and seg[h-1][j + 1] == 0:  # right
            edge[h-1][j] = LABELS[seg_name]   # * 10
        elif seg[h-1][j] != 0 and seg[h-2][j] == 0:  # up
            edge[h-1][j] = LABELS[seg_name]  # * 10

    for i in range(1, h-1):
        for j in range(1, w-1):
            if seg[i][j] != 0 and seg[i][j - 1] == 0:  # left
                edge[i][j] = LABELS[seg_name]  # * 10
            elif seg[i][j]!=0 and seg[i][j+1]==0:  # right
                edge[i][j] = LABELS[seg_name]  # * 10
            elif seg[i][j] != 0 and seg[i-1][j] == 0:  # up
                edge[i][j] = LABELS[seg_name]  # * 10
            elif seg[i][j] != 0 and seg[i+1][j] == 0:  # down
                edge[i][j] = LABELS[seg_name]  # * 10"""
    # left
    for i in range(h):
        for j in range(1, w):
            if seg[i][j] != 0 and seg[i][j - 1] == 0:
                edge[i][j] = LABELS[seg_name]

    # right
    for i in range(h):
        for j in range(w - 1):
            if seg[i][j] != 0 and seg[i][j + 1] == 0:
                edge[i][j] = LABELS[seg_name]

    # up
    for i in range(1, h):
        for j in range(w):
            if seg[i][j] != 0 and seg[i - 1][j] == 0:
                edge[i][j] = LABELS[seg_name]

    # down
    for i in range(h - 1):
        for j in range(w):
            if seg[i][j] != 0 and seg[i + 1][j] == 0:
                edge[i][j] = LABELS[seg_name]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)

    return edge

im_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/')
image_list = os.listdir(im_path)
#im = cv2.imread(im_path, cv2.IMREAD_COLOR)
image_list = [x for x in image_list if int(x[:-4]) >= 2500 and int(x[:-4]) < 5000]

parsing_anno_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno_acc/')
annotation_list = os.listdir(parsing_anno_path)

save_dir = "D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/cat_edge/"
for im_list in image_list:
    start = time.time()
    parent_img_name = im_list[:-4].zfill(5)
    print(parent_img_name)

    edge = np.zeros((512, 512))
    for idx, ann_list in enumerate(annotation_list):
        if parent_img_name in ann_list:
            annotation_path = parsing_anno_path + ann_list
            parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            #print(ann_list[6:-4])
            edge = generate_cat_edge(parsing_anno, ann_list[6:-4])

            edge = cv2.resize(edge, (473, 473), cv2.INTER_NEAREST)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_dir + parent_img_name + "_" + ann_list[6:-4] + ".png", edge)
    print("time :", time.time() - start)


