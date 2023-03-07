import os
import numpy as np
import cv2
import time

def generate_edge(label, edge_width=3):
    h, w = label.shape
    edge = np.zeros(label.shape)

    for i in range(h):
        for j in range(w):
            flag = 1
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x = i + dx
                y = j + dy
                if 0 <= x < w and 0 <= y < h:
                    if label[i, j] != label[x, y]:
                        edge[i, j] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)
    return edge

image_name = "001886-005"

parsing_anno_path = os.path.join('D:/Dataset/ETRI_MaskDB/label_split/')
annotation_list = os.listdir(parsing_anno_path)

save_dir = "D:/Dataset/ETRI_MaskDB/tmp_edges/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


label_edge = np.zeros((473, 473))
for idx, ann_list in enumerate(annotation_list):
    if image_name in ann_list:
        annotation_path = parsing_anno_path + ann_list
        parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
        parsing_anno = cv2.resize(parsing_anno, (473, 473), cv2.INTER_NEAREST)

        label_edge += generate_edge(parsing_anno)


cv2.imwrite(save_dir + image_name + ".png", label_edge)


