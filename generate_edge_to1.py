import os
import numpy as np
import cv2
import time

def generate_edge(label, edge_width=3):
    h, w = label.shape
    edge = np.zeros(label.shape)

    """# left
    for i in range(h):
        for j in range(1, w):
            if label[i][j] != 0 and label[i][j - 1] == 0:
                edge[i][j] = 1

    # right
    for i in range(h):
        for j in range(w-1):
            if label[i][j] != 0 and label[i][j+1] == 0:
                edge[i][j] = 1

    # up
    for i in range(1, h):
        for j in range(w):
            if label[i][j] != 0 and label[i-1][j] == 0:
                edge[i][j] = 1

    # down
    for i in range(h-1):
        for j in range(w):
            if label[i][j] != 0 and label[i+1][j] == 0:
                edge[i][j] = 1"""
    for i in range(h):
        for j in range(w):
            flag = 1
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x = i + dx
                y = j + dy
                if 0 <= x < w and 0 <= y < h:
                    if label[i, j] != label[x, y] and label[i, j] != 0:
                        edge[i, j] = 1

    """kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_width, edge_width))
    edge = cv2.dilate(edge, kernel)"""
    return edge

# im_path = os.path.join('D:/Dataset/ETRI_MaskDB/image/')
im_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/')
image_list = os.listdir(im_path)
#im = cv2.imread(im_path, cv2.IMREAD_COLOR)

# parsing_anno_path = os.path.join('D:/Dataset/ETRI_MaskDB/label_split/')
parsing_anno_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno_acc/')
annotation_list = os.listdir(parsing_anno_path)

save_dir = "D:/Dataset/CelebAMask-HQ/CelebA-HQ_256/edges/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

INPUT_SIZE = 256

for im_list in image_list:
    start = time.time()
    parent_img_name = im_list[:-4].zfill(5)
    print(parent_img_name)

    label_edge = np.zeros((INPUT_SIZE, INPUT_SIZE))
    for idx, ann_list in enumerate(annotation_list):
        if parent_img_name in ann_list:
            annotation_path = parsing_anno_path + ann_list
            parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            parsing_anno = cv2.resize(parsing_anno, (INPUT_SIZE, INPUT_SIZE), cv2.INTER_NEAREST)

            label_edge += generate_edge(parsing_anno)

            #cv2.imshow("im", im)
            #cv2.imshow("parsing_anno", parsing_anno)
            #cv2.imshow("label_edge", label_edge)
            #cv2.waitKey(10)

    cv2.imwrite(save_dir + parent_img_name + ".png", label_edge)
    print("time :", time.time() - start)


