import os
import numpy as np
import cv2
import time

def generate_edge(edge, label):
    h, w = edge.shape
    # print(h, w)
    for i in range(h):
        for j in range(w):
            flag = 1
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x = i + dx
                y = j + dy
                if 0 <= x < h and 0 <= y < w:
                    if label[i, j] != label[x, y]:
                        edge[i, j] = 1

    return edge

parsing_anno_path = os.path.join('D:/Dataset/LaPa/train/labels/')
annotation_list = os.listdir(parsing_anno_path)

save_dir = "D:/Dataset/LaPa/train/edges/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saved_list = os.listdir(save_dir)
annotation_list = [i for i in annotation_list if i not in saved_list]
print(len(annotation_list))

for label_name in annotation_list:
    start = time.time()
    print(label_name)

    parsing_anno = cv2.imread(os.path.join(parsing_anno_path, label_name), cv2.IMREAD_GRAYSCALE)
    h, w = parsing_anno.shape
    # print(h, w)
    label_edge = np.zeros((h, w))
    label_edge = generate_edge(label_edge, parsing_anno)

    # kernel = np.ones((2, 2), np.uint8)
    # label_edge = cv2.dilate(label_edge, kernel)
    cv2.imwrite(save_dir + label_name, label_edge)
    """cv2.imshow("label_edge", label_edge)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break"""
    print("time :", time.time() - start)

