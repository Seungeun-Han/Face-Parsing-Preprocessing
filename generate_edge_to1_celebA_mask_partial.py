import os
import numpy as np
import cv2
import time

LABELS = {'bg': 0, 'skin': 1, 'hair': 2, 'l_ear': 3, 'r_ear': 4, 'eye_g': 5, 'hat': 6, 'kf94': 8}

def generate_edge(edge, label):
    h, w = edge.shape

    for i in range(h):
        for j in range(w):
            flag = 1
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x = i + dx
                y = j + dy
                if 0 <= x < w and 0 <= y < h:
                    if label[i, j] != label[x, y] and label[i, j] != 0:
                        edge[i, j] = 1

    return edge

im_path = os.path.join('D:/Dataset/CelebAMask-HQ-mask/images_1024/')
image_list = os.listdir(im_path)
# image_list = [i for i in image_list if int(i[:5]) < 10000]

parsing_anno_path = os.path.join('D:/Dataset/CelebAMask-HQ-mask/seg_1024png/')
annotation_list = os.listdir(parsing_anno_path)
# annotation_list = [i for i in annotation_list if int(i[:5]) < 10000]
annotation_list = [i for i in os.listdir(parsing_anno_path) if i[6:-4] in LABELS.keys()]
annotation_name_list = [i[:5] for i in annotation_list]
annotation_name_list = list(set(annotation_name_list))

save_dir = "D:/Dataset/CelebAMask-HQ-mask/CelebAMask-HQ-maskRendering_Partial_256/edges/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

input_size = 256

for im_list in image_list:
    start = time.time()
    parent_img_name = im_list[:5]
    print(parent_img_name)

    label_edge = np.zeros((input_size, input_size))
    if parent_img_name in annotation_name_list:
        part_list = [i for i in annotation_list if parent_img_name in i]
        # print(part_list)
        for p in part_list:
            annotation_path = parsing_anno_path + p

            parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            parsing_anno = cv2.resize(parsing_anno, (input_size, input_size), cv2.INTER_NEAREST)

            label_edge = generate_edge(label_edge, parsing_anno)

    # kernel = np.ones((2, 2), np.uint8)
    # label_edge = cv2.dilate(label_edge, kernel)
    cv2.imwrite(save_dir + parent_img_name + "_kf94.png", label_edge)
    print("time :", time.time() - start)

