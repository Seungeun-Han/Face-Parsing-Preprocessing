import os
import numpy as np
import cv2
import time

LABELS = {'bg': 0, 'skin': 1, 'nose': 2, 'eye_g': 3, 'l_eye': 4, 'r_eye': 5,
        'l_brow': 6, 'r_brow': 7, 'l_ear': 8, 'r_ear': 9, 'mouth': 10, 'u_lip': 11,
        'l_lip': 12, 'hair': 13, 'hat': 14, 'ear_r': 15, 'neck_l': 16, 'neck': 17, 'cloth': 18,'mask' : 19}
'''LABELS = {'bg': 0, 'skin': 1, 'nose': 2, 'eye_g': 3, 'l_eye': 4, 'r_eye': 5,
        'l_brow': 6, 'r_brow': 7, 'l_ear': 8, 'r_ear': 9, 'mouth': 10, 'u_lip': 11,
        'l_lip': 12, 'hair': 13, 'hat': 14, 'ear_r': 15, 'neck_l': 16, 'neck': 17, 'cloth': 18}''' #non-mask data label

def generate_cat_edge(merged_cat_edge, edge, edge_name):
    h, w = edge.shape
    #merged_cat_edge = np.zeros(seg.shape)

    merged_cat_edge = np.reshape(merged_cat_edge, (1, -1))
    edge = np.reshape(edge, (1, -1))

    edge_list = [i for i in range(len(edge[0])) if edge[0][i] == LABELS[edge_name]]

    assert_list = ['skin', 'l_ear', 'r_ear', 'l_brow', 'r_brow', 'neck', 'l_eye', 'r_eye', 'hair', 'nose']

    if edge_name not in assert_list:
        merged_cat_edge[0][edge_list] = LABELS[edge_name]  # *10

    elif edge_name == 'skin':  # cond1, 11
        skin_idx = [x for x in range(len(merged_cat_edge[0])) if merged_cat_edge[0][x] == 0 or merged_cat_edge[0][x] == LABELS['ear_r']]
        intersection = list(set(edge_list) & set(skin_idx))
        merged_cat_edge[0][intersection] = LABELS[edge_name]  # * 10
    elif edge_name == 'l_ear' or edge_name == 'r_ear':  # cond2, 10
        ear_idx = [x for x in range(len(merged_cat_edge[0])) if
                   merged_cat_edge[0][x] != LABELS['hair'] and merged_cat_edge[0][x] != LABELS['ear_r']]  # *10
        intersection = list(set(edge_list) & set(ear_idx))
        merged_cat_edge[0][intersection] = LABELS[edge_name]  # * 10
    elif edge_name == 'l_brow' or edge_name == 'r_brow':  # cond3, 7
        brow_idx = [x for x in range(len(merged_cat_edge[0])) if
                    merged_cat_edge[0][x] != LABELS['hair'] and merged_cat_edge[0][x] != LABELS['hat']]  # *10
        intersection = list(set(edge_list) & set(brow_idx))
        merged_cat_edge[0][intersection] = LABELS[edge_name]  # * 10
    elif edge_name == 'l_eye' or edge_name == 'r_eye':  # cond4
        eye_idx = [x for x in range(len(merged_cat_edge[0])) if merged_cat_edge[0][x] != LABELS['hair']]  # *10
        intersection = list(set(edge_list) & set(eye_idx))
        merged_cat_edge[0][intersection] = LABELS[edge_name]  # * 10
    elif edge_name == 'neck':  # cond5
        neck_idx = [x for x in range(len(merged_cat_edge[0])) if merged_cat_edge[0][x] != LABELS['ear_r']]  # *10
        intersection = list(set(edge_list) & set(neck_idx))
        merged_cat_edge[0][intersection] = LABELS[edge_name]  # * 10
    elif edge_name == 'hair':  # cond6, 9
        hair_idx = [x for x in range(len(merged_cat_edge[0])) if
                    merged_cat_edge[0][x] != LABELS['eye_g'] and merged_cat_edge[0][x] != LABELS['ear_r']]  # *10
        intersection = list(set(edge_list) & set(hair_idx))
        merged_cat_edge[0][intersection] = LABELS[edge_name]  # * 10
    elif edge_name == 'nose':  # cond8
        hair_idx = [x for x in range(len(merged_cat_edge[0])) if
                    merged_cat_edge[0][x] != LABELS['hair'] and merged_cat_edge[0][x] != LABELS['eye_g']]  # *10
        intersection = list(set(edge_list) & set(hair_idx))
        merged_cat_edge[0][intersection] = LABELS[edge_name]  # * 10

    merged_cat_edge = np.reshape(merged_cat_edge, (h, -1))

    return merged_cat_edge

im_path = os.path.join('D:/Dataset/ETRI_MaskDB/image_473')
image_list = os.listdir(im_path)
#im = cv2.imread(im_path, cv2.IMREAD_COLOR)
ann_1_list = [x for x in image_list if '-00' in x]
ann_2_list = [x for x in image_list if '2022' in x]
ann_3_list = [x for x in image_list if 'IMG' in x]
parsing_anno_path = os.path.join('D:/Dataset/ETRI_MaskDB/cat_edges_thickness3/')
annotation_list = os.listdir(parsing_anno_path)

save_dir = "D:/Dataset/ETRI_MaskDB/merged_cat_edges_thickness3/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for im_list in ann_3_list:
    start = time.time()
    im_list = im_list[:-4]
    print(im_list)

    #merged_cat_edge = np.zeros((512, 512))
    merged_cat_edge = np.zeros((473, 473))
    for idx, ann_list in enumerate(annotation_list):
        if im_list in ann_list:
            annotation_path = parsing_anno_path + ann_list
            parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)

            # merged_cat_edge = generate_cat_edge(merged_cat_edge, parsing_anno, ann_list[11:-4])
            # merged_cat_edge = generate_cat_edge(merged_cat_edge, parsing_anno, ann_list[16:-4])
            merged_cat_edge = generate_cat_edge(merged_cat_edge, parsing_anno, ann_list[9:-4])
            cv2.imwrite(save_dir + im_list + ".png", merged_cat_edge)
    print("time :", time.time() - start)

'''        elif ann_list in ann_2_list:
            parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            #print(ann_list[16:-4])
            merged_cat_edge = generate_cat_edge(merged_cat_edge, parsing_anno, ann_list[16:-4])
            cv2.imwrite(save_dir + im_list + ".png", merged_cat_edge)
        elif ann_list in ann_3_list:
            parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            #print(ann_list[6:-4])
            merged_cat_edge = generate_cat_edge(merged_cat_edge, parsing_anno, ann_list[9:-4])
            cv2.imwrite(save_dir + im_list + ".png", merged_cat_edge)'''

