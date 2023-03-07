import os
import numpy as np
import cv2
import time

LABELS = {'bg': 0, 'skin': 1, 'nose': 2, 'eye_g': 3, 'l_eye': 4, 'r_eye': 5,
        'l_brow': 6, 'r_brow': 7, 'l_ear': 8, 'r_ear': 9, 'mouth': 10, 'u_lip': 11,
        'l_lip': 12, 'hair': 13, 'hat': 14, 'ear_r': 15, 'neck_l': 16, 'neck': 17, 'cloth': 18}

def make_groundTruth(label, seg, seg_name):
    h, w = seg.shape
    for i in range(h):
        for j in range(w):
            if seg[i][j] >= 150 and seg_name != 'skin':
                label[i][j] = LABELS[seg_name]
            elif seg[i][j] >= 150 and seg_name == 'skin':
                if label[i][j] != 0:
                    continue
                label[i][j] = LABELS[seg_name]

    #cv2.imwrite("D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/labels/"+seg_name+".png", label)
    #print("seg_name", label_idx)
    return label

im_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img/')
image_list = os.listdir(im_path)
#im = cv2.imread(im_path, cv2.IMREAD_COLOR)

parsing_anno_path = os.path.join('D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno_acc/')
annotation_list = os.listdir(parsing_anno_path)

save_dir = "D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/labels/"
for im_list in image_list:
    start = time.time()
    parent_img_name = im_list[:-4].zfill(5)
    print(parent_img_name)

    label = np.zeros((512, 512))
    for idx, ann_list in enumerate(annotation_list):
        if parent_img_name in ann_list:
            annotation_path = parsing_anno_path + ann_list
            parsing_anno = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
            #print(ann_list[6:-4])
            label = make_groundTruth(label, parsing_anno, ann_list[6:-4])

    label = cv2.resize(label, (473, 473), cv2.INTER_NEAREST)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(save_dir + parent_img_name + ".png", label)
    print("time :", time.time() - start)
    #cv2.imshow("im", im)
    #cv2.imshow("parsing_anno", parsing_anno)
    #cv2.imshow("label_edge", label_edge)
    #cv2.waitKey(10)



