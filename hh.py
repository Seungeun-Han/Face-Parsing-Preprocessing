import os
import cv2

def change_name_to_5blanks_and_toPng():
    path = "D:/Dataset/CelebAMask-HQ-mask/seg_1024jpg/"
    list = os.listdir(path)

    s_path = "D:/Dataset/CelebAMask-HQ-mask/seg_1024png/"
    if not os.path.exists(s_path):
        os.makedirs(s_path)

    for i in list:
        image = cv2.imread(path+i, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(s_path+i[:-9].zfill(5) + i[-9:-3] + 'png', image)

def from_celebLabelFolder_to_maskFolder():
    anno_path = "D:/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebAMask-HQ-mask-anno_acc/"
    anno_list = os.listdir(anno_path)
    anno_list = [i for i in anno_list if "nose" not in i and "lip" not in i and "mouth" not in i]
    anno_name_list = [i[:5] for i in anno_list]

    s_path = "D:/Dataset/CelebAMask-HQ-mask/seg_1024png/"
    save_list = os.listdir(s_path)
    save_name_list = [i[:5] for i in save_list]

    for i in anno_list:
        if i[:5] in save_name_list:
            print(i)
            image = cv2.imread(anno_path + i, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(s_path + i, image)

from_celebLabelFolder_to_maskFolder()