import cv2
import numpy as np
import os
from skimage import transform as trans

img_dir = "D:/Dataset/Helen/test/images"
image_list = os.listdir(img_dir)
label_dir = "D:/Dataset/Helen/test/labels"
landmark_dir = "D:/Dataset/Helen/annotation"
landmark_list = os.listdir(landmark_dir)
img_save_dir = "D:/Dataset/Helen_aligned/test/images"
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)
saved_list = os.listdir(img_save_dir)
label_save_dir = "D:/Dataset/Helen_aligned/test/labels"
if not os.path.exists(label_save_dir):
    os.makedirs(label_save_dir)
txt_save_dir = "D:/Dataset/Helen/landmark_txt"
if not os.path.exists(txt_save_dir):
    os.makedirs(txt_save_dir)
image_list = [i for i in image_list if i not in img_save_dir]
width, height = 473, 473


def make_image_name_txt():
    for t in landmark_list:
        landmark_txt = os.path.join(landmark_dir, t)
        f = open(landmark_txt, "r")
        lines = f.readlines()
        # print(lines[0].strip()+".txt")
        txt_name = lines[0].strip()+".txt"
        # print(len(lines))
        save_txt = open(os.path.join(txt_save_dir, txt_name), "w")
        for i, l in enumerate(lines):
            if i == 0:
                continue
            save_txt.write(l)

        f.close()
        save_txt.close()


if __name__ == '__main__':
    warp_arr = np.array([
        [182, 229],
        [295, 229],
        [238, 301],
        [190, 349],
        [288, 349]], dtype=np.float32)
    warp_arr[:, 0] += 8.0

    for img_name in image_list:
        print(img_name)
        label_name = img_name[:-4] + ".png"
        image = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_COLOR)
        label = cv2.imread(os.path.join(label_dir, label_name), cv2.IMREAD_GRAYSCALE)
        landmark_txt = os.path.join(txt_save_dir, img_name[:-4]+".txt")
        f = open(landmark_txt, "r")
        lines = f.readlines()
        """
        index
        왼 눈 가운데: 105
        왼 눈 제일 오른쪽: 134
        왼 눈 제일 왼쪽: 145
        왼 눈 제일 위쪽: 140
        왼 눈 제일 아래쪽: 149
        오 눈 가운데: 106
        오 눈 제일 오른쪽: 125
        오 눈 제일 왼쪽: 114
        오 눈 제일 위: 119
        오 눈 제일 아래: 130
        코: 49
        왼 입 끝: 113
        오 입 끝: 100 or 99
        """

        le_u = lines[140].strip().split(' , ')
        le_d = lines[149].strip().split(' , ')
        le_r = lines[134].strip().split(' , ')
        le_l = lines[145].strip().split(' , ')

        re_u = lines[119].strip().split(' , ')
        re_d = lines[130].strip().split(' , ')
        re_r = lines[125].strip().split(' , ')
        re_l = lines[114].strip().split(' , ')

        le_center = [(float(le_r[0]) + float(le_l[0])) / 2, (float(le_u[1]) + float(le_d[1])) / 2]  # 왼눈 중심
        re_center = [(float(re_r[0]) + float(re_l[0])) / 2, (float(re_u[1]) + float(re_d[1])) / 2]  # 오눈 중심

        nose = lines[49].strip().split(' , ')
        lm = lines[113].strip().split(' , ')
        rm = lines[100].strip().split(' , ')

        """# landms
        le_center = [int((float(le_r[0]) + float(le_l[0])) / 2), int((float(le_u[1]) + float(le_d[1])) / 2)]  # 왼눈 중심
        re_center = [int((float(re_r[0]) + float(re_l[0])) / 2), int((float(re_u[1]) + float(re_d[1])) / 2)]  # 오눈 중심
        
        cv2.circle(image, (le_center[0], le_center[1]), 1, (0, 0, 255), 4)
        cv2.circle(image, (re_center[0], re_center[1]), 1, (0, 255, 255), 4)
        cv2.circle(image, (int(float(nose[0])), int(float(nose[1]))), 1, (255, 0, 255), 4)
        cv2.circle(image, (int(float(lm[0])), int(float(lm[1]))), 1, (0, 255, 0), 4)
        cv2.circle(image, (int(float(rm[0])), int(float(rm[1]))), 1, (255, 0, 0), 4)"""

        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = [le_center[0], le_center[1]]
        landmark5[1] = [re_center[0], re_center[1]]
        landmark5[2] = [nose[0], nose[1]]
        landmark5[3] = [lm[0], lm[1]]
        landmark5[4] = [rm[0], rm[1]]

        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, warp_arr)
        M = tform.params[0:2, :]
        face_align = cv2.warpAffine(image, M, (width, height), borderValue=0.0)
        label_align = cv2.warpAffine(label, M, (width, height), borderValue=0.0)

        """cv2.imshow("face_align", face_align)
        cv2.imshow("label_align", label_align)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break"""

        cv2.imwrite(os.path.join(img_save_dir, img_name), face_align)
        cv2.imwrite(os.path.join(label_save_dir, label_name), label_align)

        f.close()





