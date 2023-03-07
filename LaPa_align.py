import cv2
import numpy as np
import os
from skimage import transform as trans

img_dir = "D:/Dataset/LaPa/val/images"
image_list = os.listdir(img_dir)
label_dir = "D:/Dataset/LaPa/val/labels"
landmark_dir = "D:/Dataset/LaPa/val/landmarks"
img_save_dir = "D:/Dataset/LaPa_aligned/valid/images"
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)
saved_list = os.listdir(img_save_dir)
label_save_dir = "D:/Dataset/LaPa_aligned/valid/labels"
if not os.path.exists(label_save_dir):
    os.makedirs(label_save_dir)
image_list = [i for i in image_list if i not in img_save_dir]
print(len(image_list))
width, height = 473, 473

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
        landmark_txt = os.path.join(landmark_dir, img_name[:-4]+".txt")
        f = open(landmark_txt, "r")
        lines = f.readlines()
        """
        index
        왼 눈 가운데: 105
        오 눈 가운데: 106
        코: 55
        왼 입 끝: 85
        오 입 끝: 101
        """

        le = lines[105].strip().split(' ')
        re = lines[106].strip().split(' ')
        nose = lines[55].strip().split(' ')
        lm = lines[85].strip().split(' ')
        rm = lines[101].strip().split(' ')

        """
        # landms
        cv2.circle(image, (int(float(le[0])), int(float(le[1]))), 1, (0, 0, 255), 4)  # 화면을 볼 때 왼쪽 눈
        cv2.circle(image, (int(float(re[0])), int(float(re[1]))), 1, (0, 255, 255), 4)  # 화면을 볼 때 오른쪽 눈
        cv2.circle(image, (int(float(nose[0])), int(float(nose[1]))), 1, (255, 0, 255), 4)
        cv2.circle(image, (int(float(lm[0])), int(float(lm[1]))), 1, (0, 255, 0), 4)
        cv2.circle(image, (int(float(rm[0])), int(float(rm[1]))), 1, (255, 0, 0), 4)
        """

        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = [le[0], le[1]]
        landmark5[1] = [re[0], re[1]]
        landmark5[2] = [nose[0], nose[1]]
        landmark5[3] = [lm[0], lm[1]]
        landmark5[4] = [rm[0], rm[1]]

        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, warp_arr)
        M = tform.params[0:2, :]
        face_align = cv2.warpAffine(image, M, (width, height), borderValue=0.0)
        label_align = cv2.warpAffine(label, M, (width, height), borderValue=0.0)


        cv2.imwrite(os.path.join(img_save_dir, img_name), face_align)
        cv2.imwrite(os.path.join(label_save_dir, label_name), label_align)

        """cv2.imshow("face_align", face_align)
        cv2.imshow("label_align", label_align)
        # cv2.imshow("image", image)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break"""
        f.close()








