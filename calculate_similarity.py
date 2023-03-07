import cv2
import os
import numpy as np

image1_dir = "D:/Dataset/CelebAMask-HQ/CelebA-HQ/test/images/00256.jpg"
image2_dir = "C:/Users/USER/Downloads/MAT-main/MAT-main/CelebMaskRendering_256_results/00256_kf94.png"

image1 = cv2.imread(image1_dir, cv2.IMREAD_COLOR)
image2 = cv2.imread(image2_dir, cv2.IMREAD_COLOR)

# BGR 이미지를 HSV 이미지로 변환
hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
# 히스토그램 연산(파라미터 순서 : 이미지, 채널, Mask, 크기, 범위)
hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
# 정규화(파라미터 순서 : 정규화 전 데이터, 정규화 후 데이터, 시작 범위, 끝 범위, 정규화 알고리즘)
cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

CORREL = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
CHISQR = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

print(CORREL)
print(CHISQR)