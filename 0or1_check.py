"""
@Author     :   Seungeun Han
@Contact    :   hse@etri.re.kr
@Time       :   23/05/31
@License    :   Licensed under the Apache License, Version 2.0 (the "License");
@Copyright  :   All Rights Reserved.

This code is designed to verify whether all pixel values in the image are 0~1 or not.

왜 이런 코드가 필요하느냐?
    >> Edge Label이 잘 만들어졌느지 확인하기 위해서

Edge Label은 배경 영역이 0이고 Edge 영역이 1으로 저장되어야 한다.
만약 그렇지 않다면, 0과 1로 만들기 위해 이 코드가 필요하다.
"""

import os
import cv2
import numpy as np

# 원본 이미지들이 저장되어있는 폴더 경로
path = r"D:\Dataset\ETRI_MaskDB\ETRI_MaskDB_473\edges"
# 폴더에 있는 파일의 이름을 모두 리스트 형태로 읽어옴
label_list = os.listdir(path)

# 변경된 이미지를 저장할 경로
s_path = r"D:\Dataset\ETRI_MaskDB\ETRI_MaskDB_473\edges_1"
# 해당 경로가 존재하지 않는다면 만든다.
if not os.path.exists(s_path):
    os.makedirs(s_path)

for i in label_list:
    image = cv2.imread(os.path.join(path, i), cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    # print(i, image.shape)

    # 더 빠른 동작을 위해 2차원을 1차원으로 변형
    image = np.reshape(image, (1, -1))
    image = np.array(image[0])

    # 만약 image에 1 초과의 값이 있다면
    if np.any(image > 1):
        print(i)

        # 해당 위치의 인덱스 가져옴
        list = [i for i, v in enumerate(image) if v > 1]

        # 그 인덱스의 값을 1로 설정
        image[list] = 1

        # 다시 2차원으로 변형
        image = np.reshape(image, (h, -1))

        # 이미지 저장
        cv2.imwrite(os.path.join(s_path, i), image)
