# Copyright (c) 한승은. All rights reserved.

import os
import cv2
import numpy as np

# 경로 1
one_path = os.path.join(r'D:\Dataset\CelebAMask-HQ\CelebA-HQ_112\edges')
one_list = os.listdir(one_path)

# 경로 2
two_path = os.path.join(r'D:\Dataset\CelebAMask-HQ\CelebA-HQ_112\edges_1')
two_list = os.listdir(two_path)

# 지워야 할 리스트 / 경로 1에 있으면서 2에도 있으면 지우기
remove_list = [i for i in one_list if i in two_list]

# 지움
print(len(remove_list))
for subject in remove_list:
    os.remove(os.path.join(one_path, subject))

# 이동 함수 / 이 예시에서는 두 번째 폴더에만 있는 이미지 리스트 가져옴
def move():
    # im_dir = r"D:\Dataset\230703_tagging_samples\intern01_tagging_results\scarf_images"
    save_dir = r"D:\Dataset\230703_tagging_samples\CCTV_DB_Dataset\images"  # 이동할 이미지를 저장할 폴더
    
    one_list = [i[:-4] for i in one_path]  # 옵션

    # CCTV DB 맞춤형 파일 이름 조건 ex) 2_QHD_4_Cap_Left_1_Front_393701.jpg / 필요 없다면 주석처리 
    set_two_list = list(set(['_'.join(i.split('_')[:-1]) if len(i.split('_')) < 10 else '_'.join(i.split('_')[:-2]) for i in two_list]))
    print(len(set_two_list))
    # print(list(set([i.split('_')[-1] for i in set_two_list])))

    #이동 리스트
    move_list = list(set(set_two_list)-set(one_list))
    print(move_list)
    
    for m in move_list:
        print(m)
        img = cv2.imread(os.path.join(one_path, m+".jpg"), cv2.IMREAD_COLOR)

        cv2.imwrite(os.path.join(save_dir, m+".jpg"), img)



    

    
