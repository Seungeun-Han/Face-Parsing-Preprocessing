import os
import cv2


def split_DataSet(upper_path = "./", trainValidTest = 'train', list = []):
    type = ['edges', 'images', 'labels']
    for t in type:
        label_path = os.path.join(upper_path, t)

        save_path = os.path.join(upper_path, trainValidTest, t)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in list:
            # print(i)
            if t == "images":
                image = cv2.imread(os.path.join(label_path, i[:-4]+".jpg"), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(save_path, i[:-4]+".jpg"), image)
            else:
                image = cv2.imread(os.path.join(label_path, i), cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(os.path.join(save_path, i), image)


upper_path = r"D:\Dataset\230703_tagging_samples\CCTV_DB_Dataset\CCTV_DB_Dataset_256"

all_path = r"D:\Dataset\230703_tagging_samples\CCTV_DB_Dataset\CCTV_DB_Dataset_473\images"
all_list = [i[:-4]+".png" for i in os.listdir(all_path)]

train_image_path = r"D:\Dataset\230703_tagging_samples\CCTV_DB_Dataset\CCTV_DB_Dataset_473\train\images"
train_image_list = [i[:-4]+".png" for i in os.listdir(train_image_path)]
# print(len(train_image_list))  #

others_image_list = list(set(all_list) - set(train_image_list))
# print(others_image_list)
# print(len(others_image_list))

valid_image_list = others_image_list[:len(others_image_list)//2]
test_image_list = others_image_list[len(others_image_list)//2:]
# print(valid_image_list)
print(len(valid_image_list))  # 46
print(len(test_image_list))  # 46

split_DataSet(upper_path, 'valid', valid_image_list)
split_DataSet(upper_path, 'test', test_image_list)
