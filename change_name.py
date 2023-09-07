# Copyright (c) 한승은. All rights reserved.

import os

upper_path = os.path.join(r'D:\Dataset\230703_tagging_samples\CCTV_DB_Dataset\annotations')

# type_list = os.listdir(upper_path)
# for type in type_list:
#     image_path = os.path.join(upper_path, type)
#     image_list = os.listdir(image_path)
#     for image in image_list:
#         if "llip" in image:
#             # print(os.path.join(image_path, image))
#             # print(os.path.join(image_path, image[:-4] + "_flip+crop256" + image[-4:]))
#             print(image)
#             # print('_'.join(image.split('_')[:-1])+"_l_eye.png")
#
#             os.renames(os.path.join(image_path, image), os.path.join(image_path, '_'.join(image.split('_')[:-1]) + "_l_lip.png"))

    # print("end")

image_list = os.listdir(upper_path)
for image in image_list:
    # if "hand" not in image:
    #     # print(image, image[:-4] + "_kf94" + image[-4:])
    #     os.renames(os.path.join(upper_path, image), os.path.join(upper_path, image[:-4]+"_hand"+image[-4:]))
    if "r-ear" in image:
        print(image)
        # print('_'.join(image.split('_')[:-1]) + "_l_lip.png")
        os.renames(os.path.join(upper_path, image), os.path.join(upper_path, '_'.join(image.split('_')[:-1]) + "_r_ear.png"))
print("end")
