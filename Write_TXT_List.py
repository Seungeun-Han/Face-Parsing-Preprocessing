import os

img_path = r'D:\Dataset\CelebAMask-HQ\CelebA-HQ_473\test\images'  # train / valid / test

img_list = os.listdir(img_path)

print(len(img_list))

txt_path = r"D:\Dataset\CelebAMask-HQ\CelebA-HQ_473/test_list.txt"   #  train / valid_list / test_list

file = open(txt_path, 'w')

for im in img_list:
    file.write("'images/{0}.jpg labels/{0}.png'\n".format(im[:-4]))

file.close()

