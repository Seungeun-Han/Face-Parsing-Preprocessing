import os

img_path = r'./CelebA-HQ+ETRI_Mask+CAMask+CAHand_CCTV_256\train/images'  # train / valid / test

img_list = [i for i in os.listdir(img_path) if "Flip" not in i]

print(len(img_list))

txt_path = r"./CelebA-HQ+ETRI_Mask+CAMask+CAHand_CCTV_256/train_list_notFLip.txt"   #  train / valid_list / test_list

file = open(txt_path, 'w')

for im in img_list:
    file.write("'images/{0}.jpg labels/{0}.png'\n".format(im[:-4]))

file.close()

