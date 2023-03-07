import os

save_dir = r"D:\Dataset\ETRI_MaskDB\mask_dataset_th3_px1_256\not_aug_labels/"
saved_list = os.listdir(save_dir)

for subject in saved_list:
    """if len(subject) > 10:
        print(subject)
        os.remove(save_dir+subject)"""
    if "RandomBrightnessContrast" in subject: # HorizontalFlip RandomBrightnessContrast
        print(subject)
        os.remove(save_dir + subject)