import os

import cv2
from tqdm import tqdm


def convert_pgm_to_png(dataset_path, target_path):
    subdirectories = os.listdir(dataset_path)
    subdirectories = [dir for dir in subdirectories if os.path.isdir(os.path.join(dataset_path, dir))]

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for subdirectory in tqdm(subdirectories):
        image_names = os.listdir(os.path.join(dataset_path, subdirectory))

        if not os.path.exists(os.path.join(target_path, subdirectory)):
            os.makedirs(os.path.join(target_path, subdirectory))

        for image_name in image_names:
            image = cv2.imread(os.path.join(dataset_path, subdirectory, image_name))
            cv2.imwrite(os.path.join(target_path, subdirectory, image_name.replace('.pgm', '.jpg')), image)

convert_pgm_to_png(r'D:\Dosyalarim\Belgelerim\Dersler\YL\CS554\Project\att_faces\orl_faces', r'D:\Dosyalarim\Belgelerim\Dersler\YL\CS554\Project\preprocessed_att_faces')