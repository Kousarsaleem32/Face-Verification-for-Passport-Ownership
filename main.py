import os.path

from landmark_localization.active_appearance_model import ActiveAppearanceModel
from landmark_localization.landmark_dataset_reader import read_landmark_dataset

dataset_path = os.path.join(os.getcwd(), 'MUCT_Dataset')
image_list, landmark_list = read_landmark_dataset(os.path.join(dataset_path, 'jpg'), os.path.join(dataset_path, 'muct-landmarks', 'muct76-opencv.csv'))

active_appearance_model = ActiveAppearanceModel(image_list, landmark_list)