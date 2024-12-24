import os

import cv2
import numpy as np
import pandas as pd


def read_landmark_dataset(image_folder, csv_file_path):
    image_names = os.listdir(image_folder)
    landmark_df = pd.read_csv(csv_file_path)

    landmark_columns = landmark_df.columns.tolist()[2:]

    image_list, landmarks_list = [], []

    for image_name in image_names:
        image_path = os.path.join(image_folder, image_name)
        # image_list.append(cv2.imread(image_path))
        image_list.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))

        landmark_df_line = landmark_df[landmark_df['name'] == image_name.replace('.jpg', '')]
        landmarks = []

        for i in range(0, len(landmark_columns), 2):
            x = landmark_df_line[landmark_columns[i]].iloc[0]
            y = landmark_df_line[landmark_columns[i + 1]].iloc[0]
            landmarks.append((x, y))

        landmarks_list.append(landmarks)

    return image_list, landmarks_list
