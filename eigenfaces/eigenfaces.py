import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score


def load_face_images(dataset_path):
    subdirectories = os.listdir(dataset_path)
    subdirectories = [dir for dir in subdirectories if os.path.isdir(os.path.join(dataset_path, dir))]

    face_images = []

    for subdirectory in subdirectories:
        images = os.listdir(os.path.join(dataset_path, subdirectory))
        face_images.append([cv2.imread(os.path.join(dataset_path, subdirectory, image), cv2.IMREAD_GRAYSCALE) for image in images])

    return face_images

image_list = load_face_images(r'D:\Dosyalarim\Belgelerim\Dersler\YL\CS554\Project\CS554_Project\preprocessed_att_faces')

train_image_list = image_list[:30]
validation_image_list = image_list[30:]

def apply_pca(image_list, n_components=20):
    image_list = np.array(image_list)
    n_ind, n_img, height, width = image_list.shape
    image_list = image_list.reshape(n_ind * n_img, height * width)
    pca = PCA(n_components=n_components)
    pca.fit(image_list)
    return pca


def project_using_pca(pca, image_list):
    image_list = np.array(image_list)
    n_ind, n_img, height, width = image_list.shape
    image_list = image_list.reshape(n_ind * n_img, height * width)
    components = pca.transform(image_list)
    components = components.reshape(n_ind, n_img, pca.components_.shape[0])
    return components


def get_X_and_y(components):
    X, y = [], []

    for individual_1_id in range(components.shape[0]):
        for individual_2_id in range(components.shape[0]):
            label = 1 if individual_1_id == individual_2_id else 0

            for comp_1 in range(components.shape[1]):
                for comp_2 in range(components.shape[1]):
                    X.append(components[individual_1_id, comp_1] - components[individual_2_id, comp_2])
                    y.append(label)

    return np.array(X), np.array(y)


pca = apply_pca(train_image_list)
train_components = project_using_pca(pca, train_image_list)
validation_components = project_using_pca(pca, validation_image_list)

X_train, y_train = get_X_and_y(train_components)
X_val, y_val = get_X_and_y(validation_components)
print()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

print('Fitting')
# model = RandomForestClassifier(verbose=2, n_jobs=-1, class_weight={0: 0.03333, 1: 0.96667}) # {0: 0.03333, 1: 0.96667})
# model = GradientBoostingClassifier(verbose=2) # {0: 0.03333, 1: 0.96667})
model = XGBClassifier(random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print(precision_score(y_val, y_pred))
print(recall_score(y_val, y_pred))

ind_1_img_1 = validation_image_list[1][0]
ind_1_img_2 = validation_image_list[1][2]

ind_2_img_1 = validation_image_list[2][0]
ind_2_img_2 = validation_image_list[2][2]

'''cv2.imshow('ind_1_img_1', ind_1_img_1)
cv2.waitKey(0)

cv2.imshow('ind_1_img_2', ind_1_img_2)
cv2.waitKey(0)

cv2.imshow('ind_2_img_1', ind_2_img_1)
cv2.waitKey(0)

cv2.imshow('ind_2_img_2', ind_2_img_2)
cv2.waitKey(0)'''

ind_1_comp_1 = project_using_pca(pca, [[ind_1_img_1]])
ind_1_comp_2 = project_using_pca(pca, [[ind_1_img_2]])

ind_2_comp_1 = project_using_pca(pca, [[ind_2_img_1]])
ind_2_comp_2 = project_using_pca(pca, [[ind_2_img_2]])

print(model.predict(np.array(ind_1_comp_1 - ind_1_comp_2)[0]))
print(model.predict(np.array(ind_2_comp_1 - ind_2_comp_2)[0]))

print(model.predict(np.array(ind_1_comp_1 - ind_2_comp_1)[0]))
print(model.predict(np.array(ind_1_comp_1 - ind_2_comp_2)[0]))
print(model.predict(np.array(ind_1_comp_2 - ind_2_comp_1)[0]))
print(model.predict(np.array(ind_1_comp_2 - ind_2_comp_2)[0]))


'''copied = np.array(image_list)
image_list = np.array(image_list)
n_ind, n_img, height, width = image_list.shape
image_list = image_list.reshape(n_ind * n_img, height * width)'''

'''n_components = 20
pca = PCA(n_components= n_components)
# pca.fit(image_list)
projected_image_list = pca.fit_transform(image_list)
components = projected_image_list.reshape(n_ind, n_img, n_components)
print(components.shape)

distance_values = []

X_train, y_train = [], []

for i in range(30):
    ind_1_components = components[i]
    for j in range(30):
        ind_2_components = components[j]

        for k in range(n_img):
            for l in range(n_img):
                c1 = ind_1_components[k]
                c2 = ind_2_components[l]
                X_train.append(c1 - c2)
                y_train.append(i == j)
                # dist = np.sum(np.square(c1 - c2))
                # distance_values.append((dist, i == j))

X_val, y_val = [], []
for i in range(30, 40):
    ind_1_components = components[i]
    for j in range(30, 40):
        ind_2_components = components[j]

        for k in range(n_img):
            for l in range(n_img):
                c1 = ind_1_components[k]
                c2 = ind_2_components[l]
                X_val.append(c1 - c2)
                y_val.append(i == j)'''

'''X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)



intra_dist, inter_dist = [], []

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

print('Fitting')
# model = RandomForestClassifier(verbose=2, n_jobs=-1, class_weight={0: 0.03333, 1: 0.96667}) # {0: 0.03333, 1: 0.96667})
# model = GradientBoostingClassifier(verbose=2) # {0: 0.03333, 1: 0.96667})
model = XGBClassifier(scale_pos_weight=29, verbosity=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print(precision_score(y_val, y_pred))
print(recall_score(y_val, y_pred))'''

'''

for dist, label in distance_values:
    if label is True:
        intra_dist.append(dist)
    else:
        inter_dist.append(dist)

print(np.mean(np.array(intra_dist)))
print(np.std(np.array(intra_dist)))

print()

print(np.mean(np.array(inter_dist)))
print(np.std(np.array(inter_dist)))

plt.hist(intra_dist)
plt.show()
plt.hist(inter_dist)
plt.show()

gt_label = []
predicted_label = []

for dist, label in distance_values:
    gt_label.append(label)
    predicted_label.append(True if dist < 1.4e7 else False)

gt_label, predicted_label = np.array(gt_label, dtype=np.float32), np.array(predicted_label, dtype=np.float32)'''

'''plt.hist(gt_label)
plt.show()
plt.hist(predicted_label)
plt.show()'''
'''print(np.sum(predicted_label == gt_label))
print(len(gt_label))

print(precision_score(gt_label, predicted_label))
print(recall_score(gt_label, predicted_label))'''

# print(np.mean(np.array(accuracy)))
'''myimg = image_list[10].reshape(height, width)
mean_image = np.array(pca.mean_.reshape(height, width), dtype=np.uint8)
comp1 = np.array(pca.components_[1].reshape(height, width))
comp1 -= np.min(comp1)
comp1 *= 255.0 / np.max(comp1)
comp1 = np.uint8(comp1)
# cv2.imshow('img', comp1)
# cv2.waitKey(0)

img1 = copied[0][0]
img2 = copied[0][1]

img1_projected, img2_projected = pca.transform([img1.flatten()]), pca.transform([img2.flatten()])

dist = np.sqrt(np.sum(np.square(img1_projected[0] - img2_projected[0])))
print(dist)'''

