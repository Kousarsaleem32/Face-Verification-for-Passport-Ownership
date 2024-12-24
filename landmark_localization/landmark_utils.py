import numpy as np
import cv2


def rescale_image_with_landmarks(image, landmarks, scale):
    image = cv2.resize(image, (int(image.shape[0] * scale), int(image.shape[1] * scale)))
    landmarks = np.array(landmarks) * scale
    return image, landmarks


def draw_landmarks_on_image(image, landmarks):
    for landmark in landmarks:
        landmark = int(landmark[0]), int(landmark[1])
        image = cv2.circle(image, landmark, 3, (0, 0, 255), -1)

    return image


def draw_triangles_on_image(image, vertices, triangles):
    for triangle in triangles:
        vertex_1, vertex_2, vertex_3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]
        vertex_1 = int(vertex_1[0]), int(vertex_1[1])
        vertex_2 = int(vertex_2[0]), int(vertex_2[1])
        vertex_3 = int(vertex_3[0]), int(vertex_3[1])

        image = cv2.circle(image, vertex_1, 2, (0, 0, 255), -1)
        image = cv2.circle(image, vertex_2, 2, (0, 0, 255), -1)
        image = cv2.circle(image, vertex_3, 2, (0, 0, 255), -1)

        image = cv2.line(image, vertex_1, vertex_2, (0, 0, 255), thickness=1)
        image = cv2.line(image, vertex_2, vertex_3, (0, 0, 255), thickness=1)
        image = cv2.line(image, vertex_3, vertex_1, (0, 0, 255), thickness=1)

    return image

def display_images_with_landmarks(image_list, landmark_list):
    for aligned_image, aligned_landmarks in zip(image_list, landmark_list):
        image = draw_landmarks_on_image(aligned_image, aligned_landmarks)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
