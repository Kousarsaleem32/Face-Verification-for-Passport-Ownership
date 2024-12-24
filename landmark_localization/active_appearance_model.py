import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import procrustes
from scipy.spatial import Delaunay
import morphops as mops
from sklearn.decomposition import PCA
from skimage.transform import PiecewiseAffineTransform, warp
from tqdm import tqdm

from landmark_localization.landmark_utils import draw_landmarks_on_image, display_images_with_landmarks, rescale_image_with_landmarks, draw_triangles_on_image


class ActiveAppearanceModel:
    def __init__(self, train_image_list, train_landmark_list):
        self.build_model(train_image_list, train_landmark_list)

    def build_model(self, train_image_list, train_landmark_list):
        procrustes_aligned_image_list, procrustes_aligned_landmark_list = self.__apply_procrustes_alignment(train_image_list, train_landmark_list, 250, 250)
        for i in range(len(procrustes_aligned_image_list)):
            procrustes_aligned_image_list[i], procrustes_aligned_landmark_list[i] = rescale_image_with_landmarks(procrustes_aligned_image_list[i], procrustes_aligned_landmark_list[i], 0.33)

        # display_images_with_landmarks(procrustes_aligned_image_list, procrustes_aligned_landmark_list)

        self.mean_shape_flattened, self.shape_components = self.get_mean_shape_and_components(procrustes_aligned_landmark_list)

        mean_shape = self.get_a_shape(np.zeros(self.shape_components.shape[0]))

        self.delaunay_triangulated_mean_shape = Delaunay(mean_shape)

        '''jacobian_canvas = np.zeros((250, 250))
        for y in range(250):
            for x in range(250):
                jacobian = self.find_jacobian_at_point((x, y))
                if jacobian is None:
                    continue
                jacobian_canvas[y, x] = jacobian[0, 1]

        jacobian_canvas -= np.min(jacobian_canvas)

        cv2.imshow('jacobian', jacobian_canvas)
        cv2.waitKey(0)

        print('')'''

        warped_image_list = self.warp_images_to_mean_shape(procrustes_aligned_image_list,
                                                           procrustes_aligned_landmark_list)
        # display_images_with_landmarks(warped_image_list, [mean_shape] * len(warped_image_list))
        warped_image_list = np.array(warped_image_list, dtype=np.uint8)

        self.mean_image_flattened, self.image_components_flattened = self.get_mean_image_and_components(
            warped_image_list)

        self.mean_image_dimensions = warped_image_list[0].shape

        mean_image = self.get_mean_image()

        rotated_image_shape_parameters = np.zeros(self.shape_components.shape[0])
        # rotated_image_shape_parameters[1] = 40 # 100
        rotated_image_shape_parameters[1] = 20 # 100

        rotated_image_shape = self.get_a_shape(rotated_image_shape_parameters)

        # rotated_image_shape = procrustes_aligned_landmark_list[0] # [0]
        # rotated_image_shape_parameters = self.shape_pca.transform(rotated_image_shape.flatten().reshape(1, -1))

        print('Real')
        print(rotated_image_shape_parameters)

        rotated_image = self.warp_with_shape_parameters(mean_image, rotated_image_shape_parameters)
        # rotated_image = procrustes_aligned_image_list[0] # [0]

        # rotated_image = cv2.imread(r"C:\Users\Utku\Desktop\messi.jpg", cv2.IMREAD_GRAYSCALE)
        # rotated_image, _ = rescale_image_with_landmarks(rotated_image, np.zeros(rotated_image_shape.shape), 0.33)

        cv2.imshow('mean_image', np.array(mean_image, dtype=np.uint8))
        cv2.waitKey(0)

        cv2.imshow('rotated_image', np.array(rotated_image, dtype=np.uint8))
        cv2.waitKey(0)

        solved_shape_parameters = self.fit_model(np.array(rotated_image, dtype=np.float32), rotated_image_shape)
        solved_shape = self.get_a_shape(solved_shape_parameters)

        # initial_estimation_image = draw_landmarks_on_image(rotated_image, mean_shape)
        # fitted_image = draw_landmarks_on_image(rotated_image, solved_shape)

        rotated_image = cv2.cvtColor(np.array(rotated_image, dtype=np.uint8), cv2.COLOR_GRAY2BGR)

        _, mean_shape_upscaled = rescale_image_with_landmarks(np.zeros(rotated_image.shape), mean_shape, 3)
        rotated_image, solved_shape = rescale_image_with_landmarks(rotated_image, solved_shape, 3)
        _, ground_truth_shape = rescale_image_with_landmarks(np.zeros(rotated_image.shape), rotated_image_shape, 3)

        initial_estimation_image = draw_triangles_on_image(np.array(rotated_image), mean_shape_upscaled, self.delaunay_triangulated_mean_shape.simplices)
        fitted_image = draw_triangles_on_image(np.array(rotated_image), solved_shape, self.delaunay_triangulated_mean_shape.simplices)
        ground_truth_image = draw_triangles_on_image(np.array(rotated_image), ground_truth_shape, self.delaunay_triangulated_mean_shape.simplices)

        cv2.imshow('initial_estimation_image', np.array(initial_estimation_image, dtype=np.uint8))
        cv2.imshow('fitted_image', np.array(fitted_image, dtype=np.uint8))
        cv2.imshow('ground_truth_image', np.array(ground_truth_image, dtype=np.uint8))
        cv2.waitKey(0)


        '''mean_image = self.mean_image_flattened.reshape(warped_image_list[0].shape)
        mean_image = np.array(mean_image, dtype=np.uint8)

        cv2.imshow('mean_image', mean_image)
        cv2.waitKey(0)

        for i, component in enumerate(self.image_components_flattened[:5]):
            component = component - np.min(component)
            component = component * 255 / np.max(component)
            component = component.reshape(warped_image_list[0].shape)
            component = np.array(component, dtype=np.uint8)
            cv2.imshow('Comp ' + str(i), component)
            cv2.waitKey(0)'''


    def __apply_procrustes_alignment(self, image_list, landmark_list, target_width, target_height):
        procrustes_result = mops.gpa(landmark_list)
        aligned_landmark_list = procrustes_result['aligned']
        scaling_factors = procrustes_result['b']

        aligned_landmark_list = aligned_landmark_list / np.mean(scaling_factors)
        x_min, x_max = np.min(aligned_landmark_list[:, :, 0]), np.max(aligned_landmark_list[:, :, 0])
        y_min, y_max = np.min(aligned_landmark_list[:, :, 1]), np.max(aligned_landmark_list[:, :, 1])
        face_width_max, face_height_max = x_max - x_min, y_max - y_min
        left_space, top_space = (target_width - face_width_max) / 2, (target_height - face_height_max) / 2

        print(f'{x_min}, {x_max}')
        print(f'{y_min}, {y_max}')
        aligned_landmark_list = aligned_landmark_list - (x_min - left_space, y_min - top_space)

        aligned_image_list = []
        for image, original_landmarks, aligned_landmarks in zip(image_list, landmark_list, aligned_landmark_list):
            transformation_matrix, inliers = cv2.estimateAffine2D(np.array(original_landmarks, dtype=np.int32), aligned_landmarks)
            image = cv2.warpAffine(image, transformation_matrix, (target_width, target_height))
            aligned_image_list.append(image)
            #aligned_image_list.append(draw_landmarks_on_image(image, aligned_landmarks))

        return aligned_image_list, aligned_landmark_list

    def get_mean_shape_and_components(self, landmark_list):
        n_img, n_lm, n_dim = landmark_list.shape
        pca_ready_landmark_list = landmark_list.reshape(n_img, n_lm * n_dim)

        shape_pca = PCA()
        shape_pca.fit(pca_ready_landmark_list)
        self.shape_pca = shape_pca
        return shape_pca.mean_, shape_pca.components_

    def get_a_shape(self, shape_parameters):
        n_components, n_lm_times_dim = self.shape_components.shape
        shape = self.mean_shape_flattened + shape_parameters.reshape(1, n_components) @ self.shape_components
        shape = shape.reshape(n_lm_times_dim // 2, 2)
        return shape

    def warp_images_to_mean_shape(self, image_list, landmark_list):
        mean_shape = self.get_a_shape(np.zeros(len(self.shape_components)))
        warped_image_list = []
        piecewise_affine_transform = PiecewiseAffineTransform()
        for aligned_image, landmarks in tqdm(zip(image_list, landmark_list), total=len(image_list)):
            piecewise_affine_transform.estimate(mean_shape, landmarks)
            warped_image = warp(aligned_image, piecewise_affine_transform, output_shape=aligned_image.shape) * 255
            #warped_image, _ = rescale_image_with_landmarks(warped_image, landmarks, 0.5)
            warped_image_list.append(warped_image)

        return warped_image_list

    def get_mean_image_and_components(self, image_list):
        n_img, height, width = image_list.shape
        warped_image_list = image_list.reshape(n_img, height * width)

        image_pca = PCA()
        image_pca.fit(warped_image_list)
        return image_pca.mean_, image_pca.components_
        # mean_image = mean_image.reshape(height, width, n_ch)
        # mean_image = np.array(mean_image, dtype=np.uint8)

    def get_mean_image(self):
        return self.mean_image_flattened.reshape(self.mean_image_dimensions)

    def warp_with_shape_parameters(self, image, shape_parameters):
        # TODO: Implement
        mean_shape = self.get_a_shape(np.zeros(shape_parameters.shape))

        new_shape = self.get_a_shape(shape_parameters)
        piecewise_affine_transform = PiecewiseAffineTransform()
        piecewise_affine_transform.estimate(new_shape, mean_shape)
        warped_image = warp(image, piecewise_affine_transform, output_shape=image.shape)
        return warped_image

    def get_image_gradients(self, image):
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        # gradient_x = (gradient_x - np.mean(gradient_x)) / np.std(gradient_x) * 255
        # gradient_y = (gradient_y - np.mean(gradient_y)) / np.std(gradient_y) * 255

        return gradient_x, gradient_y

    def get_barycentric_coordinates(self, point, vertex_1, vertex_2, vertex_3):
        x, y = point
        x_i, y_i = vertex_1
        x_j, y_j = vertex_2
        x_k, y_k = vertex_3

        alpha_num = (x - x_i) * (y_k - y_i) - (y - y_i) * (x_k - x_i)
        beta_num = (y - y_i) * (x_j - x_i) - (x - x_i) * (y_j - y_i)

        denum = (x_j - x_i) * (y_k - y_i) - (y_j - y_i) * (x_k - x_i)

        alpha = alpha_num / denum
        beta = beta_num / denum

        return alpha, beta

    def find_jacobian_at_point(self, point, delaunay_triangulated_mean_shape):
        # triangle_index = self.delaunay_triangulated_mean_shape.find_simplex(point)[0]
        # triangle_index = int(self.delaunay_triangulated_mean_shape.find_simplex(point))
        triangle_index = int(delaunay_triangulated_mean_shape.find_simplex(point))
        if triangle_index < 0:
            return None

        mean_shape = self.get_a_shape(np.zeros(self.shape_components.shape[0]))
        #triangle_vertex_indices = self.delaunay_triangulated_mean_shape.simplices[triangle_index]
        triangle_vertex_indices = delaunay_triangulated_mean_shape.simplices[triangle_index]

        jacobian = 0

        for i in range(3):
            # vertex_1, vertex_2, vertex_3 = i, (i + 1) % 3, (i + 2) % 3 # TODO: These are not vertex indices, these are the indices of vertices in triangle_vertex_indices array, update accordingly.
            vertex_1_index, vertex_2_index, vertex_3_index = triangle_vertex_indices[i], triangle_vertex_indices[(i + 1) % 3], triangle_vertex_indices[(i + 2) % 3]
            vertex_1, vertex_2, vertex_3 = mean_shape[vertex_1_index], mean_shape[vertex_2_index], mean_shape[vertex_3_index]
            alpha, beta = self.get_barycentric_coordinates(point, vertex_1, vertex_2, vertex_3)
            dw_dx_i = np.array((1 - alpha - beta, 0)).reshape((2, 1))
            dw_dy_i = np.array((0, 1 - alpha - beta)).reshape((2, 1))

            dx_i_dp = self.shape_components[:, vertex_1_index * 2]
            dy_i_dp = self.shape_components[:, vertex_1_index * 2 + 1]

            dx_i_dp, dy_i_dp = dx_i_dp.reshape((1, dx_i_dp.shape[0])), dy_i_dp.reshape((1, dy_i_dp.shape[0]))

            jacobian += dw_dx_i @ dx_i_dp + dw_dy_i @ dy_i_dp

        return jacobian

    def fit_model(self, image, my_shape):
        mean_image = self.get_mean_image()
        # error_image = mean_image - self.warp_with_shape_parameters(image)
        # total_error = np.sum(np.square(error_image))
        image_error = []
        landmark_error = []
        delta_p_values = []


        # my_shape_parameters = np.zeros(self.shape_components.shape[0])
        # my_shape_parameters[1] = 40  # 100
        #my_shape = self.get_a_shape(my_shape_parameters)

        # while total_error > some_threshold:

        shape_parameters = np.zeros(self.shape_components.shape[0])

        mean_shape = self.get_a_shape(np.zeros(shape_parameters.shape))
        piecewise_affine_transform = PiecewiseAffineTransform()

        for i in tqdm(range(200)):
            delta_p = 0
            # warped_image = self.warp_with_shape_parameters(image, shape_parameters)
            shape_to_warp = self.get_a_shape(shape_parameters)
            piecewise_affine_transform.estimate(mean_shape, shape_to_warp)
            warped_image = warp(image, piecewise_affine_transform, output_shape=image.shape)
            image_gradient_x, image_gradient_y = self.get_image_gradients(warped_image)
            current_shape = self.get_a_shape(shape_parameters)
            delaunay_triangulated_current_shape = Delaunay(current_shape)

            jacobians_per_pixel = np.zeros((mean_image.shape[0], mean_image.shape[1], 2, self.shape_components.shape[0]))
            pixel_in_base_mesh = np.zeros((mean_image.shape[0], mean_image.shape[1]))

            for y in range(mean_image.shape[0]):
                for x in range(mean_image.shape[1]):
                    jacobian = self.find_jacobian_at_point((x, y), delaunay_triangulated_current_shape)
                    if jacobian is None:
                        continue

                    jacobians_per_pixel[y, x] = jacobian
                    pixel_in_base_mesh[y, x] = 1

            hessian = 0
            for y in range(mean_image.shape[0]):
                for x in range(mean_image.shape[1]):
                    image_gradient = np.array((image_gradient_x[y, x], image_gradient_y[y, x])).reshape((1, 2))
                    # jacobian = self.find_jacobian_at_point((x, y), delaunay_triangulated_current_shape)
                    jacobian = jacobians_per_pixel[y, x]
                    # if jacobian is None:
                    if pixel_in_base_mesh[y, x] < 1:
                        continue
                    sd = image_gradient @ jacobian
                    hessian += sd.T @ sd

            for y in range(mean_image.shape[0]):
                for x in range(mean_image.shape[1]):
                    # continue if the point (x, y) does not inside the convex hull
                    difference = mean_image[y, x] - warped_image[y, x]
                    image_gradient = np.array((image_gradient_x[y, x], image_gradient_y[y, x])).reshape((1, 2))
                    # jacobian = self.find_jacobian_at_point((x, y), delaunay_triangulated_current_shape)
                    jacobian = jacobians_per_pixel[y, x]
                    # if jacobian is None:
                    if pixel_in_base_mesh[y, x] < 1:
                        continue
                    sd = image_gradient @ jacobian
                    delta_p += sd.T * difference

            delta_p = np.linalg.pinv(hessian) @ delta_p
            shape_parameters += delta_p[:, 0]

            solved_shape = self.get_a_shape(shape_parameters)
            image_error.append(np.sum(np.square(mean_image - warped_image)))
            landmark_error.append(np.sum(np.square(solved_shape - my_shape)))
            delta_p_values.append(np.sum(np.absolute(delta_p)))

        print('Solved')
        print(shape_parameters)
        plt.plot(np.sqrt(np.array(image_error, dtype=np.float32) / len(image_error)))
        plt.show()
        plt.plot(np.sqrt(np.array(landmark_error, dtype=np.float32) / len(landmark_error)))
        plt.show()
        plt.plot(np.array(delta_p_values))
        plt.show()
        return shape_parameters

