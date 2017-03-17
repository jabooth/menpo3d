import numpy as np
import scipy.sparse as sp
from menpo.visualize import print_progress

from ..algorithm.derivatives import (d_camera_d_shape_parameters,
                                     d_camera_d_camera_parameters)
from ..algorithm.lk import camera_parameters_update
from ..algorithm.lk.base import gradient_xy
from ..algorithm.lk.projectout import project_out, sample_uv_terms
from .jacobian import (initialize_jacobian_and_error,
                       insert_frame_to_e, insert_frame_to_J)


def J_data(camera, warped_uv, shape_pc_uv, texture_pc_uv, grad_x_uv,
           grad_y_uv, focal_length_update=False):
    # Compute derivative of camera wrt shape and camera parameters
    dp_da_dr = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)

    dp_dr = d_camera_d_camera_parameters(
        camera, warped_uv, with_focal_length=focal_length_update)

    # stack the shape_parameters/camera_parameters updates
    dp_da_dr = np.hstack((dp_da_dr, dp_dr))
    n_camera_parameters = dp_dr.shape[1]

    # Multiply image gradient with camera derivative
    permuted_grad_x = np.transpose(grad_x_uv[..., None], (0, 2, 1))
    permuted_grad_y = np.transpose(grad_y_uv[..., None], (0, 2, 1))
    J = permuted_grad_x * dp_da_dr[0] + permuted_grad_y * dp_da_dr[1]

    # Project-out
    n_params = J.shape[1]
    J = np.transpose(J, (1, 0, 2)).reshape(n_params, -1)
    PJ = project_out(J, texture_pc_uv)

    return PJ, n_camera_parameters


def J_lms(camera, warped_uv, shape_pc_uv, focal_length_update=False):
    # Compute derivative of camera wrt shape and camera parameters
    J = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)
    dp_dr = d_camera_d_camera_parameters(
        camera, warped_uv, with_focal_length=focal_length_update)
    J = np.hstack((J, dp_dr))
    n_camera_parameters = dp_dr.shape[1]

    # Reshape to : n_params x (2 * N)
    n_params = J.shape[1]
    J = np.transpose(J, (1, 0, 2)).reshape(n_params, -1)
    return J, n_camera_parameters


def jacobians(shape_parameters, camera_parameters,
              image, lms_points, mm, id_ind, exp_ind, camera,
              grad_x, grad_y, shape_pc, shape_pc_lms, n_samples):

    instance = mm.shape_model.instance(shape_parameters)
    camera = camera.from_vector(camera_parameters)

    (instance_w, instance_in_image, warped_uv, img_error_uv,
     shape_pc_uv, texture_pc_uv, grad_x_uv, grad_y_uv
     ) = sample_uv_terms(instance, image, camera, mm, shape_pc,
                         grad_x, grad_y, n_samples)

    # Data term Jacobian
    sd, n_camera_parameters = J_data(camera, warped_uv, shape_pc_uv,
                                     texture_pc_uv, grad_x_uv, grad_y_uv)

    # Landmarks term Jacobian
    # Get projected instance on landmarks and error term
    warped_lms = instance_in_image.points[mm.model_landmarks_index]
    lms_error = (warped_lms[:, [1, 0]] - lms_points).T.ravel()
    warped_view_lms = instance_w[mm.model_landmarks_index]
    sd_lms, n_camera_parameters = J_lms(camera, warped_view_lms, shape_pc_lms)

    cam_n_params = camera_parameters.shape[0]
    # form the main two Jacobians...
    J_f = sd.T
    J_l = sd_lms.T
    # and then slice at the appropriate indices to break down by param type.
    return {
        'J_f_p': J_f[:, id_ind],
        'J_f_q': J_f[:, exp_ind],
        'J_f_c': J_f[:, -cam_n_params:],

        'J_l_p': J_l[:, id_ind],
        'J_l_q': J_l[:, exp_ind],
        'J_l_c': J_l[:, -cam_n_params:],

        'e_f': img_error_uv,
        'e_l': lms_error
    }


def fit_video(images, cameras, mm, id_indices, exp_indices, p, qs,
              c_id=1., c_l=1., c_exp=1., c_sm=1., n_samples=1000):

    n_frames = len(images)
    n_lms = images[0].landmarks[None].n_points
    n_channels = images[0].n_channels
    n_p = len(id_indices)
    n_q = len(exp_indices)
    n_c = cameras[0].n_parameters

    n_pixels_per_frame = n_channels * n_samples
    n_sites_per_frame = n_pixels_per_frame + (2 * n_lms)

    n_points = mm.shape_model.template_instance.n_points

    print('Precomputing....')
    # Rescale shape components to have size:
    # n_points x (n_components * n_dims)
    shape_pc = mm.shape_model.components.T.reshape([n_points, -1])
    shape_pc_lms = shape_pc.reshape([n_points, 3, -1])[mm.model_landmarks_index]

    print('Initializing Jacobian for frame....')
    J, e = initialize_jacobian_and_error(c_id, c_exp, c_sm, n_p, n_q, n_c, p,
                                         qs, n_sites_per_frame, n_frames)
    print('J.shape: {}'.format(J.shape))

    for (i, image), camera, q in zip(enumerate(print_progress(images)),
                                     cameras, qs):
        camera_params = camera.as_vector()

        shape_params = np.zeros(mm.shape_model.n_active_components)
        shape_params[id_indices] = p
        shape_params[exp_indices] = q

        # TODO fix this two wrongs make a right
        lms_points = image.landmarks[None].points[:, [1, 0]]

        # Compute input image gradient
        grad_x, grad_y = gradient_xy(image)

        j = jacobians(shape_params, camera_params, image, lms_points,
                      mm, id_indices, exp_indices, camera,
                      grad_x, grad_y,
                      shape_pc, shape_pc_lms, n_samples)
        insert_frame_to_J(J, j, i, c_l, n_p, n_q, n_pixels_per_frame, n_frames)
        insert_frame_to_e(e, j, i, n_sites_per_frame)
    print('Converting J to efficient format...')
    J = J.tocsr()
    print('Calculating H = J.T.dot(J)...')
    H = J.T.dot(J)
    print('Calculating J.T.dot(e)...')
    J_T_e = J.T.dot(e)

    print("Sparsity (prop. 0's) of H: {:.2%}".format(
        1 - (H.count_nonzero() / np.prod(np.array(H.shape)))))
    print('Solving for parameter update')
    dp = sp.linalg.spsolve(H, J_T_e)
    return locals(), dp
