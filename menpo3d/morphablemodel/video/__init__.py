import numpy as np
from menpo.visualize import print_progress

from ..algorithm.derivatives import (d_camera_d_shape_parameters,
                                     d_camera_d_camera_parameters)
from ..algorithm.lk import camera_parameters_update
from ..algorithm.lk.base import (gradient_xy, sample_at_bc_vi,
                                 visible_sample_points)
from ..algorithm.lk.projectout import project_out


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
              image, lms_points, model, id_ind, exp_ind, camera,
              grad_x, grad_y, shape_pc, shape_pc_lms, n_samples):

    instance = model.shape_model.instance(shape_parameters)
    camera = camera.from_vector(camera_parameters)
    instance_in_image = camera.apply(instance)
    m = model.texture_model.n_active_components

    # Compute indices locations for sampling
    (vertex_indices, bcoords, tri_indices,
     yx) = visible_sample_points(instance_in_image, image.shape, n_samples)

    # Warp the mesh with the view matrix (rotation + translation)
    instance_w = camera.view_transform.apply(instance.points)

    # Sample all the terms from the model part at the sample locations
    warped_uv = sample_at_bc_vi(instance_w, bcoords, vertex_indices)
    texture_pc_uv = model.sample_texture_model(bcoords,
                                               tri_indices).reshape((-1, m))
    m_texture_uv = (
        model.instance().sample_texture_with_barycentric_coordinates(
            bcoords, tri_indices).T)

    # Reshape shape basis after sampling
    # shape_pc_uv: (n_samples, xyz, shape_components)
    shape_pc_uv = sample_at_bc_vi(shape_pc, bcoords, vertex_indices).reshape([n_samples,
                                                                              3, -1])

    # Sample all the terms from the image part at the sample locations
    # img_uv: (channels, samples)
    img_uv = image.sample(yx)
    grad_x_uv = grad_x.sample(yx)
    grad_y_uv = grad_y.sample(yx)

    # Compute error
    # img_error_uv: (channels x samples,)
    img_error_uv = (img_uv - m_texture_uv).ravel()

    # Data term Jacobian
    sd, n_camera_parameters = J_data(
        camera, warped_uv, shape_pc_uv, texture_pc_uv, grad_x_uv,
        grad_y_uv)

    # Landmarks term Jacobian
    # Get projected instance on landmarks and error term
    warped_lms = instance_in_image.points[model.model_landmarks_index]
    lms_error = (warped_lms[:, [1, 0]] - lms_points).T.ravel()
    warped_view_lms = instance_w[model.model_landmarks_index]
    sd_lms, n_camera_parameters = J_lms(
        camera, warped_view_lms, shape_pc_lms)

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


def single_iteration_update(shape_parameters, camera_parameters,
                            image, lms_points, model, camera,
                            grad_x, grad_y, shape_pc, shape_pc_lms, n_samples):

    j = jacobians(shape_parameters, camera_parameters, image, lms_points,
                  model, camera, grad_x, grad_y, shape_pc, shape_pc_lms,
                  n_samples)

    sd = j['J_f'].T * reconstruction_weight
    img_error_uv = j['e_d']
    n_camera_parameters = j['n_camera_parameters']

    hessian = sd.dot(sd.T)
    # consider img_errror_uv could be -1 * to match Tassos maths in
    # new paper (and we remove the -1 in solve) maybe this is
    # the cost issue?
    sd_error = sd.dot(img_error_uv)

    # Compute Jacobian, update SD and Hessian wrt shape prior
    sd_shape = shape_prior_weight * J_shape_prior

    hessian[:self.n, :self.n] += np.diag(sd_shape)
    sd_error[:self.n] += sd_shape * shape_parameters

    # Compute Jacobian, update SD and Hessian wrt landmarks prior

    # Get projected instance on landmarks and error term
    warped_lms = instance_in_image.points[model.model_landmarks_index]
    lms_error = (warped_lms[:, [1, 0]] - lms_points).T.ravel()
    warped_view_lms = instance_w[model.model_landmarks_index]
    sd_lms, n_camera_parameters = J_lms(
        camera, warped_view_lms, self.shape_pc_lms, camera_update,
        focal_length_update)

    idx = self.n + n_camera_parameters
    hessian[:idx, :idx] += (landmarks_prior_weight *
                            sd_lms.dot(sd_lms.T))
    sd_error[:idx] = landmarks_prior_weight * sd_lms.dot(lms_error)



    # Solve
    ds = - np.linalg.solve(hessian, sd_error)

    # Get shape parameters increment
    d_shape = ds[:self.n]

    # Get camera parameters increment
    if camera_update:
        # Keep the rest
        ds = ds[self.n:]

        # If focal length is not updated, then set its increment to zero
        if not focal_length_update:
            ds = np.insert(ds, 0, [0.])

        # Set increment of the 1st quaternion to one
        ds = np.insert(ds, 1, [1.])

        # Get camera parameters update
        d_camera = ds
    else:
        d_camera = None

    # Update parameters
    shape_parameters += d_shape
    camera_parameters = camera_parameters_update(camera_parameters,
                                                 d_camera)

    return shape_parameters, camera_parameters


def fit_video(frames, cameras, shape_parameters,
              mm, id_indices, exp_indices,
              reconstruction_weight=1., shape_prior_weight=1.,
              landmarks_prior_weight=1.,
              n_iters=20, verbose=False, n_samples=1000):

    n_points = mm.shape_model.template_instance.n_points

    # Rescale shape components to have size:
    # n_points x (n_components * n_dims)
    shape_pc = mm.shape_model.components.T.reshape([n_points, -1])
    shape_pc_lms = shape_pc.reshape([n_points, 3, -1])[mm.model_landmarks_index]

    # Priors
    J_shape_prior = 1. / np.array(mm.shape_model.eigenvalues)

    image = frames[0]
    camera = cameras[0]
    camera_parameters = camera.as_vector()

    # TODO fix this two wrongs make a right
    lms_points = image.landmarks[None].points[:, [1, 0]]

    # Compute input image gradient
    grad_x, grad_y = gradient_xy(image)

    for _ in print_progress(list(range(n_iters))):
        shape_parameters, camera_parameters = single_iteration_update(
            shape_parameters, camera_parameters, image, lms_points, mm,
            camera, grad_x, grad_y, shape_pc, shape_pc_lms, n_samples)
