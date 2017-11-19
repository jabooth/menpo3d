import numpy as np

from menpo.visualize import print_dynamic
from menpo3d.morphablemodel.result import MMAlgorithmResult

from ..derivatives import (d_camera_d_camera_parameters,
                           d_camera_d_shape_parameters)
from .base import (camera_parameters_update, gradient_xy, J_lms, LucasKanade,
                   sample_at_bc_vi, visible_sample_points)


def J_data(camera, warped_uv, shape_pc_uv, texture_pc_uv, grad_x_uv,
           grad_y_uv, camera_update, focal_length_update,
           reconstruction_prior_weight):
    # Compute derivative of camera wrt shape and camera parameters
    dp_da_dr = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)
    n_camera_parameters = 0
    if camera_update:
        dp_dr = d_camera_d_camera_parameters(
            camera, warped_uv, with_focal_length=focal_length_update)
        dp_da_dr = np.hstack((dp_da_dr, dp_dr))
        n_camera_parameters = dp_dr.shape[1]

    # Multiply image gradient with camera derivative
    J = grad_x_uv[:, None] * dp_da_dr[0] + grad_y_uv[:, None] * dp_da_dr[1]

    # Computer derivative of texture wrt texture parameters
    dt_db = - np.rollaxis(texture_pc_uv, 0, 3)

    # Concatenate to create the data term steepest descent
    J = np.hstack((J, dt_db))

    # Reshape to : n_params x (2 * N)
    n_params = J.shape[1]
    J = np.transpose(J, (1, 0, 2)).reshape(n_params, -1)
    return reconstruction_prior_weight * J, n_camera_parameters


def solve(hessian, sd_error, reconstruction_prior_weight, n, m,
          camera_update, focal_length_update, camera):
    # Solve
    ds = - np.linalg.solve(hessian, sd_error)

    # Get shape parameters increment
    d_shape = ds[:n]

    # Initialize texture parameters update
    d_texture = np.zeros(m)

    # Get camera parameters increment
    if camera_update:
        # Keep the rest
        ds = ds[n:]

        # If focal length is not updated, then set its increment to zero
        if not focal_length_update:
            ds = np.insert(ds, 0, [0.])

        # Set increment of the 1st quaternion to one
        ds = np.insert(ds, 1, [1.])

        # Get camera parameters update
        d_camera = ds[:camera.n_parameters]

        # Get texture parameters increment
        if reconstruction_prior_weight is not None:
            d_texture = ds[camera.n_parameters:]
    else:
        d_camera = None
        if reconstruction_prior_weight is not None:
            d_texture = ds[n:]

    return d_shape, d_camera, d_texture


class SimultaneousForwardAdditive(LucasKanade):
    r"""
    Simultaneous Forward Additive Morphable Model optimization algorithm.
    """
    def __str__(self):
        return "Simultaneous Forward Additive"

    def _precompute(self):
        # Rescale shape and appearance components to have size:
        # n_vertices x (n_active_components * n_dims)
        shape_pc = self.model.shape_model.components.T
        self.shape_pc = shape_pc.reshape([self.n_vertices, -1])

        # Priors
        self.J_shape_prior = 1. / np.array(self.model.shape_model.eigenvalues)
        self.J_texture_prior = 1. / np.array(self.model.texture_model.eigenvalues)
        self.shape_pc_lms = shape_pc.reshape([self.n_vertices, 3, -1])[
            self.model.model_landmarks_index]

    def run(self, image, initial_mesh, camera, gt_mesh=None, max_iters=20,
            camera_update=False, focal_length_update=False,
            reconstruction_weight=1., shape_prior_weight=1.,
            texture_prior_weight=1., landmarks=None, landmarks_prior_weight=1.,
            return_costs=False, verbose=True):
        # Parse landmarks prior options
        if landmarks is None or landmarks_prior_weight is None:
            landmarks_prior_weight = None
            landmarks = None
        lms_points = None
        if landmarks is not None:
            lms_points = landmarks.points[:, [1, 0]]

        # Retrieve camera parameters from the provided camera object.
        # Project provided instance to retrieve shape and texture parameters.
        camera_parameters = camera.as_vector()
        shape_parameters = self.model.shape_model.project(initial_mesh)
        texture_parameters = self.model.project_instance_on_texture_model(
            initial_mesh)

        # Reconstruct provided instance
        instance = self.model.instance(shape_weights=shape_parameters,
                                       texture_weights=texture_parameters)

        # Compute input image gradient
        grad_x, grad_y = gradient_xy(image)

        # Initialize lists
        shape_parameters_per_iter = [shape_parameters]
        texture_parameters_per_iter = [texture_parameters]
        camera_per_iter = [camera]
        instance_per_iter = [instance.rescale_texture(0., 1.)]
        costs = None
        if return_costs:
            costs = []

        # Initialize iteration counter and epsilon
        k = 0
        eps = np.Inf

        # Main loop
        while k < max_iters and eps > self.eps:
            if verbose:
                print_dynamic("{}/{}".format(k + 1, max_iters))
            # Apply camera projection on current instance
            instance_in_image = camera.apply(instance)

            # Compute indices locations for sampling
            (vert_indices, bcoords, tri_indices,
             yx) = visible_sample_points(instance_in_image, image.shape,
                                         self.n_samples)

            # Warp the mesh with the view matrix (rotation + translation)
            instance_w = camera.view_transform.apply(instance.points)

            # Sample all the terms from the model part at the sample locations
            warped_uv = sample_at_bc_vi(instance_w, bcoords, vert_indices)
            texture_uv = instance.sample_texture_with_barycentric_coordinates(
                bcoords, tri_indices)
            texture_pc_uv = self.model.sample_texture_model(bcoords,
                                                            tri_indices)
            shape_pc_uv = sample_at_bc_vi(self.shape_pc, bcoords, vert_indices)
            # Reshape shape basis after sampling
            shape_pc_uv = shape_pc_uv.reshape([self.n_samples, 3, -1])

            # Sample all the terms from the image part at the sample locations
            img_uv = image.sample(yx)
            grad_x_uv = grad_x.sample(yx)
            grad_y_uv = grad_y.sample(yx)

            # Compute error
            img_error_uv = (img_uv - texture_uv.T).ravel()

            # Compute Jacobian, SD and Hessian of data term
            if reconstruction_weight is not None:
                sd, n_camera_parameters = J_data(
                    camera, warped_uv, shape_pc_uv, texture_pc_uv, grad_x_uv,
                    grad_y_uv, camera_update, focal_length_update,
                    reconstruction_weight)
                hessian = sd.dot(sd.T)
                sd_error = sd.dot(img_error_uv)
            else:
                n_camera_parameters = 0
                if camera_update:
                    if focal_length_update:
                        n_camera_parameters = camera.n_parameters - 1
                    else:
                        n_camera_parameters = camera.n_parameters - 2
                hessian = np.zeros((self.n+n_camera_parameters,
                                    self.n+n_camera_parameters))
                sd_error = np.zeros(self.n+n_camera_parameters)

            # Compute Jacobian, update SD and Hessian wrt shape prior
            if shape_prior_weight is not None:
                sd_shape = shape_prior_weight * self.J_shape_prior
                hessian[:self.n, :self.n] += np.diag(sd_shape)
                sd_error[:self.n] += sd_shape * shape_parameters

            # Compute Jacobian, update SD and Hessian wrt texture prior
            if texture_prior_weight is not None:
                idx = self.n + n_camera_parameters
                sd_texture = texture_prior_weight * self.J_texture_prior
                hessian[idx:, idx:] += np.diag(sd_texture)
                sd_error[idx:] += sd_texture * texture_parameters

            # Compute Jacobian, update SD and Hessian wrt landmarks prior
            lms_error = None
            if landmarks_prior_weight is not None:
                # Get projected instance on landmarks and error term
                warped_lms = instance_in_image.points[
                    self.model.model_landmarks_index]
                lms_error = (warped_lms[:, [1, 0]] - lms_points).T.ravel()
                warped_view_lms = instance_w[self.model.model_landmarks_index]

                # Jacobian and Hessian wrt shape parameters
                sd_lms, n_camera_parameters = J_lms(
                    camera, warped_view_lms, self.shape_pc_lms, camera_update,
                    focal_length_update)
                idx = self.n + n_camera_parameters
                hessian[:idx, :idx] += (landmarks_prior_weight *
                                        sd_lms.dot(sd_lms.T))
                sd_error[:idx] = landmarks_prior_weight * sd_lms.dot(lms_error)

            if return_costs:
                costs.append(self.compute_cost(
                    img_error_uv, lms_error, shape_parameters,
                    texture_parameters, shape_prior_weight,
                    texture_prior_weight, landmarks_prior_weight))

            # Solve to find the increment of parameters
            d_shape, d_camera, d_texture = solve(
                hessian, sd_error, reconstruction_weight, self.n,
                self.m, camera_update, focal_length_update, camera)

            # Update parameters
            shape_parameters += d_shape
            if camera_update:
                camera_parameters = camera_parameters_update(
                    camera_parameters, d_camera)
                camera = camera.from_vector(camera_parameters)
            texture_parameters += d_texture

            # Generate the updated instance
            instance = self.model.instance(shape_weights=shape_parameters,
                                           texture_weights=texture_parameters)

            # Update lists
            shape_parameters_per_iter.append(shape_parameters.copy())
            texture_parameters_per_iter.append(texture_parameters.copy())
            camera_per_iter.append(camera)
            instance_per_iter.append(instance.rescale_texture(0., 1.))

            # Increase iteration counter
            k += 1

            # shape_parameters, texture_parameters, camera_parameters = yield \
            #     shape_parameters, texture_parameters, camera_parameters

        return MMAlgorithmResult(
            shape_parameters=shape_parameters_per_iter,
            texture_parameters=texture_parameters_per_iter,
            meshes=instance_per_iter, camera_transforms=camera_per_iter,
            image=image, initial_mesh=initial_mesh.rescale_texture(0., 1.),
            initial_camera_transform=camera_per_iter[0], gt_mesh=gt_mesh,
            costs=costs)

    def compute_cost(self, data_error, lms_error, shape_parameters,
                     texture_parameters, shape_prior_weight,
                     texture_prior_weight, landmarks_prior_weight):

        # Cost of data term
        data_cost = data_error.T.dot(data_error)
        # Cost of shape prior
        if shape_prior_weight is not None:
            # print('shape_prior: {}'.format(shape_prior_weight))
            shape_cost = (shape_prior_weight *
                          np.sum((shape_parameters ** 2) * self.J_shape_prior))
        else:
            # print('Warning - no shape prior weight')
            shape_cost = 0

        # Cost of texture prior
        if texture_prior_weight is not None:
            # print('texture_prior: {}'.format(texture_prior_weight))
            # print('texture_parameters: {}'.format(texture_parameters))
            # print('J_texture_prior: {}'.format(self.J_texture_prior))
            texture_cost = (texture_prior_weight *
                            np.sum((texture_parameters ** 2) *
                                   self.J_texture_prior))
        else:
            # print('Warning - no texture prior weight')
            texture_cost = 0

        # Cost of landmarks prior
        if landmarks_prior_weight is not None:
            # print('landmarks_prior: {}'.format(landmarks_prior_weight))
            landmarks_cost = landmarks_prior_weight * lms_error.T.dot(lms_error)
        else:
            # print('Warning - no landmarks prior weight')
            landmarks_cost = 0

        total_cost = data_cost + shape_cost + texture_cost + landmarks_cost

        # print('COST: {}\n    Data: {}\n    Shape:scost, landmarks_cost))
        return total_cost
