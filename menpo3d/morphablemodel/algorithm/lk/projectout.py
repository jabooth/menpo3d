import numpy as np

from menpo.visualize import print_dynamic
from menpo3d.morphablemodel.result import MMAlgorithmResult

from ..derivatives import (d_camera_d_camera_parameters,
                           d_camera_d_shape_parameters)
from .base import (camera_parameters_update, gradient_xy, J_lms, LucasKanade,
                   sample_at_bc_vi, visible_sample_points)


def J_data(camera, warped_uv, shape_pc_uv, texture_pc_uv, grad_x_uv,
           grad_y_uv, camera_update, focal_length_update):
    # Compute derivative of camera wrt shape and camera parameters
    dp_da_dr = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)
    n_camera_parameters = 0
    if camera_update:
        dp_dr = d_camera_d_camera_parameters(
            camera, warped_uv, with_focal_length=focal_length_update)
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

    # Concatenate to create the data term steepest descent
    return PJ, n_camera_parameters


def project_out(J, U):
    tmp = J.dot(U)
    return J - tmp.dot(U.T)


def solve(hessian, sd_error, n, camera_update, focal_length_update):
    # Solve
    ds = - np.linalg.solve(hessian, sd_error)

    # Get shape parameters increment
    d_shape = ds[:n]

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
        d_camera = ds
    else:
        d_camera = None

    return d_shape, d_camera


def sample_uv_terms(instance, image, camera, mm, shape_pc, grad_x, grad_y,
                    n_samples):
    # subsample all the terms we need to compute a project out update.

    # Apply camera projection on current instance
    instance_in_image = camera.apply(instance)

    # Compute indices locations for sampling
    (vert_indices, bcoords, tri_indices,
     yx) = visible_sample_points(instance_in_image, image.shape, n_samples)

    # Warp the mesh with the view matrix (rotation + translation)
    instance_w = camera.view_transform.apply(instance.points)

    # Sample all the terms from the model part at the sample locations
    warped_uv = sample_at_bc_vi(instance_w, bcoords, vert_indices)

    # n_samples x n_channels x n_texture_comps
    texture_pc_uv = mm.sample_texture_model(bcoords, tri_indices)

    # (n_samples * n_channels) x n_texture_comps
    texture_pc_uv = texture_pc_uv.reshape((-1, texture_pc_uv.shape[-1]))

    # n_channels x n_samples
    m_texture_uv = mm.instance().sample_texture_with_barycentric_coordinates(
        bcoords, tri_indices).T

    # n_samples x 3 x n_shape_components
    shape_pc_uv = (sample_at_bc_vi(shape_pc, bcoords, vert_indices)
                   .reshape([n_samples, 3, -1]))

    # Sample all the terms from the image part at the sample locations
    # img_uv: (channels, samples)
    img_uv = image.sample(yx)
    grad_x_uv = grad_x.sample(yx)
    grad_y_uv = grad_y.sample(yx)

    # Compute error
    # img_error_uv: (channels x samples,)
    img_error_uv = (img_uv - m_texture_uv).ravel()

    return (instance_w, instance_in_image, warped_uv, img_error_uv,
            shape_pc_uv, texture_pc_uv, grad_x_uv, grad_y_uv)


class ProjectOutForwardAdditive(LucasKanade):
    r"""
    Project Out Forward Additive Morphable Model optimization algorithm.
    """
    def __str__(self):
        return "Project Out Forward Additive"

    def _precompute(self):
        # call super method
        super(ProjectOutForwardAdditive, self)._precompute()
        self.texture_T = self.model.texture_model.components.T

    def run(self, image, initial_mesh, camera, landmarks=None,
            camera_update=False, focal_length_update=False,
            reconstruction_weight=1., shape_prior_weight=1.,
            texture_prior_weight=1.,  landmarks_prior_weight=1.,
            gt_mesh=None, max_iters=20, return_costs=False, verbose=True):

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

            (instance_w, instance_in_image, warped_uv, img_error_uv,
             shape_pc_uv, texture_pc_uv, grad_x_uv, grad_y_uv
             ) = sample_uv_terms(
                instance, image, camera, self.model, self.shape_pc,
                grad_x, grad_y, self.n_samples)

            # Compute Jacobian, SD and Hessian of data term
            if reconstruction_weight is not None:
                sd, n_camera_parameters = J_data(
                    camera, warped_uv, shape_pc_uv, texture_pc_uv, grad_x_uv,
                    grad_y_uv, camera_update, focal_length_update)
                # TODO this doesn't seem balanced with the other weights?
                sd *= reconstruction_weight
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
                texture_parameters = np.linalg.lstsq(texture_pc_uv,
                                                     img_error_uv)[0]
                costs.append(self.compute_cost(
                    img_error_uv, lms_error, shape_parameters,
                    texture_parameters, shape_prior_weight,
                    texture_prior_weight, landmarks_prior_weight))

            # Solve to find the increment of parameters
            d_shape, d_camera = solve(hessian, sd_error, self.n,
                                      camera_update, focal_length_update)

            # Update parameters
            shape_parameters += d_shape
            if camera_update:
                camera_parameters = camera_parameters_update(
                    camera_parameters, d_camera)
                camera = camera.from_vector(camera_parameters)

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

        return MMAlgorithmResult(
            shape_parameters=shape_parameters_per_iter,
            texture_parameters=texture_parameters_per_iter,
            meshes=instance_per_iter, camera_transforms=camera_per_iter,
            image=image, initial_mesh=initial_mesh.rescale_texture(0., 1.),
            initial_camera_transform=camera_per_iter[0], gt_mesh=gt_mesh,
            costs=costs)
