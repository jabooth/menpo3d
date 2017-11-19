import numpy as np

from menpo.feature import gradient as fast_gradient
from menpo.image import Image
from menpo3d.rasterize import rasterize_barycentric_coordinates

from ..derivatives import (d_camera_d_camera_parameters,
                           d_camera_d_shape_parameters)


def camera_parameters_update(c, dc):
    # Add for focal length and translation parameters, but multiply for
    # quaternions
    new = c + dc
    new[1:5] = quaternion_multiply(c[1:5], dc[1:5])
    return new


def quaternion_multiply(current_q, increment_q):
    # Make sure that the q increment has unit norm
    increment_q /= np.linalg.norm(increment_q)
    # Update
    w0, x0, y0, z0 = current_q
    w1, x1, y1, z1 = increment_q
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                      x1*w0 + y1*z0 - z1*y0 + w1*x0,
                     -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                      x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)


def gradient_xy(image):
    # Compute the gradient of the image
    grad = fast_gradient(image)

    # Slice off the gradient for X and Y separately
    grad_y = Image(grad.pixels[:image.n_channels])
    grad_x = Image(grad.pixels[image.n_channels:])

    return grad_x, grad_y


def sample_at_bc_vi(x, bcoords, vertex_indices):
    per_vertex_per_pixel = x[vertex_indices]
    return np.sum(per_vertex_per_pixel * bcoords[..., None], axis=1)


def visible_sample_points(instance_in_img, image_shape, n_samples):
    # Inverse rendering
    yx, bcoords, tri_indices = rasterize_barycentric_coordinates(
        instance_in_img, image_shape)

    # Select triangles randomly
    rand = np.random.permutation(bcoords.shape[0])
    bcoords = bcoords[rand[:n_samples]]
    yx = yx[rand[:n_samples]]
    tri_indices = tri_indices[rand[:n_samples]]

    # Build the vertex indices (3 per pixel) for the visible triangles
    vertex_indices = instance_in_img.trilist[tri_indices]

    return vertex_indices, bcoords, tri_indices, yx


def J_lms(camera, warped_uv, shape_pc_uv, camera_update, focal_length_update):
    # Compute derivative of camera wrt shape and camera parameters
    J = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)
    n_camera_parameters = 0
    if camera_update:
        dp_dr = d_camera_d_camera_parameters(
            camera, warped_uv, with_focal_length=focal_length_update)
        J = np.hstack((J, dp_dr))
        n_camera_parameters = dp_dr.shape[1]

    # Reshape to : n_params x (2 * N)
    n_params = J.shape[1]
    J = np.transpose(J, (1, 0, 2)).reshape(n_params, -1)
    return J, n_camera_parameters


class LucasKanade(object):
    def __init__(self, model, n_samples, eps=1e-3):
        self.model = model
        self.eps = eps
        self.n_samples = n_samples
        # Call precompute
        self._precompute()

    @property
    def n(self):
        r"""
        Returns the number of active components of the shape model.

        :type: `int`
        """
        return self.model.shape_model.n_active_components

    @property
    def m(self):
        r"""
        Returns the number of active components of the texture model.

        :type: `int`
        """
        return self.model.texture_model.n_active_components

    @property
    def n_vertices(self):
        r"""
        Returns the number of vertices of the shape model's trimesh.

        :type: `int`
        """
        return self.model.shape_model.template_instance.n_points

    @property
    def n_channels(self):
        r"""
        Returns the number of channels of the texture model.

        :type: `int`
        """
        return self.model.n_channels

    def _precompute(self):
        print('Override this in subclasses')
