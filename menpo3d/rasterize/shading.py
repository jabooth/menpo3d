import numpy as np
from menpo.shape import TriMesh, ColouredTriMesh
from menpo.transform import UniformScale, Translation

__all__ = ["LambertianModel"]
__default_light_sources__ = ((.667, -.66, .33), (-.667, .66, .33))

def l2_normalize(x, axis=0, epsilon=1e-12):
    """
    Transforms an `ndarray` to have a unit l2 norm along
    a given direction.
    ----------
    x : `ndarray`
        The array to be transformed.
    axis : `int`
        The axis that will be l2 unit normed.
    epsilon: `float`
        A small value such as to avoid division by zero.

    Returns
    -------
    x : (D,) `ndarray`
        The transformed array.
    """
    return x / np.maximum(np.linalg.norm(x, axis=axis), epsilon)

def mesh_in_a_sphere(mesh):
    scale = UniformScale(1 / mesh.norm(), mesh.n_dims)

    translation = Translation(-scale.apply(mesh).centre())
    return translation.compose_after(scale)

class LambertianModel(object):
    def __init__(self, diffuse_colour=(.5, .5, .5), light_positions=__default_light_sources__):
        self.diffuse_colour = np.asarray(diffuse_colour)
        self.light_positions = l2_normalize(np.asarray(light_positions).reshape(-1, 3), axis=0)

    def apply(self, mesh, use_texture=False, copy=True):
        mesh = mesh.as_colouredtrimesh(copy=copy)
            
        unit_transform = mesh_in_a_sphere(mesh)
        mesh = unit_transform.apply(mesh)
        light_directions = l2_normalize(
            self.light_positions.reshape(-1, 1, 3) - mesh.points[None, ...], axis=0)
        
        # Calculate the lambertian reflectance for each light source.
        # This will be an `ndarray` of dimensions num_light_sources x num_vertices.
        lambertian = np.sum(light_directions * mesh.vertex_normals()[None, ...], 2)[..., None]

        # Sum up the contribution of all the light sources and multiply by the
        # diffusion colour.
        lambertian = lambertian.sum(0) * self.diffuse_colour

        texture = mesh.colours if use_texture else 0
        mesh.colours[..., :] = texture + lambertian
        mesh.colours[...] = np.clip(mesh.colours, 0, 1)

        return unit_transform.pseudoinverse().apply(mesh)