from functools import partial
import numpy as np
from menpo.image import BooleanImage
from menpo3d.rasterize import (GLRasterizer, model_to_clip_transform,
                               dims_3to2)
from menpo3d.unwrap import optimal_cylindrical_unwrap


class FlattenRasterize(object):
    r"""Callable that will rasterize
    """

    def __init__(self, sparse_template_3d, flattener=None,
                 image_width=1000, clip_space_scale=0.8):
        if flattener is None:
            flattener = optimal_cylindrical_unwrap(sparse_template_3d)
        self.transform = flattener
        self.sparse_template_3d = sparse_template_3d
        test_points = sparse_template_3d.copy()
        test_points.landmarks['test_flatten'] = test_points
        # find where the template landmarks end up in the flattened space (in
        #  3D)
        f_template_3d = self.transform.apply(test_points, group='test_flatten',
                                             label=None)
        f_template_3d._landmarks = None

        # Need to decide the appropriate size of the image - check the
        # ratio of the flatted 2D template and use it to infer height
        r_h, r_w = dims_3to2().apply(f_template_3d).range()
        ratio_w_to_h = (1.0 * r_w) / r_h
        image_height = int(ratio_w_to_h * image_width)

        # Build the rasterizer providing the clip space transform and shape
        cs_transform = model_to_clip_transform(f_template_3d,
                                               xy_scale=clip_space_scale)
        # now we have everything we need to construct an appropriate rasterizer
        self.rasterizer = GLRasterizer(projection_matrix=cs_transform.h_matrix,
                                       width=image_width, height=image_height)

        # Save out where the target landmarks land in the image
        self.sparse_template_2d = (
            self.rasterizer.model_to_image_transform.apply(f_template_3d))

    def template_image(self):
        image = BooleanImage.init_blank((self.rasterizer.height,
                                         self.rasterizer.width))
        image.landmarks['sparse_template_2d'] = self.sparse_template_2d
        return image

    @property
    def template_image_width(self):
        return self.rasterizer.width

    @property
    def template_image_height(self):
        return self.rasterizer.height

    def __call__(self, mesh, verbose=False, **kwargs):
        r"""
        Use a flattened warped mesh to build back a TriMesh in dense
        correspondence.

        Parameters
        ----------

        mesh: :class:`TriMesh`
            The original (rigidly aligned) TriMesh that we want to place in
            dense correspondence

        flattened_warped_mesh: :class:`TriMesh`


        """
        f_mesh = self.transform.apply(mesh, **kwargs)

        # we now have the flattened mesh, in an arbitrary coordinate space.
        # consult the rasterizer to find out where the triangles will be on
        # the image plane.
        f_mesh_img = self.rasterizer.model_to_image_transform.apply(f_mesh)
        # prune the mesh here to avoid artifacts
        good_tris = prune_wrapped_tris(f_mesh_img, self.template_image_width,
                                       self.template_image_height)
        # note that we only mask the triangulation - all the physical points
        # are still present (and will be sent to the GPU)
        f_mesh.trilist = f_mesh.trilist[good_tris]
        if verbose:
                print('removed {} problematic tris'.format(np.sum(~good_tris)))
        texture, shape = self.rasterizer.rasterize_mesh_with_f3v_interpolant(
            f_mesh, per_vertex_f3v=mesh.points)
        return {'texture_image': texture,
                'shape_image': shape}


def tri_in_set(tl, s):
    return np.in1d(tl.ravel(), s).reshape([-1, 3]).any(axis=1)


def prune_wrapped_tris(trimesh, width, height, dead_zone=0.2):
    r"""Returns a triangle mask that can be used to filter out triangles
    that wrap around the extremities of an imag plane.
    """
    p_h = trimesh.points[..., 0]
    p_w = trimesh.points[..., 1]
    tl = trimesh.trilist
    in_set_tl = partial(tri_in_set, tl)

    lt_w = np.nonzero(p_w < width * dead_zone)[0]
    gt_w = np.nonzero(p_w > width * (1 - dead_zone))[0]
    lt_h = np.nonzero(p_h < height * dead_zone)[0]
    gt_h = np.nonzero(p_h > height * (1 - dead_zone))[0]

    # problem_w = triangles with vertices in both extreme left and extreme
    # right of the image
    problem_w = np.logical_and(in_set_tl(lt_w), in_set_tl(gt_w))
    # problem_h = triangles with vertices in both extreme top and extreme
    # bottom of the image
    problem_h = np.logical_and(in_set_tl(lt_h), in_set_tl(gt_h))
    # we don't want triangles in problem_h or problem_w
    bad_tris = np.logical_or(problem_w, problem_h)

    # good triangle are all that's left
    return ~bad_tris
