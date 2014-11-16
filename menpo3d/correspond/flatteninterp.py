from collections import namedtuple
from functools import partial
import numpy as np

from menpo.image import BooleanImage
from menpo.transform import Transform, AlignmentSimilarity, ThinPlateSplines
from menpo.shape import TriMesh

from menpo3d.rasterize import (GLRasterizer, model_to_clip_transform,
                               dims_3to2, dims_2to3)
from menpo3d.unwrap import optimal_cylindrical_unwrap


tri_in_set = lambda tl, s: np.in1d(tl.ravel(), s).reshape([-1, 3]).any(axis=1)


def prune_wrapped_tris(trimesh, width, height, dead_zone=0.2):
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
    trimesh = trimesh.copy()
    # remove any problematic tris
    trimesh.trilist = trimesh.trilist[~bad_tris]
    return trimesh

FlattenRasterizerResult = namedtuple('FlattenRasterizerResult',
                                     'rgb_image shape_image')


class FlattenRasterizer(object):

    def __init__(self, sparse_template_3d, transform=None,
                 image_width=1000, clip_space_scale=0.8):
        if transform is None:
            transform = optimal_cylindrical_unwrap(sparse_template_3d)
        self.transform = transform
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
        image = BooleanImage.blank((self.rasterizer.height,
                                    self.rasterizer.width))
        image.landmarks['sparse_template_2d'] = self.sparse_template_2d
        return image

    @property
    def template_image_width(self):
        return self.rasterizer.width

    @property
    def template_image_height(self):
        return self.rasterizer.height

    def __call__(self, mesh, **kwargs):
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
        # prune the mesh here to avoid artifacts
        f_mesh_pruned = prune_wrapped_tris(f_mesh, self.template_image_width,
                                           self.template_image_height)
        if f_mesh.n_tris != f_mesh_pruned.n_tris:
            print('removed {} problematic triangles'.format(
                f_mesh.n_tris - f_mesh_pruned.n_tris))
        return FlattenRasterizerResult(
            *self.rasterizer.rasterize_mesh_with_f3v_interpolant(f_mesh_pruned,
                                             per_vertex_f3v=mesh.points))


LandmarkAligningFRResult = namedtuple('AligningFRResult',
                                      ['sparse_3d', 'rgb_image',
                                       'shape_image'])


class AligningFR(object):

    def __init__(self, flatten_rasterizer):
        self.fr = flatten_rasterizer

    def template_image(self):
        return self.fr.template_image()

    @property
    def sparse_template_3d(self):
        return self.fr.sparse_template_3d

    @property
    def sparse_template_2d(self):
        return self.fr.sparse_template_2d


class LandmarkAligningFR(AligningFR):

    def __call__(self, mesh, group=None, label=None):
        aligned_mesh = AlignmentSimilarity(mesh.landmarks[group][label],
                                           self.sparse_template_3d).apply(mesh)
        fi_result = self.fr(aligned_mesh, group=group, label=label)
        return LandmarkAligningFRResult(aligned_mesh.landmarks[group][label],
                                        *fi_result)


def warp_to_template(template, image, transform=ThinPlateSplines,
                     group=None, label=None):
    t = transform(image.landmarks[group][label],
                  template.landmarks[group][label])
    return image.warp_to(template, t)

LAFRWIResult = namedtuple('LAFRWIResult',
                          ['sparse_3d', 'rgb_image', 'shape_image',
                          'rgb_image_warped', 'shape_image_warped'])


class LandmarkAligningFRWithInterpolation(LandmarkAligningFR):

    def __init__(self, fr):
        super(LandmarkAligningFRWithInterpolation, self).__init__(fr)
        ti = self.template_image()
        ti.constrain_to_landmarks()
        self.template_image_masked = ti

    def __call__(self, mesh, group=None, label=None):
        lafr_result = super(LandmarkAligningFRWithInterpolation,
                            self).__call__(mesh, group=group, label=label)
        rgb_image_warped = warp_to_template(self.template_image_masked,
                                            lafr_result.rgb_image)
        shape_image_warped = warp_to_template(self.template_image_masked,
                                              lafr_result.shape_image)
        return LAFRWIResult(*lafr_result, rgb_image_warped=rgb_image_warped,
                            shape_image_warped=shape_image_warped)


class FlattenInterpolater(Transform):
    r"""
    Class for bringing landmarked :map:`TriMesh` instances into dense
    correspondence by flattening the mesh into a 2D space.

    Parameters
    ----------

    target : :map:`PointCloud`
        The sparse target that input :map:`TriMesh` landmarks will be
        aligned against

    flatten : :map:`Transform`
        :map:`Transform` that will serve to flatten the mesh into a 2D
        space. If None probided, an optoimal cylindrical unwrap for target
        will be used.

    interp : :map:`Alignment`
        The 2D alignment that should be used to interpolate the flattened
        meshes.

    """
    def __init__(self, tgt, transform=None, interp=ThinPlateSplines):
        # Transform to flatten the mesh ready for interpolation
        if transform is None:
            transform = optimal_cylindrical_unwrap(tgt)
        self.flattener = transform
        self.interpolator = interp
        # Save out the rigid target
        self.tgt = tgt
        # Prepare the 2D/3d flattened targets
        self.f_tgt_3d = self.flattener.apply(tgt)
        self.dims_3to2, self.dims_2to3 = dims_3to2(), dims_2to3()
        self.f_tgt_2d = self.dims_3to2.apply(self.f_tgt_3d)

    def _apply(self, x, group=None, label='all'):
        r"""
        Return a version of the mesh that has been flattened and interpolated
        to be aligned with the target of this :map:`FlattenInterp`.

        Parameters
        ----------

        x : :map:`Transformable`
            The mesh that should be placed in correspondence
        group : `string`, optional
            The group label of the landmarks that should be used for
            interpolation. By default None, which will select the only
            available group if only one landmark group is present

            Default None
        label : `string`, optional
            The label of a particular set of landmarks, e.g. 'left_eye'.

            Default 'all'

        Returns
        -------

        :map:`TriMesh`
            A version of the mesh flattened and interpolated in 2D to be in
            correspondence with the original target.

        """
        # 2. Flatten the mesh, and warp it to align with the flattened target
        f_3d = self.flattener.apply(x)
        f_2d = self.dims_3to2.apply(f_3d)

        # 3. Warp the 2D flatted target to be in dense correspondence
        w_2d = self.interpolator(f_2d.landmarks[group][label],
                                 self.f_tgt_2d).apply(f_2d)

        # 4. Append on the Z dim + set it to what it was in the flattened case
        w_3d = self.dims_2to3.apply(w_2d)
        w_3d.points[:, 2] = f_3d.points[:, 2]
        return w_3d


def sample_points_and_trilist(mask, sampling_rate=1):
    r"""
    Returns sampling indices in x and y along with a trilist that joins the
    together the points.

    Parameters
    ----------

    mask: :class:`BooleanImage`
        The mask that should be sampled from

    sampling_rate: integer, optional
        The spacing of the grid

        Default 1

    Returns
    -------

    sample_x, sample_y, trilist: All ndarray
        To be used on a pixel array of an shape Image as follows
        points = img.pixels[sample_x, sample_y, :]
        points and trilist are now useful to construct a TriMesh

    """
    x, y = np.meshgrid(np.arange(0, mask.height, sampling_rate),
                       np.arange(0, mask.width, sampling_rate))
    # use the mask's mask to filter which points should be
    # selectable
    sampler_in_mask = mask.mask[x, y]
    sample_x, sample_y = x[sampler_in_mask], y[sampler_in_mask]
    # build a cheeky TriMesh to get hassle free trilist
    tm = TriMesh(np.vstack([sample_x, sample_y]).T)
    return sample_x, sample_y, tm.trilist