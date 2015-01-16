from collections import namedtuple
from menpo.transform import AlignmentSimilarity
from menpo.shape import PointCloud

AlignFRResult = namedtuple('AlignFRResult',
                           ['sparse_3d', 'texture_image', 'shape_image'])


class AlignFR(object):

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


class LandmarkAFR(AlignFR):

    def __call__(self, mesh, group=None, label=None):
        alignment = AlignmentSimilarity(
            mesh.landmarks[group][label],
            self.sparse_template_3d).as_non_alignment()
        aligned_mesh = alignment.apply(mesh)
        result = self.fr(aligned_mesh, group=group, label=label)
        result['sparse_3d'] = aligned_mesh.landmarks[group][label]
        result['alignment'] = alignment
        return result


class PartialLandmarkAFR(AlignFR):

    def __call__(self, mesh, mask, group=None, label=None):
        # copy the template
        masked_sparse_template_3d = PointCloud(self.sparse_template_3d.points)
        # filter the points based on the mask
        masked_sparse_template_3d.points = masked_sparse_template_3d.points[mask, :]

        alignment = AlignmentSimilarity(
            mesh.landmarks[group][label],
            masked_sparse_template_3d).as_non_alignment()
        aligned_mesh = alignment.apply(mesh)
        result = self.fr(aligned_mesh)
        result['sparse_3d'] = aligned_mesh.landmarks[group][label]
        result['alignment'] = alignment
        result['mask'] = mask
        return result
