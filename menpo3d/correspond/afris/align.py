from collections import namedtuple
from menpo.transform import AlignmentSimilarity


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
        aligned_mesh = AlignmentSimilarity(mesh.landmarks[group][label],
                                           self.sparse_template_3d).apply(mesh)
        fi_result = self.fr(aligned_mesh, group=group, label=label)
        return AlignFRResult(aligned_mesh.landmarks[group][label],
                             *fi_result)


class PartialLandmarkAFR(AlignFR):

    def __call__(self, mesh, mask, group=None, label=None):
        aligned_mesh = AlignmentSimilarity(
            mesh.landmarks[group][label],
            self.sparse_template_3d.from_mask(mask)).apply(mesh)
        fi_result = self.fr(aligned_mesh)
        return AlignFRResult(aligned_mesh.landmarks[group][label],
                             *fi_result)
