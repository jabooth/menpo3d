from menpo.shape import PointCloud
from functools import partial
from menpo.transform import PiecewiseAffine, ThinPlateSplines


def interpolate_to_template(template_image, transform, image,
                            group=None, label=None):
    transform.set_target(image.landmarks[group][label])
    return image.as_unmasked(copy=False).warp_to_mask(template_image,
                                                      transform)


def pwa_interpolate_for_flatten_rasterize(template_image,
                                          group=None, label=None):
    template_image.constrain_to_landmarks(group=group, label=label)
    pwa = PiecewiseAffine(template_image.landmarks[group][label],
                          template_image.landmarks[group][label])
    interp = partial(interpolate_to_template, template_image, pwa,
                     group=group, label=label)
    return interp


def tps_interpolate_for_flatten_rasterize(template_image,
                                          group=None, label=None):
    src = template_image.landmarks[group][label]
    tgt = PointCloud(src.points - 2)  # just deform the points a little
    tps = ThinPlateSplines(src, tgt)
    interp = partial(interpolate_to_template, template_image, tps,
                     group=group, label=label)
    return interp


# def warp_to_template(template, image, transform=ThinPlateSplines,
#                      group=None, label=None):
#     t = transform(image.landmarks[group][label],
#                   template.landmarks[group][label])
#     return image.warp_to(template, t)

# LAFRIResult = namedtuple('LAFRIResult',
#                          ['sparse_3d', 'texture_image', 'shape_image',
#                           'texture_image_warped', 'shape_image_warped'])
#
#
# class LandmarkAFRInterpolate(LandmarkAFR):
#
#     def __init__(self, fr):
#         super(LandmarkAFRInterpolate, self).__init__(fr)
#         ti = self.template_image()
#         ti.constrain_to_landmarks()
#         self.template_image_masked = ti
#
#     def __call__(self, mesh, group=None, label=None):
#         lafr_result = super(LandmarkAFRInterpolate,
#                             self).__call__(mesh, group=group, label=label)
#         texture_image_warped = warp_to_template(self.template_image_masked,
#                                                 lafr_result.texture_image)
#         shape_image_warped = warp_to_template(self.template_image_masked,
#                                               lafr_result.shape_image)
#         return LAFRIResult(*lafr_result, rgb_image_warped=texture_image_warped,
#                            shape_image_warped=shape_image_warped)
