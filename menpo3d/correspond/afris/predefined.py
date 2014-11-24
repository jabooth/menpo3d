from menpo3d.unwrap import optimal_cylindrical_unwrap
from .flattenrasterize import FlattenRasterize
from .align import LandmarkAFR


def landmark_align_cylindrical_unwrap_lafr(template_3d):
    cy_unwrap = optimal_cylindrical_unwrap(template_3d)
    fr = FlattenRasterize(template_3d, flattener=cy_unwrap)
    return LandmarkAFR(fr)
