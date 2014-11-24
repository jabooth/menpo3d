import numpy as np
from menpo.shape import TriMesh


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
