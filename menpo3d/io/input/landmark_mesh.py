from collections import OrderedDict
import numpy as np

from menpo.io.input.landmark import LandmarkImporter, PTSImporter
from menpo.shape import PointCloud


class MeshPTSImporter(PTSImporter):
    r"""
    Implements the :meth:`_build_points` method for meshes. Here, `x` is the
    first axis.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the landmarks
    """
    def __init__(self, filepath):
        super(MeshPTSImporter, self).__init__(filepath)

    def _build_points(self, xs, ys):
        """
        For meshes, `axis 0 = xs` and `axis 1 = ys`. Therefore, return the
        appropriate points array ordering.

        Parameters
        ----------
        xs : (N,) ndarray
            Row vector of `x` coordinates
        ys : (N,) ndarray
            Row vector of `y` coordinates

        Returns
        -------
        points : (N, 2) ndarray
            Array with `xs` as the first axis: `[xs; ys]`
        """
        return np.hstack([xs, ys])


class LM3Importer(LandmarkImporter):
    r"""
    Importer for the LM3 file format from the bosphorus dataset. This is a 3D
    landmark type and so it is assumed it only applies to meshes.

    Landmark set label: LM3

    Landmark labels:

    +------------------------+
    | label                  |
    +========================+
    | outer_left_eyebrow     |
    | middle_left_eyebrow    |
    | inner_left_eyebrow     |
    | inner_right_eyebrow    |
    | middle_right_eyebrow   |
    | outer_right_eyebrow    |
    | outer_left_eye_corner  |
    | inner_left_eye_corner  |
    | inner_right_eye_corner |
    | outer_right_eye_corner |
    | nose_saddle_left       |
    | nose_saddle_right      |
    | left_nose_peak         |
    | nose_tip               |
    | right_nose_peak        |
    | left_mouth_corner      |
    | upper_lip_outer_middle |
    | right_mouth_corner     |
    | upper_lip_inner_middle |
    | lower_lip_inner_middle |
    | lower_lip_outer_middle |
    | chin_middle            |
    +------------------------+
    """
    def __init__(self, filepath):
        super(LM3Importer, self).__init__(filepath)

    def _parse_format(self, asset=None):
        with open(self.filepath, 'r') as f:
            landmarks = f.read()

        # Remove comments and blank lines
        landmark_text = [l for l in landmarks.splitlines()
                         if (l.rstrip() and not '#' in l)]

        # First line says how many landmarks there are: 22 Landmarks
        # So pop it off the front
        num_points = int(landmark_text.pop(0).split()[0])
        xs = []
        ys = []
        zs = []
        labels = []

        # The lines then alternate between the labels and the coordinates
        for i in xrange(num_points * 2):
            if i % 2 == 0:  # label
                # Lowercase, remove spaces and replace with underscores
                l = landmark_text[i]
                l = '_'.join(l.lower().split())
                labels.append(l)
            else:  # coordinate
                p = landmark_text[i].split()
                xs.append(float(p[0]))
                ys.append(float(p[1]))
                zs.append(float(p[2]))

        xs = np.array(xs, dtype=np.float).reshape((-1, 1))
        ys = np.array(ys, dtype=np.float).reshape((-1, 1))
        zs = np.array(zs, dtype=np.float).reshape((-1, 1))

        self.pointcloud = PointCloud(np.hstack([xs, ys, zs]))
        # Create the mask whereby there is one landmark per label
        # (identity matrix)
        masks = np.eye(num_points).astype(np.bool)
        masks = np.vsplit(masks, num_points)
        masks = [np.squeeze(m) for m in masks]
        self.labels_to_masks = OrderedDict(zip(labels, masks))


class LANImporter(LandmarkImporter):
    r"""
    Importer for the LAN file format for the GOSH dataset. This is a 3D
    landmark type and so it is assumed it only applies to meshes.

    Landmark set label: LAN

    Note that the exact meaning of each landmark in this set varies,
    so all we can do is import all landmarks found under the label 'LAN'

    """
    def __init__(self, filepath):
        super(LANImporter, self).__init__(filepath)

    def _parse_format(self, asset=None):
        with open(self.filepath, 'r') as f:
            landmarks = np.fromfile(
                f, dtype=np.float32)[3:].reshape([-1, 3]).astype(np.double)

        self.pointcloud = PointCloud(landmarks)
        self.labels_to_masks = OrderedDict(
            [('all', np.ones(landmarks.shape[0], dtype=np.bool))])


class BNDImporter(LandmarkImporter):
    r"""
    Importer for the BND file format for the BU-3DFE dataset. This is a 3D
    landmark type and so it is assumed it only applies to meshes.

    Landmark set label: BND

    Landmark labels:

    +---------------+
    | label         |
    +===============+
    | left_eye      |
    | right_eye     |
    | left_eyebrow  |
    | right_eyebrow |
    | nose          |
    | mouth         |
    | chin          |
    +---------------+
    """
    def __init__(self, filepath):
        super(BNDImporter, self).__init__(filepath)

    def _indices_to_mask(n_points, indices):
        """
        Helper function to turn an array of indices in to a boolean mask.

        Parameters
        ----------
        n_points : int
            The total number of points for the mask
        indices : ndarray of ints
            An array of integers representing the `True` indices.

        Returns
        -------
        boolean_mask : ndarray of bools
            The mask for the set of landmarks where each index from indices is set
            to `True` and the rest are `False`
        """
        mask = np.zeros(n_points, dtype=np.bool)
        mask[indices] = True
        return mask

    def _parse_format(self, asset=None):
        with open(self.filepath, 'r') as f:
            landmarks = f.read()

        # Remove blank lines
        landmark_text = [l for l in landmarks.splitlines() if l.rstrip()]
        landmark_text = [l.split() for l in landmark_text]

        n_points = len(landmark_text)
        landmarks = np.zeros([n_points, 3])
        for i, l in enumerate(landmark_text):
            # Skip the first number as it's an index into the mesh
            landmarks[i, :] = np.array([float(l[1]), float(l[2]), float(l[3])],
                                       dtype=np.float)

        self.pointcloud = PointCloud(landmarks)
        self.labels_to_masks = OrderedDict([
            ('left_eye', self._indices_to_mask(n_points, np.arange(8))),
            ('right_eye', self._indices_to_mask(n_points, np.arange(8, 16))),
            ('left_eyebrow', self._indices_to_mask(n_points,
                                                   np.arange(16, 26))),
            ('right_eyebrow', self._indices_to_mask(n_points,
                                                    np.arange(26, 36))),
            ('nose', self._indices_to_mask(n_points, np.arange(36, 48))),
            ('mouth', self._indices_to_mask(n_points, np.arange(48, 68))),
            ('chin', self._indices_to_mask(n_points, np.arange(68, 83)))
        ])
