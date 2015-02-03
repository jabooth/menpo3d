from . import io
from . import rasterize
from . import unwrap
from . import visualize


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


import numpy as np
import vtk
from vtk.util.numpy_support import (numpy_to_vtk, numpy_to_vtkIdTypeArray,
                                    vtk_to_numpy)
from menpo.shape import TriMesh


def to_vtk(self):
    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(self.points, deep=1))
    mesh.SetPoints(points)

    cells = vtk.vtkCellArray()
    cells.SetCells(self.n_tris,
                   numpy_to_vtkIdTypeArray(np.hstack((np.ones(self.n_tris)[:, None] * 3,
                                                      self.trilist)).astype(np.int64).ravel(), deep=1))
    mesh.SetPolys(cells)
    return mesh


def from_vtk(cls, vtk_mesh):
    points = vtk_to_numpy(vtk_mesh.GetPoints().GetData())
    trilist = vtk_to_numpy(vtk_mesh.GetPolys().GetData())
    return TriMesh(points, trilist=trilist.reshape([-1, 4])[:, 1:])


def triangle_areas(self):
    t = self.points[self.trilist]
    ij, ik = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return np.linalg.norm(np.cross(ij, ik), axis=1) / 2


def edges(self):
    t = self.points[self.trilist]
    return np.vstack((t[:, 1] - t[:, 0],
                      t[:, 2] - t[:, 0],
                      t[:, 1] - t[:, 2]))


def edge_indices(self):
    tl = self.trilist
    return np.vstack((tl[:, [0, 1]],
                      tl[:, [0, 2]],
                      tl[:, [1, 2]]))


def unique_edge_indicies(self):
    # Get a sorted list of edge pairs. sort ensures that each edge is ordered
    # from lowest index to highest.
    edge_pairs = np.sort(self.edge_indicies())

    # We want to remove duplicates - this is a little hairy, but basically we
    # get a view on the array where each pair is considered by numpy to be
    # one item
    edge_pair_view = np.ascontiguousarray(edge_pairs).view(
        np.dtype((np.void, edge_pairs.dtype.itemsize * edge_pairs.shape[1])))
    # Now we can use this view to ask for only unique edges...
    unique_edge_index = np.unique(edge_pair_view, return_index=True)[1]
    # And use that to filter our original list down
    return edge_pairs[unique_edge_index]


def unique_edges(self):
    x = self.points[self.unique_edge_indicies()]
    return x[:, 1] - x[:, 0]


def edge_lengths(self):
    return np.linalg.norm(self.edges(), axis=1)


def unique_edge_lengths(self):
    return np.linalg.norm(self.unique_edges(), axis=1)



TriMesh.to_vtk = to_vtk
TriMesh.from_vtk = classmethod(from_vtk)
TriMesh.triangle_areas = triangle_areas
TriMesh.edges = edges
TriMesh.edge_indicies = edge_indices
TriMesh.unique_edge_indicies = unique_edge_indicies
TriMesh.unique_edges = unique_edges
TriMesh.edge_lengths = edge_lengths
TriMesh.unique_edge_lengths = unique_edge_lengths
