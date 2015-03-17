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


TriMesh.to_vtk = to_vtk
TriMesh.from_vtk = classmethod(from_vtk)