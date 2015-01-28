import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
import vtk
from menpo.shape import TriMesh


def build_intersector(mesh):
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(mesh)
    obbTree.BuildLocator()

    def intersect(source, target):
        intersections = vtk.vtkPoints()
        tri_ids = vtk.vtkIdList()
        code = obbTree.IntersectWithLine(source, target,
                                         intersections, tri_ids)
        if code in (1, -1):
            i_data = intersections.GetData()
            points = np.array([i_data.GetTuple3(i) for i in range(i_data.GetNumberOfTuples())])
            ids = np.array([tri_ids.GetId(i) for i in range(tri_ids.GetNumberOfIds())])
            return points, ids
        else:
            return [], []

    return intersect


def build_closest_point_locator(mesh):
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()

    c_point = [0., 0., 0.]
    cell_id = vtk.mutable(0)
    sub_id = vtk.mutable(0)
    distance = vtk.mutable(0.0)

    def closest_point(point):
        cell_locator.FindClosestPoint(point, c_point,
                                      cell_id, sub_id, distance)
        return c_point[:], cell_id.get()

    return closest_point


def non_rigid_icp(source, target, eps=1e-3):
    r"""
    Deforms the source trimesh to align with to optimally the target.
    """
    n_dims = source.n_dims
    # Homogeneous dimension (1 extra for translation effects)
    h_dims = n_dims + 1
    points = source.points
    trilist = source.trilist

    # Configuration
    upper_stiffness = 101
    lower_stiffness = 1
    stiffness_step = 5

    # Get a sorted list of edge pairs (note there will be many mirrored pairs
    # e.g. [4, 7] and [7, 4])
    edge_pairs = np.sort(np.vstack((trilist[:, [0, 1]],
                                    trilist[:, [0, 2]],
                                    trilist[:, [1, 2]])))

    # We want to remove duplicates - this is a little hairy, but basically we
    # get a view on the array where each pair is considered by numpy to be
    # one item
    edge_pair_view = np.ascontiguousarray(edge_pairs).view(
        np.dtype((np.void, edge_pairs.dtype.itemsize * edge_pairs.shape[1])))
    # Now we can use this view to ask for only unique edges...
    unique_edge_index = np.unique(edge_pair_view, return_index=True)[1]
    # And use that to filter our original list down
    unique_edge_pairs = edge_pairs[unique_edge_index]

    # record the number of unique edges and the number of points
    n = points.shape[0]
    m = unique_edge_pairs.shape[0]

    # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
    row = np.hstack((np.arange(m), np.arange(m)))
    col = unique_edge_pairs.T.ravel()
    data = np.hstack((-1 * np.ones(m), np.ones(m)))
    M_s = sp.coo_matrix((data, (row, col)))

    # weight matrix
    G = np.identity(n_dims + 1)

    M_kron_G_s = sp.kron(M_s, G)

    # build octree for finding closest points on target.
    target_vtk = target.to_vtk()
    print('building nearest point locator for target...')
    closest_point_on_target = build_closest_point_locator(target_vtk)

    # save out the target normals. We need them for the weight matrix.
    target_tri_normals = target.face_normals()

    # init transformation
    X_prev = np.tile(np.zeros((n_dims, h_dims)), n).T
    v_i = points

    # start nicp
    # for each stiffness
    stiffness = range(upper_stiffness, lower_stiffness, -stiffness_step)
    errs = []


    # we need to prepare some indices for efficient construction of the D
    # sparse matrix.
    row = np.hstack((np.repeat(np.arange(n)[:, None], n_dims, axis=1).ravel(),
                     np.arange(n)))

    x = np.arange(n * h_dims).reshape((n, h_dims))
    col = np.hstack((x[:, :n_dims].ravel(),
                     x[:, n_dims]))

    o = np.ones(n)

    for alpha in stiffness:
        print(alpha)
        # get the term for stiffness
        alpha_M_kron_G_s = alpha * M_kron_G_s

        # iterate until X converge
        while True:

            # find nearest neighbour and the normals
            points = []
            tri_indicies = []
            for p in v_i:
                point, tri_index = closest_point_on_target(p)
                points.append(point)
                tri_indicies.append(tri_index)
            U = np.array(points)


            u_i_n = target_tri_normals[tri_indicies]

            # calculate the normals of the current v_i
            v_i_n = TriMesh(v_i, trilist=trilist, copy=False).vertex_normals()

            # If the dot of the normals is lt 0.9 don't contrib to deformation
            w_i_n = (u_i_n * v_i_n).sum(axis=1) > 0.9

            # Form the overall w_i from the normals, edge case and self
            # intersection
            w_i = w_i_n

            # Build the sparse diagonal weight matrix
            W_s = sp.diags(w_i.astype(np.float)[None, :], [0])


            data = np.hstack((v_i.ravel(), o))
            D_s = sp.coo_matrix((data, (row, col)))

            # nullify the masked U values
            U[~w_i] = 0

            A_s = sp.vstack((alpha_M_kron_G_s, W_s.dot(D_s))).tocsr()
            B_s = sp.vstack((np.zeros((alpha_M_kron_G_s.shape[0], n_dims)),
                             U)).tocsr()
            X_s = spsolve(A_s.T.dot(A_s), A_s.T.dot(B_s))
            X = X_s.toarray()

            # deform template
            v_i = D_s.dot(X)
            err = np.linalg.norm(X_prev - X, ord='fro')
            errs.append([alpha, err])
            X_prev = X

            if err / np.sqrt(np.size(X_prev)) < eps:
                break

    # final result
    point_corr = np.array([closest_point_on_target(p)[0]
                           for p in v_i])
    # only update the points for the non-problematic ones
    v_i[w_i] = point_corr[w_i]
    return v_i
