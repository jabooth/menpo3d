from collections import Counter
import numpy as np
import scipy.sparse as sp
import vtk
from menpo.shape import TriMesh
from menpo.transform import Translation, UniformScale

try:
    from scikits.sparse.cholmod import cholesky_AAt

    # user has cholesky available - provide a fast solve
    def spsolve(sparse_X, dense_b):
        factor = cholesky_AAt(sparse_X.T)
        return factor(sparse_X.T.dot(dense_b)).toarray()

except ImportError:
    # fallback to (much slower) scipy solve
    from scipy.sparse.linalg import spsolve as scipy_spsolve

    def spsolve(sparse_X, dense_b):
        return spsolve(sparse_X.T.dot(sparse_X), sparse_X.T.dot(dense_b)).toarray()


def build_intersector(mesh):
    r"""
    Build a function that can be used for calculating intersections of a ray
    with a mesh.

    Parameters
    ----------
    mesh : VTK mesh

    """
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


def edge_triangles(trilist):
    # Get a sorted list of edge pairs
    edge_pairs = np.sort(np.vstack((trilist[:, [0, 1]],
                                    trilist[:, [0, 2]],
                                    trilist[:, [1, 2]])))

    # convert to a tuple per edge pair
    edges = [tuple(x) for x in edge_pairs]
    # count the occurrences of the ordered edge pairs - edge pairs that
    # occur once are at the edge of the whole mesh
    mesh_edges = (e for e, i in Counter(edges).items() if i == 1)
    # index back into the edges to find which triangles contain these edges
    return np.array(list(set(edges.index(e) % trilist.shape[0]
                             for e in mesh_edges)))


def unique_edges(trilist):
    # Get a sorted list of edge pairs. sort ensures that each edge is ordered
    # from lowest index to highest.
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
    return edge_pairs[unique_edge_index]


def node_arc_incidence_matrix(trilist):
    unique_edge_pairs = unique_edges(trilist)
    m = unique_edge_pairs.shape[0]

    # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
    row = np.hstack((np.arange(m), np.arange(m)))
    col = unique_edge_pairs.T.ravel()
    data = np.hstack((-1 * np.ones(m), np.ones(m)))
    return sp.coo_matrix((data, (row, col)))


def non_rigid_icp(source, target, eps=1e-3, stiffness_values=None,
                  verbose=False):
    r"""
    Deforms the source trimesh to align with to optimally the target.
    """
    # Scale factors completely change the behavior of the algorithm - always
    # rescale the source down to a sensible size (so it fits inside box of
    # diagonal 1) and is centred on the origin. We'll undo this after the fit
    # so the user can use whatever scale they prefer.
    tr = Translation(-1 * source.centre())
    sc = UniformScale(1.0 / np.sqrt(np.sum(source.range() ** 2)), 3)
    prepare = tr.compose_before(sc)

    source = prepare.apply(source)
    target = prepare.apply(target)

    # store how to undo the similarity transform
    restore = prepare.pseudoinverse()

    n_dims = source.n_dims
    # Homogeneous dimension (1 extra for translation effects)
    h_dims = n_dims + 1
    points, trilist = source.points, source.trilist
    n = points.shape[0]  # record number of points

    edge_tris = edge_triangles(trilist)

    M_s = node_arc_incidence_matrix(trilist)

    # weight matrix
    G = np.identity(n_dims + 1)

    M_kron_G_s = sp.kron(M_s, G)

    # build octree for finding closest points on target.
    target_vtk = target.to_vtk()
    closest_point_on_target = build_closest_point_locator(target_vtk)

    # save out the target normals. We need them for the weight matrix.
    target_tri_normals = target.face_normals()

    # init transformation
    X_prev = np.tile(np.zeros((n_dims, h_dims)), n).T
    v_i = points

    if stiffness_values is not None:
        stiffness = stiffness_values
        if verbose:
            print('using user defined stiffness values: {}'.format(stiffness))
    else:
        # these values have been empirically found to perform well for well
        # rigidly aligned facial meshes
        stiffness = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]
        if verbose:
            print('using default stiffness values: {}'.format(stiffness))

    # to store per iteration information
    info = []

    # we need to prepare some indices for efficient construction of the D
    # sparse matrix.
    row = np.hstack((np.repeat(np.arange(n)[:, None], n_dims, axis=1).ravel(),
                     np.arange(n)))

    x = np.arange(n * h_dims).reshape((n, h_dims))
    col = np.hstack((x[:, :n_dims].ravel(),
                     x[:, n_dims]))

    o = np.ones(n)

    for alpha in stiffness:
        # get the term for stiffness
        alpha_M_kron_G_s = alpha * M_kron_G_s
        j = 0
        # iterate until X converge
        while True:
            # find nearest neighbour and the normals
            points = []
            tri_indices = []
            for p in v_i:
                point, tri_index = closest_point_on_target(p)
                points.append(point)
                tri_indices.append(tri_index)
            U = np.array(points)

            # ---- WEIGHTS ----
            # 1.  Edges
            # Are any of the corresponding tris on the edge of the target?
            # Where they are we return a false weight (we *don't* want to
            # include these points in the solve)
            w_i_e = np.in1d(tri_indices, edge_tris, invert=True)

            # 2. Normals
            # Calculate the normals of the current v_i
            v_i_tm = TriMesh(v_i, trilist=trilist, copy=False)
            v_i_n = v_i_tm.vertex_normals()
            # Extract the corresponding normals from the target
            u_i_n = target_tri_normals[tri_indices]
            # If the dot of the normals is lt 0.9 don't contrib to deformation
            w_i_n = (u_i_n * v_i_n).sum(axis=1) > 0.9

            # 3. Self-intersection
            # This adds approximately 12% to the running cost and doesn't seem
            # to be very critical in helping mesh fitting performance so for
            # now it's removed. Revisit later.
            # # Build an intersector for the current deformed target
            # intersect = build_intersector(v_i_tm.to_vtk())
            # # budge the source points 1% closer to the target
            # source = v_i + ((U - v_i) * 0.5)
            # # if the vector from source to target intersects the deformed
            # # template we don't want to include it in the optimisation.
            # problematic = [i for i, (s, t) in enumerate(zip(source, U))
            #                if len(intersect(s, t)[0]) > 0]
            # print(len(problematic) * 1.0 / n)
            # w_i_i = np.ones(v_i_tm.n_points, dtype=np.bool)
            # w_i_i[problematic] = False


            # Form the overall w_i from the normals, edge case
            w_i = np.logical_and(w_i_n, w_i_e)
            # we could add self intersection at a later date too...
            # w_i = np.logical_and(np.logical_and(w_i_n, w_i_e), w_i_i)


            prop_w_i = (n - w_i.sum() * 1.0) / n
            prop_w_i_n = (n - w_i_n.sum() * 1.0) / n
            prop_w_i_e = (n - w_i_e.sum() * 1.0) / n
            if verbose:
                print('alpha: {} ({}) - total : {:.0%} norms: {:.0%} '
                      'edges: {:.0%}'.format(alpha, j + 1, prop_w_i,
                                             prop_w_i_n, prop_w_i_e))
            j = j + 1

            # Build the sparse diagonal weight matrix
            W_s = sp.diags(w_i.astype(np.float)[None, :], [0])


            data = np.hstack((v_i.ravel(), o))
            D_s = sp.coo_matrix((data, (row, col)))

            # nullify the masked U values
            U[~w_i] = 0

            A_s = sp.vstack((alpha_M_kron_G_s, W_s.dot(D_s))).tocsr()
            B_s = sp.vstack((np.zeros((alpha_M_kron_G_s.shape[0], n_dims)),
                             U)).tocsr()
            X = spsolve(A_s, B_s)

            # deform template
            v_i = D_s.dot(X)
            err = np.linalg.norm(X_prev - X, ord='fro')
            info.append({
                'alpha': alpha,
                'iteration': j + 1,
                'prop_omitted': prop_w_i,
                'prop_omitted_norms': prop_w_i_n,
                'prop_omitted_edges': prop_w_i_e,
                'delta': err
            })
            X_prev = X

            if err / np.sqrt(np.size(X_prev)) < eps:
                break

    # final result if we choose closest points
    point_corr = np.array([closest_point_on_target(p)[0]
                           for p in v_i])

    return {
        'deformed_source': restore.apply(v_i),
        'matched_target': restore.apply(point_corr),
        'matched_tri_indices': tri_indices,
        'info': info
    }
