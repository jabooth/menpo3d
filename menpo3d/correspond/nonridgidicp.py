import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


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

    # build the kD-tree
    print('building KD-tree for target...')
    kdtree = KDTree(target.points)

    # init transformation
    X_prev = np.zeros((n_dims, n_dims + 1))
    X_prev = np.tile(X_prev, n).T
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
            # find nearest neighbour
            match = kdtree.query(v_i)[1]

            # formulate target and template data, and distance term
            U = target.points[match, :]

            data = np.hstack((v_i.ravel(), o))
            D_s = sp.coo_matrix((data, (row, col)))

            # correspondence detection for setting weight
            # add distance term
            A_s = sp.vstack((alpha_M_kron_G_s, D_s)).tocsr()
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
    point_corr = kdtree.query(v_i)[1]
    return v_i, point_corr
