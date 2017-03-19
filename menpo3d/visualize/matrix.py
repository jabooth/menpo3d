import numpy as np
from matplotlib import pyplot as plt


def starts_and_ends(groups):
    ends = np.cumsum(groups)
    starts = np.hstack([0, ends[:-1]])
    return np.vstack([starts, ends]).T


def intersperse(iterable, delimiter):
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x


def intersperse_row(row, width=1):
    spacer = np.ones((row[0].shape[0], width))
    return np.hstack(list(intersperse(row, spacer)))


def intersperse_col(col, width=1):
    spacer = np.ones((width, col[0].shape[1]))
    return np.vstack(list(intersperse(col, spacer)))


def matrix_with_gridlines(A, i_groups, j_groups, width=1):
    cols = []
    if A.shape[0] != np.sum(i_groups):
        raise ValueError('i groups mismatch - length of groups: {} '
                         'rows of A: {}'.format(np.sum(i_groups), A.shape[0]))
    if A.shape[1] != np.sum(j_groups):
        raise ValueError('j groups mismatch - length of groups: {} '
                         'cols of A: {}'.format(np.sum(j_groups), A.shape[1]))
    for s_i, e_i in starts_and_ends(i_groups):
        row = intersperse_row(
            [A[s_i:e_i, s_j:e_j] for s_j, e_j in
             starts_and_ends(j_groups)], width=width)
        cols.append(row)
    return intersperse_col(cols, width=width)


def view_matrix(A, colorbar=True):
    plt.matshow(A, interpolation='none', cmap='hot')
    if colorbar:
        plt.colorbar()
