import numpy as np
from scipy.linalg import sqrtm
from menpo.transform import Homogeneous


def sfm_param(r1, r2):
    return np.array([r1[0] * r2[0],
                     r1[0] * r2[1] + r1[1] * r2[0],
                     r1[0] * r2[2] + r1[2] * r2[0],
                     r1[1] * r2[1],
                     r1[1] * r2[2] + r1[2] * r2[1],
                     r1[2] * r2[2]])


def orthogonalize_sfm(M_hat, S_hat):
    # reshape the rotations to n_frames, 2, 3 to make selecting each frames'
    # rotation easy
    M_hat_r = M_hat.reshape([-1, 2, 3])
    n_frames = M_hat_r.shape[0]

    A = np.vstack([
        sfm_param(M_hat_r[:, 0].T, M_hat_r[:, 0].T).T,
        sfm_param(M_hat_r[:, 1].T, M_hat_r[:, 1].T).T,
        sfm_param(M_hat_r[:, 0].T, M_hat_r[:, 1].T).T
    ])
    b = np.hstack([np.ones(2 * n_frames), np.zeros(n_frames)])

    # Solve for v, the 6 unknown values of the symmetric value
    v = np.linalg.lstsq(A, b)[0]

    # C is a symmetric 3x3 matrix such that C = G * G'
    C = np.array([
        [v[0], v[1], v[2]],
        [v[1], v[3], v[4]],
        [v[2], v[4], v[5]]
    ])
    # Find G by decomposing C
    U, s, _ = np.linalg.svd(C)
    G = U * np.sqrt(s)

    num = M_hat_r.dot(G).reshape([-1, 3])
    den = np.vstack([np.linalg.pinv(x.T) for x in num.reshape([-1, 2, 3])])

    Q = G.dot(sqrtm(np.linalg.lstsq(num, den)[0]))

    M = M_hat.dot(Q)
    S = np.linalg.inv(Q).dot(S_hat.T).T
    return M, S


def transforms_for_sfm_result(M, T):
    return [Homogeneous(np.vstack(
        [np.hstack([m, t[:, None]]),
         [0, 0, 0, 1]])) for m, t in zip(M, T)]


def decompose_sfm_m_to_rot_scale(m):
    R1, s, R2 = np.linalg.svd(m, full_matrices=False)

    scale = s.mean()
    Rab = R1.dot(R2)
    Rc = np.cross(Rab[0], Rab[1])
    R = np.vstack([Rab, Rc[None]])
    return R, scale


def enforce_uniform_scale_constraint(M, L_hat):
    R_and_scale = [decompose_sfm_m_to_rot_scale(m) for m in M]
    P = np.array([(R * s)[:2] for R, s in R_and_scale]).reshape([-1, 3])
    S = np.linalg.lstsq(P, L_hat)[0].T
    return P, S


def structure_from_motion(pointclouds_2d, uniform_scale_constraint=True):
    # n_frames, 2, n_lms
    # For SfM we always consider landmarks [x, y], which goes against Menpo's
    # conventions...
    W = np.concatenate([l.points[:, ::-1].T[None] for l in pointclouds_2d],
                       axis=0)
    # W = np.concatenate([l.points.T[None] for l in pointclouds_2d], axis=0)
    T = W.mean(axis=-1)
    W_centred = W - T[..., None]

    # n_frames, 2 x n_lms
    L_hat = W_centred.reshape([len(pointclouds_2d) * 2, -1])

    # sum of largest independent x/y displacement across sequence
    scale = np.abs(L_hat).max(axis=-1).reshape([-1, 2]).max(axis=0).sum()

    # rescale down all our shapes to keep things stable
    L_hat_scaled = L_hat / scale

    U, s_squared, _ = np.linalg.svd(L_hat_scaled.dot(L_hat_scaled.T))
    s = np.sqrt(s_squared)
    V = np.linalg.solve(U * s, L_hat_scaled).T

    s_sqrt = np.sqrt(s[:3])
    M_hat = U[:, :3] * s_sqrt  # Rotation matrix and configuration weights
    S_hat = s_sqrt * V[:, :3]  # recovered 3D shape

    M_i, S = orthogonalize_sfm(M_hat, S_hat)

    if uniform_scale_constraint:
        print('enforcing uniform scale')
        M_i, S = enforce_uniform_scale_constraint(M_i.reshape([-1, 2, 3]),
                                                  L_hat)

    # reform M into meaningful arrays
    M = M_i.reshape([-1, 2, 3])

    pointcloud_3d = pointclouds_2d[0].copy()
    pointcloud_3d._landmarks = None
    pointcloud_3d.points = np.ascontiguousarray(S)
    # TODO
    # We should have a formalized Orthographic Camera - it should flip ouput
    # so that [X, Y, (Z)] maps to [Y', X`] in the image.
    return pointcloud_3d, M, T, scale
    # return pointcloud_3d, transforms_for_sfm_result(M, T)
