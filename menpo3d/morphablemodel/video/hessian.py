import numpy as np


# -------------------------- HESSIAN INITIALIZATION -------------------------- #
# Functions needed to construct the per-iteration ITW-V video Hessian object.
def insert_id_constraint(H, c_id, n_p):
    H[:n_p, :n_p] += np.eye(n_p) * c_id


def insert_exp_constraint(H, c_exp, n_p, n_q, n_frames):
    i_offset = n_p
    j_offset = n_p

    size = n_q * n_frames
    exp_const = np.eye(size) * c_exp
    H[i_offset:i_offset + size, j_offset:j_offset + size] = exp_const


# def insert_smoothness_constraint(H, c_sm, n_p, n_q, n_sites_per_frame,
#                                  n_frames):
#     A = np.eye(n_frames - 2)
#     i, j = np.diag_indices(n_frames - 2)
#     A[i[:-1], j[:-1] + 1] = -2
#     A[i[:-2], j[:-2] + 2] = 1
#     B = sp.csr_matrix(np.eye(n_q) * np.sqrt(c_sm))
#
#     smoothing = sp.kron(A, B)
#     x, y = smoothing.shape
#
#     i_offset = n_frames * n_sites_per_frame + n_p + n_frames * n_q
#     j_offset = n_p
#     H[i_offset:(i_offset + x), j_offset:(j_offset + y)] = smoothing


# ----------------------- J.T.dot(e) INITIALIZATION  ------------------------- #
def insert_id_constraint_to_JTe(JTe, p, c_id, n_p):
    JTe[:n_p] += -c_id * p


def insert_exp_constraint_to_JTe(JTe, qs, c_exp, n_p, n_q, n_frames):
    size = n_q * n_frames
    JTe[n_p:n_p + size] += -c_exp * qs


# def insert_smoothness_constraint_to_JTe(JTe, qs, c_sm, n_p, n_q,
#                                         n_sites_per_frame, n_frames):
#     offset = n_frames * n_sites_per_frame + n_p + n_frames * n_q
#
#     # form the central difference scheme for qs:
#     qsr = qs.reshape(n_frames, n_q)
#     qs_a = qsr[:-2]
#     qs_b = qsr[1:-1]
#     qs_c = qsr[2:]
#
#     smoothness = (-np.sqrt(c_sm) * (qs_a - 2 * qs_b + qs_c)).ravel()
#     JTe[offset:offset + smoothness.size] = smoothness


# -------------------------- TOTAL INITIALIZATION ---------------------------- #
def initialize_hessian_and_JTe(c_id, c_exp, c_sm, n_p, n_q, n_c,
                               p, qs, n_sites_per_frame, n_frames):
    n = n_p + n_frames * n_q + n_frames * n_c
    H = np.zeros((n, n))
    insert_id_constraint(H, c_id, n_p)
    insert_exp_constraint(H, c_exp, n_p, n_q, n_frames)
    # insert_smoothness_constraint(H, c_sm, n_p, n_q, n_sites_per_frame, n_frames)

    # The J.T.dot(e) term is always the size of the Hessian w/h (n total params)
    JTe = np.zeros(n)
    insert_id_constraint_to_JTe(JTe, p, c_id, n_p)
    insert_exp_constraint_to_JTe(JTe, qs, c_exp, n_p, n_q, n_frames)
    # insert_smoothness_constraint_to_JTe(JTe, qs, c_sm, n_p, n_q,
    #                                     n_sites_per_frame, n_frames)
    return H, JTe


# -------------------------- PER-FRAME UPDATES  ------------------------------ #

def insert_frame_to_H(H, j, f, n_p, n_q, n_c, c_l, n_frames):

    pp = j['J_f_p'].T @ j['J_f_p'] + c_l * j['J_l_p'].T @ j['J_l_p']
    qq = j['J_f_q'].T @ j['J_f_q'] + c_l * j['J_l_q'].T @ j['J_l_q']
    cc = j['J_f_c'].T @ j['J_f_c'] + c_l * j['J_l_c'].T @ j['J_l_c']

    pq = j['J_f_p'].T @ j['J_f_q'] + c_l * j['J_l_p'].T @ j['J_l_q']
    pc = j['J_f_p'].T @ j['J_f_c'] + c_l * j['J_l_p'].T @ j['J_l_c']

    qc = j['J_f_q'].T @ j['J_f_c'] + c_l * j['J_l_q'].T @ j['J_l_c']

    # Find the right offset into the Hessian for Q/C (which are per-frame
    # terms). Note that p terms always sum, so no per-frame offset needed.
    offset_q = n_p + f * n_q
    offset_c = n_p + n_frames * n_q + f * n_c

    # 1. Update the terms along the block diagonal of the Hessian
    H[:n_p, :n_p] += pp
    H[offset_q:offset_q + n_q, offset_q:offset_q + n_q] += qq
    H[offset_c:offset_c + n_c, offset_c:offset_c + n_c] += cc

    # 2. Update terms at the off diagonal. Each time we can immediately fill
    # in the symmetric part of the term (as we know the Hessian is symmetric).
    H[:n_p, offset_q:offset_q + n_q] += pq
    H[offset_q:offset_q + n_q, :n_p] += pq.T

    H[:n_p, offset_c:offset_c + n_c] += pc
    H[offset_c:offset_c + n_c, :n_p] += pc.T

    H[offset_q:offset_q + n_q, offset_c:offset_c + n_c] += qc
    H[offset_c:offset_c + n_c, offset_q:offset_q + n_q] += qc.T


def insert_frame_to_JTe(JTe, j, f, n_p, n_q, n_c, c_l, n_frames):

    JTe_p = j['J_f_p'].T @ j['e_f'] + c_l * j['J_l_p'].T @ j['e_l']
    JTe_q = j['J_f_q'].T @ j['e_f'] + c_l * j['J_l_q'].T @ j['e_l']
    JTe_c = j['J_f_c'].T @ j['e_f'] + c_l * j['J_l_c'].T @ j['e_l']

    JTe[:n_p] += JTe_p

    offset_q = n_p + f * n_q
    JTe[offset_q:offset_q + n_q] += JTe_q

    offset_c = n_p + n_frames * n_q + f * n_c
    JTe[offset_c:offset_c + n_c] += JTe_c
