import numpy as np
import scipy.sparse as sp


# ------------------------- JACOBIAN INITIALIZATION -------------------------- #
# Functions needed to construct the per-iteration ITW-V video Jacobian object.
def insert_id_constraint(J, c_id, n_p, n_sites_per_frame, n_frames):
    offset = n_sites_per_frame * n_frames
    J[offset:offset + n_p, :n_p] = np.eye(n_p) * np.sqrt(c_id)


def insert_exp_constraint(J, c_exp, n_p, n_q, n_sites_per_frame, n_frames):
    i_offset = (n_sites_per_frame * n_frames) + n_p
    j_offset = n_p

    size = n_q * n_frames
    exp_const = sp.dia_matrix((np.ones(size) * np.sqrt(c_exp), np.array([0])),
                              shape=(size, size))
    J[i_offset:i_offset + size, j_offset:j_offset + size] = exp_const


def insert_smoothness_constraint(J, c_sm, n_p, n_q, n_sites_per_frame,
                                 n_frames):
    A = np.eye(n_frames - 2)
    i, j = np.diag_indices(n_frames - 2)
    A[i[:-1], j[:-1] + 1] = -2
    A[i[:-2], j[:-2] + 2] = 1
    B = sp.csr_matrix(np.eye(n_q) * np.sqrt(c_sm))

    smoothing = sp.kron(A, B)
    x, y = smoothing.shape

    i_offset = n_frames * n_sites_per_frame + n_p + n_frames * n_q
    j_offset = n_p
    J[i_offset:(i_offset + x), j_offset:(j_offset + y)] = smoothing


# -------------------------- ERROR INITIALIZATION ---------------------------- #
def insert_id_constraint_to_e(e, p, c_id, n_sites_per_frame, n_frames):
    offset = n_sites_per_frame * n_frames
    e[offset:offset + p.size] = -np.sqrt(c_id) * p


def insert_exp_constraint_to_e(e, qs, c_exp, n_p, n_sites_per_frame, n_frames):
    offset = n_sites_per_frame * n_frames + n_p
    e[offset:offset + qs.size] = -np.sqrt(c_exp) * qs


def insert_smoothness_constraint_to_e(e, qs, c_sm, n_p, n_q, n_sites_per_frame,
                                      n_frames):
    offset = n_frames * n_sites_per_frame + n_p + n_frames * n_q

    # form the central difference scheme for qs:
    qsr = qs.reshape(n_frames, n_q)
    qs_a = qsr[:-2]
    qs_b = qsr[1:-1]
    qs_c = qsr[2:]

    smoothness = (-np.sqrt(c_sm) * (qs_a - 2 * qs_b + qs_c)).ravel()
    e[offset:offset + smoothness.size] = smoothness


# -------------------------- TOTAL INITIALIZATION ---------------------------- #
def initialize_jacobian_and_error(c_id, c_exp, c_sm, n_p, n_q, n_c,
                                  p, qs, n_sites_per_frame, n_frames):
    n_jac_m = ((n_frames * n_sites_per_frame) +  # Image/Landmark
               n_p +                             # ID constraint
               (n_frames * n_q) +                # Expression constraint
               (n_frames - 2) * n_q)             # Smoothness constraint
    n_jac_n = n_p + n_frames * n_q + n_frames * n_c
    J = sp.lil_matrix((n_jac_m, n_jac_n))
    insert_id_constraint(J, c_id, n_p, n_sites_per_frame, n_frames)
    insert_exp_constraint(J, c_exp, n_p, n_q, n_sites_per_frame, n_frames)
    # insert_smoothness_constraint(J, c_sm, n_p, n_q, n_sites_per_frame, n_frames)

    # The error term is always the size of the Jacobians' n. rows
    e = np.empty(n_jac_m)
    insert_id_constraint_to_e(e, p, c_id, n_sites_per_frame, n_frames)
    insert_exp_constraint_to_e(e, qs, c_exp, n_p, n_sites_per_frame, n_frames)
    # insert_smoothness_constraint_to_e(e, qs, c_sm, n_p, n_q, n_sites_per_frame,
    #                                   n_frames)
    return J, e


# ----------------------------- IMAGE JACOBIANS ------------------------------ #
def insert_J_f_p(J, J_f_p, f):
    si, sj = J_f_p.shape
    i_offset = f * si
    J[i_offset:(i_offset + si), :sj] = J_f_p


def insert_J_f_q(J, J_f_q, f, n_p):
    si, sj = J_f_q.shape
    i_offset = f * si
    j_offset = n_p + (f * sj)
    J[i_offset:(i_offset + si), j_offset:(j_offset + sj)] = J_f_q


def insert_J_f_c(J, J_f_c, f, n_p, n_q, n_frames):
    si, sj = J_f_c.shape
    i_offset = f * si
    j_offset = n_p + (n_frames * n_q) + f * sj
    J[i_offset:(i_offset + si), j_offset:(j_offset + sj)] = J_f_c


# ---------------------------- LANDMARK JACOBIANS ---------------------------- #
def insert_J_l_p(J, J_l_p, f, n_elements_per_frame, n_frames):
    si, sj = J_l_p.shape
    i_offset = n_frames * n_elements_per_frame + f * si
    J[i_offset:(i_offset + si), :sj] = J_l_p


def insert_J_l_q(J, J_l_q, f, n_p, n_elements_per_frame, n_frames):
    si, sj = J_l_q.shape
    i_offset = n_frames * n_elements_per_frame + f * si
    j_offset = n_p + f * sj
    J[i_offset:(i_offset + si), j_offset:(j_offset + sj)] = J_l_q


def insert_J_l_c(J, J_l_c, f, n_p, n_q, n_elements_per_frame, n_frames):
    si, sj = J_l_c.shape
    i_offset = n_frames * n_elements_per_frame + f * si
    j_offset = n_p + n_frames * n_q + f * sj
    J[i_offset:(i_offset + si), j_offset:(j_offset + sj)] = J_l_c


def insert_frame_to_J(J, j, f, c_l, n_p, n_q, n_elements_per_frame, n_frames):
    insert_J_f_p(J, j['J_f_p'], f)
    insert_J_f_q(J, j['J_f_q'], f, n_p)
    insert_J_f_c(J, j['J_f_c'], f, n_p, n_q, n_frames)

    c_l_srqt = np.sqrt(c_l)
    insert_J_l_p(J, c_l_srqt * j['J_l_p'], f, n_elements_per_frame, n_frames)
    insert_J_l_q(J, c_l_srqt * j['J_l_q'], f, n_p, n_elements_per_frame,
                 n_frames)
    insert_J_l_c(J, c_l_srqt * j['J_l_c'], f, n_p, n_q, n_elements_per_frame,
                 n_frames)


def insert_frame_to_e(e, j, f, c_l, n_elements_per_frame, n_lms, n_frames):
    offset_f = n_elements_per_frame * f
    e[offset_f:offset_f + n_elements_per_frame] = j['e_f']
    offset_l = n_elements_per_frame * n_frames + n_lms * 2 * f
    e[offset_l:offset_l + (n_lms * 2)] = np.sqrt(c_l) * j['e_l']
