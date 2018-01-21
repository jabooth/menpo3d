from collections import defaultdict
from datetime import timedelta
from menpo.transform import Scale
from time import time

import numpy as np
import scipy.sparse as sp
from menpo.visualize import print_progress, bytes_str

from ..algorithm.derivatives import (d_camera_d_shape_parameters,
                                     d_camera_d_camera_parameters)
from ..algorithm.lk import camera_parameters_update
from ..algorithm.lk.base import gradient_xy
from ..algorithm.lk.projectout import project_out, sample_uv_terms
from .hessian import (initialize_hessian_and_JTe, insert_frame_to_H,
                      insert_frame_to_JTe)


def J_data(camera, warped_uv, shape_pc_uv, U_tex_pc, grad_x_uv,
           grad_y_uv, focal_length_update=False):
    # Compute derivative of camera wrt shape and camera parameters
    dp_da_dr = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)

    dp_dr = d_camera_d_camera_parameters(
        camera, warped_uv, with_focal_length=focal_length_update)

    # stack the shape_parameters/camera_parameters updates
    dp_da_dr = np.hstack((dp_da_dr, dp_dr))

    # Multiply image gradient with camera derivative
    permuted_grad_x = np.transpose(grad_x_uv[..., None], (0, 2, 1))
    permuted_grad_y = np.transpose(grad_y_uv[..., None], (0, 2, 1))
    J = permuted_grad_x * dp_da_dr[0] + permuted_grad_y * dp_da_dr[1]

    # Project-out
    n_params = J.shape[1]
    J = np.transpose(J, (1, 0, 2)).reshape(n_params, -1)
    return project_out(J, U_tex_pc)


def J_lms(camera, warped_uv, shape_pc_uv, focal_length_update=False):
    # Compute derivative of camera wrt shape and camera parameters
    J = d_camera_d_shape_parameters(camera, warped_uv, shape_pc_uv)
    dp_dr = d_camera_d_camera_parameters(
        camera, warped_uv, with_focal_length=focal_length_update)
    J = np.hstack((J, dp_dr))

    # Reshape to : n_params x (2 * N)
    n_params = J.shape[1]
    J = np.transpose(J, (1, 0, 2)).reshape(n_params, -1)
    return J


def jacobians(s, c, image, lms_points_xy, mm, id_ind, exp_ind, template_camera,
              grad_x, grad_y, shape_pc, shape_pc_lms, n_samples,
              compute_costs=False):

    instance = mm.shape_model.instance(s, normalized_weights=True)
    camera = template_camera.from_vector(c)

    (instance_w, instance_in_image, warped_uv, img_error_uv,
     shape_pc_uv, U_tex_pc, grad_x_uv, grad_y_uv
     ) = sample_uv_terms(instance, image, camera, mm, shape_pc, grad_x, grad_y,
                         n_samples)

    # Data term Jacobian
    JT = J_data(camera, warped_uv, shape_pc_uv, U_tex_pc, grad_x_uv, grad_y_uv)

    # Landmarks term Jacobian
    # Get projected instance on landmarks and error term
    warped_lms = instance_in_image.points[mm.model_landmarks_index]
    # TODO check the ordering here.
    # Reverted to match image case, check.
    lms_error_xy = (lms_points_xy - warped_lms[:, [1, 0]]).T.ravel()
    warped_view_lms = instance_w[mm.model_landmarks_index]
    J_lT = J_lms(camera, warped_view_lms, shape_pc_lms)

    # form the main two Jacobians...
    J_f = JT.T
    J_l = J_lT.T

    # ...and then slice at the appropriate indices to break down by param type.
    c_offset = id_ind.shape[0] + exp_ind.shape[0]
    jacs = {
        'J_f_p': J_f[:, id_ind],
        'J_f_q': J_f[:, exp_ind],
        'J_f_c': J_f[:, c_offset:],

        'J_l_p': J_l[:, id_ind],
        'J_l_q': J_l[:, exp_ind],
        'J_l_c': J_l[:, c_offset:],

        'e_f': img_error_uv,
        'e_l': lms_error_xy
    }

    if compute_costs:
        resid_f = project_out(img_error_uv, U_tex_pc)
        err_f = (resid_f ** 2).sum()

        resid_l = lms_error_xy
        err_l = (resid_l ** 2).sum()

        jacs['costs'] = {
            'err_f': err_f,
            'err_l': err_l
        }

    return jacs


def increment_parameters(images, mm, id_indices, exp_indices, template_camera,
                         p, qs, cs,
                         c_f=1, c_l=1, c_id=1, c_exp=1, c_sm=1,
                         lm_group=None, n_samples=1000,
                         compute_costs=True):

    n_frames = len(images)
    n_points = mm.shape_model.template_instance.n_points
    n_p = len(id_indices)
    n_q = len(exp_indices)
    n_c = cs.shape[1] - 2  # sub one for quaternion, one for focal length

    print('Precomputing....')
    # Rescale shape components to have size:
    # n_points x (n_components * n_dims)
    # and to be scaled by the relevant standard deviation.
    shape_pc = (
            mm.shape_model.components.T *
            np.sqrt(mm.shape_model.eigenvalues)
    ).reshape([n_points, -1])
    # include std.dev in principal components
    shape_pc_lms = shape_pc.reshape([n_points, 3, -1])[mm.model_landmarks_index]

    print('Initializing Hessian/JTe for frame...')
    H, JTe = initialize_hessian_and_JTe(c_id, c_exp, c_sm, n_p, n_q, n_c, p, qs,
                                        n_frames)
    print('H: {} ({})'.format(H.shape, bytes_str(H.nbytes)))

    if compute_costs:
        costs = defaultdict(list)

    for (f, image), c, q in zip(enumerate(print_progress(
            images, prefix='Incrementing H/JTe')), cs, qs):

        # Form the overall shape parameter: [p, q]
        s = np.zeros(mm.shape_model.n_active_components)
        s[id_indices] = p
        s[exp_indices] = q

        # In our error we consider landmarks stored [x, y] - so flip here.
        lms_points_xy = image.landmarks[lm_group].points[:, [1, 0]]

        # Compute input image gradient
        grad_x, grad_y = gradient_xy(image)

        j = jacobians(s, c, image, lms_points_xy, mm, id_indices, exp_indices,
                      template_camera, grad_x, grad_y, shape_pc, shape_pc_lms,
                      n_samples, compute_costs=compute_costs)
        insert_frame_to_H(H, j, f, n_p, n_q, n_c, c_f, c_l, n_frames)
        insert_frame_to_JTe(JTe, j, f, n_p, n_q, n_c, c_f, c_l, n_frames)

        if compute_costs:
            for cost, val in j['costs'].items():
                costs[cost].append(val)

    print('Converting Hessian to sparse format')
    H = sp.csr_matrix(H)
    print("Sparsity (prop. 0's) of H: {:.2%}".format(
        1 - (H.count_nonzero() / np.prod(np.array(H.shape)))))
    print('Solving for parameter update')
    d = sp.linalg.spsolve(H, JTe)
    dp = d[:n_p]
    dqs = d[n_p:(n_p + (n_frames * n_q))].reshape([n_frames, n_q])
    dcs = d[-(n_frames * n_c):].reshape([n_frames, n_c])
    # Add the focal length and degenerate quaternion parameters back on as
    # null delta updates
    dcs = np.hstack([np.tile(np.array([0, 1]), (n_frames, 1)), dcs])

    new_p = p + dp
    new_qs = qs + dqs
    new_cs = np.array([camera_parameters_update(c, dc)
                       for c, dc in zip(cs, dcs)])

    params = {
        'p': new_p,
        'qs': new_qs,
        'cs': new_cs,
        'dp': dp,
        'dqs': dqs,
        'dcs': dcs,
    }

    if compute_costs:
        c = {k: np.array(v) for k, v in costs.items()}

        err_s_id = (p ** 2).sum()
        err_s_exp = (qs ** 2).sum()
        err_sm = ((qs[:-2] - 2 * qs[1:-1] + qs[2:]) ** 2).sum()

        err_f_tot = c['err_f'].sum() * c_f / (n_c * n_samples)
        err_l_tot = c['err_l'].sum()

        total_energy = (err_f_tot +
                        c_l * err_l_tot +
                        c_id * err_s_id +
                        c_exp * err_s_exp +
                        c_sm * err_sm)

        c['total_energy'] = total_energy
        c['err_s_id'] = (c_id, err_s_id)
        c['err_s_exp'] = (c_exp, err_s_exp)
        c['err_sm'] = (c_sm, err_sm)
        c['err_f_tot'] = err_f_tot
        c['err_l_tot'] = (c_l, err_l_tot)

        print_cost_dict(c)

        params['costs'] = c
    return params


def fit_video(images, mm, id_indices, exp_indices, template_camera,
              p, qs, cs, c_f=1, c_l=1, c_id=1, c_exp=1, c_sm=1, lm_group=None,
              n_samples=1000, n_iters=10, compute_costs=True):
    params = [
        {
            "p": p,
            "qs": qs,
            "cs": cs
        }]

    for i in range(1, n_iters + 1):
        print('{} / {}'.format(i, n_iters))
        # retrieve the last used parameters and pass them into the increment
        l = params[-1]
        t1 = time()
        incs = increment_parameters(images, mm, id_indices, exp_indices,
                                    template_camera, l['p'], l['qs'], l['cs'],
                                    c_f=c_f, c_l=c_l, c_id=c_id, c_exp=c_exp,
                                    c_sm=c_sm,
                                    lm_group=lm_group, n_samples=n_samples,
                                    compute_costs=compute_costs)
        # update the parameter list
        params.append(incs)
        # And report the time taken for the iteration.
        dt = int(time() - t1)
        print('Iteration {} complete in {}\n'.format(i, timedelta(seconds=dt)))
    return params


def fit_image(image, mm, id_indices, exp_indices, template_camera,
              p, q, c, c_f=1, c_l=1, c_id=1, c_exp=1, lm_group=None,
              n_samples=1000, n_iters=10, compute_costs=True):
    # fit image is the same as fit_video, just for a single length video.
    return fit_video(
        [image], mm, id_indices, exp_indices, template_camera,
        p, q[None, :], c[None, :], c_f=c_f, c_l=c_l,
        c_id=c_id, c_exp=c_exp, c_sm=0, lm_group=lm_group,
        n_samples=n_samples, n_iters=n_iters, compute_costs=compute_costs
    )


def instance_for_params(mm, id_ind, exp_ind, template_camera, p, q,
                        c):
    shape_params = np.zeros(mm.shape_model.n_active_components)
    shape_params[id_ind] = p
    shape_params[exp_ind] = q
    instance = mm.shape_model.instance(shape_params, normalized_weights=True)
    camera = template_camera.from_vector(c)
    instance_in_img = camera.apply(instance)
    return {
        'instance': instance,
        'instance_in_img': instance_in_img,
        'camera': camera
    }


def render_iteration(mm, id_ind, exp_ind, img_shape, camera, params,
                     img_index, iteration):
    from lsfm.visualize import lambertian_shading
    from menpo3d.rasterize import rasterize_mesh
    params_i = params[iteration]
    c_i = params_i['cs'][img_index]
    p_i = params_i['p']
    q_i = params_i['qs'][img_index]
    i_in_img = instance_for_params(mm, id_ind, exp_ind, camera,
                                   p_i, q_i, c_i)['instance_in_img']
    [x_r, y_r, z_r] = i_in_img.range()
    av_xy_r = (x_r + y_r) / 2.0
    i_in_img = Scale([1, 1, av_xy_r / z_r]).apply(i_in_img)
    mesh_in_img_lit = lambertian_shading(i_in_img.as_colouredtrimesh())
    return rasterize_mesh(mesh_in_img_lit, img_shape).as_unmasked()


def render_initialization(images, mm, id_indices, exp_indices, template_camera,
                          p, qs, cs, img_index):
    from lsfm.visualize import lambertian_shading
    from menpo3d.rasterize import rasterize_mesh
    c_i = cs[img_index]
    q_i = qs[img_index]
    i_in_img = instance_for_params(mm, id_indices, exp_indices,
                                   template_camera,
                                   p, q_i, c_i)['instance_in_img']
    [x_r, y_r, z_r] = i_in_img.range()
    av_xy_r = (x_r + y_r) / 2.0
    i_in_img = Scale([1, 1, av_xy_r / z_r]).apply(i_in_img)
    mesh_in_img_lit = lambertian_shading(i_in_img.as_colouredtrimesh())
    return rasterize_mesh(mesh_in_img_lit, images[0].shape).as_unmasked()


def print_single_cost(k, c, tot):
    if isinstance(c, tuple):
        key = '{:03.0%} | {:>12}'.format((c[0] * c[1]) / tot, k)
        val = '{:>12.2f} x {:>12.2f} = {:.2f}'.format(c[0], c[1], c[0] * c[1])
    else:
        key = '{:03.0%} | {:>12}'.format(c / tot, k)
        val = '{:.2f}'.format(c)
    print('{:>20}: {}'.format(key, val))


def print_cost_dict(d):
    print('------------------------------------------------------------------')
    print_single_cost('total_energy', d['total_energy'], d['total_energy'])
    print('------------------------------------------------------------------')
    for k in ['err_f_tot', 'err_l_tot', 'err_s_id',
              'err_s_exp', 'err_sm']:
        print_single_cost(k, d[k], d['total_energy'])
    print('------------------------------------------------------------------')
    for k in ['err_f', 'err_l']:
        print('{} (median over frames): {:.2f}'.format(k, np.median(d[k])))
    print('------------------------------------------------------------------')
