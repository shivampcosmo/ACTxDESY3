import numpy as np
import pdb
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import LSS_funcs as hmf
from scipy import interpolate

Color = ['#0072b1', '#009d73', '#d45e00', 'k', 'grey', 'yellow']

# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True


matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.unicode'] = False


# font = {'size': 18}
# matplotlib.rc('font', **font)
# plt.rc('text', usetex=False)
# plt.rc('font', family='serif')


# Get ellipse given 2D covariance matrix
def get_ellipse(cov, contour_levels=[1]):
    sigmax = np.sqrt(cov[0, 0])
    sigmay = np.sqrt(cov[1, 1])
    sigmaxy = cov[1, 0]

    # Silly
    all_sigma_list = np.array([1, 2, 3])
    all_alpha_list = np.array([1.52, 2.48, 3.44])
    alpha_list = np.zeros(len(contour_levels))
    for ii in range(len(contour_levels)):
        match = np.where(all_sigma_list == contour_levels[ii])[0]
        alpha_list[ii] = all_alpha_list[match]

    num_points = 5000
    rot_points = np.zeros((len(alpha_list), 2, 2 * num_points))

    for ai in range(len(alpha_list)):
        alpha = alpha_list[ai]
        alphasquare = alpha ** 2.

        # Rotation angle
        if sigmax != sigmay:
            theta = 0.5 * np.arctan((2. * sigmaxy / (sigmax ** 2. - sigmay ** 2.)))
        if sigmaxy == 0.:
            theta = 0.
        if sigmaxy != 0. and sigmax == sigmay:
            # is this correct?
            theta = np.pi / 4.

        # Determine major and minor axes
        eigval, eigvec = np.linalg.eig(cov)
        major = np.sqrt(eigval[0])
        minor = np.sqrt(eigval[1])
        if sigmax > sigmay:
            asquare = major ** 2.
            bsquare = minor ** 2.
        if sigmax <= sigmay:
            asquare = minor ** 2.
            bsquare = major ** 2.
            theta += np.pi / 2.

        # Get ellipse defined by pts
        xx = np.linspace(-0.99999 * np.sqrt(alphasquare * asquare), 0.99999 * np.sqrt(alphasquare * asquare),
                         num=num_points)
        yy = np.sqrt(alphasquare * bsquare * (1 - (xx ** 2.) / (alphasquare * asquare)))
        minusy = -yy
        points = np.vstack((np.append(xx, -xx), np.append(yy, minusy)))

        # Rotation matrix
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for pi in range(2 * num_points):
            rot_points[ai, :, pi] = np.dot(rot_matrix, points[:, pi])

    return rot_points


def get_points_from_fisher(fisher_matrix_wpriors, indices, sigma_levels):
    cov = np.linalg.inv(fisher_matrix_wpriors)
    cov_reduced = cov[indices, :]
    cov_reduced = cov_reduced[:, indices]

    points = get_ellipse(cov_reduced, contour_levels=sigma_levels)
    return points


def get_points_from_cov(cov, indices, sigma_levels):
    cov_reduced = cov[indices, :]
    cov_reduced = cov_reduced[:, indices]

    points = get_ellipse(cov_reduced, contour_levels=sigma_levels)
    return points


def plot_contours(F_mat_wpriors_input, fid_values_input, param_labels_input, sigma_levels, save_name=None,
                  parameter_indices=None):
    if (parameter_indices is None):
        parameter_indices = np.arange(0, F_mat_wpriors_input.shape[0])
    fid_values = fid_values_input[parameter_indices]
    param_labels = np.array(param_labels_input)[parameter_indices]

    nparam = len(parameter_indices)
    # Marginalize first
    param_cov = (np.linalg.inv(F_mat_wpriors_input)[parameter_indices, :])[:, parameter_indices]

    # controls plotting range in units of standard deviation
    max_sd = 5.

    # Determine plot limits
    param_ranges = []
    for parami in range(nparam):
        varparami = param_cov[parami, parami]
        param_ranges.append((fid_values[parami] - max_sd * np.sqrt(varparami),
                             fid_values[parami] + max_sd * np.sqrt(varparami)))

    figx = nparam * 3.
    fig, ax = plt.subplots(nparam, nparam, figsize=(figx, figx))
    # fig.subplots_adjust(hspace = 0., wspace = 0.)

    # rows
    for parami in range(nparam):
        # cols
        for paramj in range(nparam):
            # ellipses on lower triangle
            if parami > paramj:
                points = get_points_from_cov(param_cov, [parami, paramj], sigma_levels)
                for ii in range(len(sigma_levels)):
                    ax[parami, paramj].plot(points[ii, 1, :] + fid_values[paramj],
                                            points[ii, 0, :] + fid_values[parami], lw=2)
                    ax[parami, paramj].set_xlim((param_ranges[paramj][0], param_ranges[paramj][1]))
                    ax[parami, paramj].set_ylim((param_ranges[parami][0], param_ranges[parami][1]))
            # Get rid of upper triangle
            if paramj > parami:
                fig.delaxes(ax[parami, paramj])
            # 1d gaussian on diagonal
            if parami == paramj:
                varparami = param_cov[parami, paramj]
                xx = np.linspace(param_ranges[parami][0], param_ranges[parami][1], num=100)
                yy = np.exp(-((xx - fid_values[parami]) ** 2.) / (2. * varparami))
                if nparam > 1:
                    ax_handle = ax[parami, paramj]
                else:
                    ax_handle = ax
                ax_handle.plot(xx, yy, lw=2)
                ax_handle.set_xlim((param_ranges[parami][0], param_ranges[parami][1]))
                ax_handle.set_ylim((0., 1.1 * np.max(yy)))
                ax_handle.set_yticklabels([])
                ax_handle.yaxis.set_ticks_position('none')
                ax_handle.set_yticklabels([])
            if paramj > 0:
                ax[parami, paramj].set_yticklabels([])
            if paramj == 0:
                if nparam > 1:
                    ax_handle = ax[parami, paramj]
                else:
                    ax_handle = ax
                ax_handle.set_ylabel(param_labels[parami], fontsize=12)
            if parami == nparam - 1:
                if nparam > 1:
                    ax_handle = ax[parami, paramj]
                else:
                    ax_handle = ax
                ax_handle.set_xlabel(param_labels[paramj], fontsize=12)

    if save_name is not None:
        fig.savefig(save_name)

    return fig


def plot_Cls(l_array, Cl_dict, Cl_noise_yy_l_array, Cl_noise_gg_l_array, save_suffix, plot_dir, cov_fid_dict_G=None,
             cov_fid_dict_NG=None, show_errorbars=True, show_noise_curves=False, show_miscentering=True,
             ind_select_survey=None):
    l_array_full = np.logspace(-3.4, 4.1, 12000)

    l_array = l_array
    Cl_gg_1h_array = Cl_dict['gg']['1h']
    Cl_gg_2h_array = Cl_dict['gg']['2h']
    Cl_gg_total_array = Cl_dict['gg']['total']
    Cl_yg_1h_array = Cl_dict['yg']['1h']
    Cl_yg_2h_array = Cl_dict['yg']['2h']
    Cl_yg_total_array = Cl_dict['yg']['total']
    Cl_yy_1h_array = Cl_dict['yy']['1h']
    Cl_yy_2h_array = Cl_dict['yy']['2h']
    Cl_yy_total_array = Cl_dict['yy']['total']

    # Cl_gg_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_gg_total_array), fill_value='extrapolate')
    # Cl_gg_full = np.exp(Cl_gg_interp(np.log(l_array_full)))
    #
    # Cl_yg_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yg_total_array), fill_value='extrapolate')
    # Cl_yg_full = np.exp(Cl_yg_interp(np.log(l_array_full)))
    #
    # Cl_yy_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yy_total_array), fill_value='extrapolate')
    # Cl_yy_full = np.exp(Cl_yy_interp(np.log(l_array_full)))
    #
    # theta_min = 2.5
    # theta_max = 250.
    # theta_array_arcmin = np.logspace(np.log10(theta_min), np.log10(theta_max), 20)
    # theta_array_rad = theta_array_arcmin * (np.pi / 180.) * (1. / 60.)
    # wtheta_gg = np.zeros(len(theta_array_rad))
    # wtheta_yg = np.zeros(len(theta_array_rad))
    # wtheta_yy = np.zeros(len(theta_array_rad))
    #
    # for j in range(len(theta_array_rad)):
    #     wtheta_gg[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_gg_full)
    #     wtheta_yg[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yg_full)
    #     wtheta_yy[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yy_full)
    #
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # ax.plot(theta_array_arcmin, wtheta_gg, color='blue', marker='', linestyle='-',
    #         label=r'$w^{gg}(\theta)$')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_ylabel(r'$w(\theta)$', size=22)
    # ax.set_xlabel(r'$\theta$', size=22)
    # ax.legend(fontsize=20, frameon=False)
    # plt.tick_params(axis='both', which='major', labelsize=15)
    # plt.tick_params(axis='both', which='minor', labelsize=15)
    # plt.tight_layout()
    # plt.savefig(plot_dir + 'wtheta_gg_total' + save_suffix + '.png')
    # plt.close()
    #
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # ax.plot(theta_array_arcmin, wtheta_yg, color='blue', marker='', linestyle='-',
    #         label=r'$w^{yg}(\theta)$')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_ylabel(r'$w(\theta)$', size=22)
    # ax.set_xlabel(r'$\theta$', size=22)
    # ax.legend(fontsize=20, frameon=False)
    # plt.tick_params(axis='both', which='major', labelsize=15)
    # plt.tick_params(axis='both', which='minor', labelsize=15)
    # plt.tight_layout()
    # plt.savefig(plot_dir + 'wtheta_yg_total' + save_suffix + '.png')
    # plt.close()
    #
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # ax.plot(theta_array_arcmin, wtheta_yy, color='blue', marker='', linestyle='-',
    #         label=r'$w^{yy}(\theta)$')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_ylabel(r'$w(\theta)$', size=22)
    # ax.set_xlabel(r'$\theta$', size=22)
    # ax.legend(fontsize=20, frameon=False, loc='upper left')
    # plt.tick_params(axis='both', which='major', labelsize=15)
    # plt.tick_params(axis='both', which='minor', labelsize=15)
    # plt.tight_layout()
    # plt.savefig(plot_dir + 'wtheta_yy_total' + save_suffix + '.png')
    # plt.close()

    if ind_select_survey is None:
        ind_select_survey = np.arange(len(l_array))

    l_array_survey = l_array[ind_select_survey]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * Cl_gg_1h_array[ind_select_survey] / (2 * np.pi),
            color='blue', marker='',
            linestyle='-.', label=r'1-Halo')
    ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * Cl_gg_2h_array[ind_select_survey] / (2 * np.pi),
            color='red', marker='',
            linestyle='--', label=r'2-Halo')
    if not show_errorbars:
        ax.plot(l_array_survey,
                l_array_survey * (l_array_survey + 1.) * Cl_gg_total_array[ind_select_survey] / (2 * np.pi), color='k',
                marker='', linestyle='-', label=r'Total')
    if show_noise_curves:
        ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * Cl_noise_gg_l_array / (2 * np.pi),
                color='brown',
                marker='', linestyle=':', label=r'Shot Noise')
    if cov_fid_dict_G is not None:
        if 'gg_gg' in cov_fid_dict_G.keys():
            sigG_gg_array = np.sqrt(np.diag(cov_fid_dict_G['gg_gg']))
            sigNG_gg_array = np.sqrt(np.diag(cov_fid_dict_NG['gg_gg']))
            if show_errorbars:
                ax.errorbar(l_array_survey,
                            l_array_survey * (l_array_survey + 1.) * Cl_gg_total_array[ind_select_survey] / (2 * np.pi),
                            l_array_survey * (l_array_survey + 1.) * (sigG_gg_array + sigNG_gg_array) / (2 * np.pi),
                            color='k',
                            marker='', linestyle='-', label='Total')
            else:
                ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * sigG_gg_array / (2 * np.pi),
                        color='magenta',
                        marker='', linestyle='-.', label=r'Gaussian Noise')
                ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * sigNG_gg_array / (2 * np.pi),
                        color='green',
                        marker='', linestyle='-.', label=r'Non-Gaussian Noise')
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_ylim(1e-2, 10)
    ax.set_ylabel(r'$\ell \ (\ell + 1) \ C^{gg}_\ell/ 2 \pi$', size=22)
    ax.set_xlabel(r'$\ell$', size=22)
    ax.legend(fontsize=20, frameon=False, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    plt.savefig(plot_dir + 'Cl_gg_total' + save_suffix + '.png')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * Cl_yg_1h_array[ind_select_survey] / (2 * np.pi),
            color='blue', marker='',
            linestyle='-.', label=r'1-Halo')
    ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * Cl_yg_2h_array[ind_select_survey] / (2 * np.pi),
            color='red', marker='',
            linestyle='--', label=r'2-Halo')
    if not show_errorbars:
        ax.plot(l_array_survey,
                l_array_survey * (l_array_survey + 1.) * Cl_yg_total_array[ind_select_survey] / (2 * np.pi), color='k',
                marker='', linestyle='-', label=r'Total')

    if show_miscentering:
        ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * (
                Cl_yg_1h_array[ind_select_survey] + Cl_yg_2h_array[ind_select_survey]) / (2 * np.pi),
                color='orange',
                marker='', linestyle='-', label=r'Zero Miscentering')

    if cov_fid_dict_G is not None:
        if 'gy_gy' in cov_fid_dict_G.keys():
            sigG_gy_array = np.sqrt(np.diag(cov_fid_dict_G['gy_gy']))
            sigNG_gy_array = np.sqrt(np.diag(cov_fid_dict_NG['gy_gy']))
            if show_errorbars:
                ax.errorbar(l_array_survey,
                            l_array_survey * (l_array_survey + 1.) * Cl_yg_total_array[ind_select_survey] / (2 * np.pi),
                            l_array_survey * (l_array_survey + 1.) * (sigG_gy_array + sigNG_gy_array) / (2 * np.pi),
                            color='k',
                            marker='', linestyle='-', label='Total')
            else:
                ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * sigG_gy_array / (2 * np.pi),
                        color='magenta',
                        marker='', linestyle='-.', label=r'Gaussian Noise')
                ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * sigNG_gy_array / (2 * np.pi),
                        color='green',
                        marker='', linestyle='-.', label=r'Non-Gaussian Noise')
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_ylim(1e-9,1e-6)
    ax.set_ylabel(r'$\ell \ (\ell + 1) \ C^{yg}_\ell/ 2 \pi$', size=22)
    ax.set_xlabel(r'$\ell$', size=22)
    ax.legend(fontsize=20, frameon=False, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    plt.savefig(plot_dir + 'Cl_yg_total' + save_suffix + '.png')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * Cl_yy_1h_array[ind_select_survey] / (2 * np.pi),
            color='blue', marker='',
            linestyle='-.', label=r'1-Halo')
    ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * Cl_yy_2h_array[ind_select_survey] / (2 * np.pi),
            color='red', marker='',
            linestyle='--', label=r'2-Halo')
    if not show_errorbars:
        ax.plot(l_array_survey,
                l_array_survey * (l_array_survey + 1.) * Cl_yy_total_array[ind_select_survey] / (2 * np.pi), color='k',
                marker='', linestyle='-', label=r'Total')
    if show_noise_curves:
        ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * Cl_noise_yy_l_array / (2 * np.pi),
                color='brown',
                marker='', linestyle=':', label=r'Noise')
    if cov_fid_dict_G is not None:
        if 'yy_yy' in cov_fid_dict_G.keys():
            sigG_yy_array = np.sqrt(np.diag(cov_fid_dict_G['yy_yy']))
            sigNG_yy_array = np.sqrt(np.diag(cov_fid_dict_NG['yy_yy']))
            if show_errorbars:
                ax.errorbar(l_array_survey,
                            l_array_survey * (l_array_survey + 1.) * Cl_yy_total_array[ind_select_survey] / (2 * np.pi),
                            l_array_survey * (l_array_survey + 1.) * (sigG_yy_array + sigNG_yy_array) / (2 * np.pi),
                            color='k',
                            marker='', linestyle='-', label='Total')
            else:
                ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * sigG_yy_array / (2 * np.pi),
                        color='magenta',
                        marker='', linestyle='-.', label=r'Gaussian Noise')
                ax.plot(l_array_survey, l_array_survey * (l_array_survey + 1.) * sigNG_yy_array / (2 * np.pi),
                        color='green',
                        marker='', linestyle='-.', label=r'Non-Gaussian Noise')
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_ylim(1e-15, 5e-12)
    ax.set_ylabel(r'$\ell \ (\ell + 1) \ C^{yy}_\ell/ 2 \pi$', size=22)
    ax.set_xlabel(r'$\ell$', size=22)
    ax.legend(fontsize=20, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    plt.savefig(plot_dir + 'Cl_yy_total' + save_suffix + '.png')
    plt.close()
    # pdb.set_trace()


def plot_wtheta(l_array, Cl_dict, theta_min, theta_max, ntheta, save_suffix, plot_dir, cov_wtheta_G_dict=None,
                show_yg_1h_2h=True, chi_zeval=None):
    theta_array_all = np.logspace(np.log10(theta_min), np.log10(theta_max), ntheta)
    theta_array = (theta_array_all[1:] + theta_array_all[:-1]) / 2.
    theta_array_rad = theta_array * (np.pi / 180.) * (1. / 60.)

    l_array_full = np.logspace(0, 4.1, 14000)
    Cl_gg_total_array = Cl_dict['gg']['total']
    Cl_yg_1h_array = Cl_dict['yg']['1h']
    Cl_yg_2h_array = Cl_dict['yg']['2h']
    Cl_yg_total_array = Cl_dict['yg']['total']
    Cl_yy_total_array = Cl_dict['yy']['total']

    # Cl_gg_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_gg_total_array), fill_value='extrapolate')
    # Cl_gg_full = np.exp(Cl_gg_interp(np.log(l_array_full)))
    #
    # Cl_yg_1h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yg_1h_array), fill_value='extrapolate')
    # Cl_yg_1h_full = np.exp(Cl_yg_1h_interp(np.log(l_array_full)))
    # Cl_yg_2h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yg_2h_array), fill_value='extrapolate')
    # Cl_yg_2h_full = np.exp(Cl_yg_2h_interp(np.log(l_array_full)))
    #
    # Cl_yy_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yy_total_array), fill_value='extrapolate')
    # Cl_yy_full = np.exp(Cl_yy_interp(np.log(l_array_full)))

    Cl_gg_interp = interpolate.interp1d(np.log(l_array), Cl_gg_total_array, fill_value=0.0, bounds_error=False)
    Cl_gg_full = Cl_gg_interp(np.log(l_array_full))

    Cl_yg_1h_interp = interpolate.interp1d(np.log(l_array), Cl_yg_1h_array, fill_value=0.0, bounds_error=False)
    Cl_yg_1h_full = Cl_yg_1h_interp(np.log(l_array_full))
    Cl_yg_2h_interp = interpolate.interp1d(np.log(l_array), Cl_yg_2h_array, fill_value=0.0, bounds_error=False)
    Cl_yg_2h_full = Cl_yg_2h_interp(np.log(l_array_full))

    Cl_yy_interp = interpolate.interp1d(np.log(l_array), Cl_yy_total_array, fill_value=0.0, bounds_error=False)
    Cl_yy_full = Cl_yy_interp(np.log(l_array_full))

    theta_array_arcmin = theta_array_rad * (180. / np.pi) * 60.0
    wtheta_gg = np.zeros(len(theta_array_rad))
    wtheta_yg_1h = np.zeros(len(theta_array_rad))
    wtheta_yg_2h = np.zeros(len(theta_array_rad))
    wtheta_yg = np.zeros(len(theta_array_rad))
    wtheta_yy = np.zeros(len(theta_array_rad))

    for j in range(len(theta_array_rad)):
        wtheta_gg[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_gg_full)
        wtheta_yg_1h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yg_1h_full)
        wtheta_yg_2h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yg_2h_full)
        wtheta_yg[j] = wtheta_yg_1h[j] + wtheta_yg_2h[j]
        wtheta_yy[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yy_full)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if cov_wtheta_G_dict is not None:
        if 'gg_gg' in cov_wtheta_G_dict.keys():
            sigG_gg_array = np.sqrt(np.diag(cov_wtheta_G_dict['gg_gg']))

            ax.errorbar(theta_array_arcmin, wtheta_gg, sigG_gg_array, color='blue', marker='', linestyle='-',
                        label=r'$w^{gg}(\theta)$')
        else:
            ax.plot(theta_array_arcmin, wtheta_gg, color='blue', marker='', linestyle='-', label=r'$w^{gg}(\theta)$')
    else:
        ax.plot(theta_array_arcmin, wtheta_gg, color='blue', marker='', linestyle='-', label=r'$w^{gg}(\theta)$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$w(\theta)$', size=22)
    ax.set_xlabel(r'$\theta$', size=22)
    ax.legend(fontsize=20, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    plt.savefig(plot_dir + 'wtheta_gg_total' + save_suffix + '.png')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if chi_zeval is not None:
        x_array = theta_array_rad * chi_zeval
        # ax.set_ylim(10 ** -9, 4*10**-6)
        ax.set_xlabel(r'$r$ (Mpc/h)', size=22)
    else:
        x_array = theta_array_arcmin
        ax.set_xlim(0.1, 100.)
        ax.set_ylim(4 * 10 ** -7, np.max(wtheta_yg) * 2.)
        ax.set_xlabel(r'$\theta$ (arcmin)', size=22)

    if cov_wtheta_G_dict is not None:
        if 'gy_gy' in cov_wtheta_G_dict.keys():
            sigG_gy_array = np.sqrt(np.diag(cov_wtheta_G_dict['gy_gy']))

            print('snr = ', np.sqrt(np.dot(np.dot(wtheta_yg, np.linalg.inv(cov_wtheta_G_dict['gy_gy'])), wtheta_yg.T)))

            # pdb.set_trace()
            ax.errorbar(x_array, wtheta_yg, sigG_gy_array, color='black', marker='', linestyle='-',
                        label=r'Total')
            if show_yg_1h_2h:
                ax.plot(x_array, wtheta_yg_1h, color='red', marker='', linestyle=':',
                        label=r'1-Halo')
                ax.plot(x_array, wtheta_yg_2h, color='blue', marker='', linestyle=':',
                        label=r'2-Halo')
        else:
            ax.plot(x_array, wtheta_yg, color='black', marker='', linestyle='-',
                    label=r'Total')
            if show_yg_1h_2h:
                ax.plot(x_array, wtheta_yg_1h, color='red', marker='', linestyle=':',
                        label=r'1-Halo')
                ax.plot(x_array, wtheta_yg_2h, color='blue', marker='', linestyle=':',
                        label=r'2-Halo')
    else:
        ax.plot(x_array, wtheta_yg, color='black', marker='', linestyle='-',
                label=r'Total')
        if show_yg_1h_2h:
            ax.plot(x_array, wtheta_yg_1h, color='red', marker='', linestyle=':',
                    label=r'1-Halo')
            ax.plot(x_array, wtheta_yg_2h, color='blue', marker='', linestyle=':',
                    label=r'2-Halo')

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_ylabel(r'$\xi^{yg}$', size=22)

    ax.legend(fontsize=20, frameon=False)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    plt.savefig(plot_dir + 'wtheta_yg_total' + save_suffix + '.png')
    plt.close()

    if cov_wtheta_G_dict is not None:
        if 'gy_gy' in cov_wtheta_G_dict.keys():
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            sigG_gy_array = np.sqrt(np.diag(cov_wtheta_G_dict['gy_gy']))
            ax.plot(theta_array_arcmin, wtheta_yg / sigG_gy_array, color='black', marker='', linestyle='-',
                    label=r'SNR')
            # ax.set_yscale('log')
            ax.set_xscale('log')
            # ax.set_ylim(10**-8,np.max(wtheta_yg) * 2.)
            ax.set_xlim(0.1, 100.)
            ax.set_ylabel(r'$\xi^{yg}(\theta)$', size=22)
            ax.set_xlabel(r'$\theta$ (arcmin)', size=22)
            ax.legend(fontsize=20, frameon=False)
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.tick_params(axis='both', which='minor', labelsize=15)
            plt.tight_layout()
            plt.savefig(plot_dir + 'wtheta_yg_snr_total' + save_suffix + '.png')
            plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if cov_wtheta_G_dict is not None:
        if 'yy_yy' in cov_wtheta_G_dict.keys():
            sigG_yy_array = np.sqrt(np.diag(cov_wtheta_G_dict['yy_yy']))
            ax.errorbar(theta_array_arcmin, wtheta_yg, sigG_yy_array, color='blue', marker='', linestyle='-',
                        label=r'$w^{yy}(\theta)$')
        else:
            ax.plot(theta_array_arcmin, wtheta_yy, color='blue', marker='', linestyle='-',
                    label=r'$w^{yy}(\theta)$')
    else:
        ax.plot(theta_array_arcmin, wtheta_yy, color='blue', marker='', linestyle='-',
                label=r'$w^{yy}(\theta)$')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$w(\theta)$', size=22)
    ax.set_xlabel(r'$\theta$', size=22)
    ax.legend(fontsize=20, frameon=False, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    plt.savefig(plot_dir + 'wtheta_yy_total' + save_suffix + '.png')
    plt.close()


def plot_fishermat(F_mat, plot_dir='./', save_suffix='', save_plots=False):

    print('Fmat : ', F_mat)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fmat = ax.imshow(F_mat)
    fig.colorbar(fmat, ax=ax)
    fig.savefig(plot_dir + 'fisher_mat' + save_suffix + '.png')

    return fig


def plot_cov(cov_fid_G, cov_fid_NG, plot_dir='./', save_suffix='', save_plots=False):
    cov_fid = cov_fid_G + cov_fid_NG

    fig1, ax1 = plt.subplots(1, 1, figsize=(13, 13))
    cov = ax1.imshow(np.log(np.abs(cov_fid_G)))
    fig1.colorbar(cov, ax=ax1)
    fig1.savefig(plot_dir + 'logabs_covG_mat' + save_suffix + '.png')

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    cov = ax2.imshow(np.log(np.abs(cov_fid_NG)))
    fig2.colorbar(cov, ax=ax2)
    fig2.savefig(plot_dir + 'logabs_covNG_mat' + save_suffix + '.png')

    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))
    cov = ax3.imshow(np.log(np.abs(cov_fid)))
    fig3.colorbar(cov, ax=ax3)
    fig3.savefig(plot_dir + 'logabs_covTotal_mat' + save_suffix + '.png')

    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8))
    corr = ax4.imshow(hmf.get_corr(cov_fid))
    fig4.colorbar(corr, ax=ax4)
    fig4.savefig(plot_dir + 'corr_mat' + save_suffix + '.png')

    return [fig1, fig2, fig3, fig4]


def plot_Ptheta_samples(x_array, Ptheta_mat, Ptheta_fid, plot_dir='./', percentiles=None, do_samples=False, xlim=None,
                        ylim=None,
                        save_plots=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    nsamples, numx = Ptheta_mat.shape

    if percentiles is None:
        percentiles = [16., 84.]

    print("percentiles = ", percentiles)


    Pr_low = np.percentile(Ptheta_mat, percentiles[0], axis=0)
    Pr_high = np.percentile(Ptheta_mat, percentiles[1], axis=0)
    ax.fill_between(x_array, Ptheta_mat, Ptheta_mat, color='blue', alpha=0.4, label=r'${\rm Forecast}$')
    if do_samples:
        for j in range(nsamples):
            ax.plot(x_array, Ptheta_mat[j, :], color='blue', linewidth=1., alpha=0.6)

    ax.plot(x_array, Ptheta_fid[0][0], color='red', label=r'${\rm Fiducial}$', lw='2')

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta$', size=20)
    ax.set_ylabel(r'$ P_e \ (\rm{eV \ cm^{-3}})$', size=20)
    legend = ax.legend(loc='upper right', fontsize=20, framealpha=1.0)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_linewidth(0)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    fig.tight_layout()

    if save_plots:
        fig.savefig(plot_dir + 'Pr_relation_' + save_suffix + '.png')

    return fig


def plot_Pr_samples(x_array, Pr_mat, Pr_fid, plot_dir='./', percentiles=None, do_samples=None, xlim=None, ylim=None,
                    save_plots=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    nsamples, numx = Pr_mat.shape

    if percentiles is None:
        percentiles = [16., 84.]

    print("percentiles = ", percentiles)

    Pr_low = np.percentile(Pr_mat, percentiles[0], axis=0)
    Pr_high = np.percentile(Pr_mat, percentiles[1], axis=0)
    ax.fill_between(x_array, Pr_low, Pr_high, color='blue', alpha=0.4, label=r'${\rm Forecast}$')
    if do_samples:
        for j in range(nsamples):
            ax.plot(x_array, Pr_mat[j, :], color='blue', linewidth=1., alpha=0.6)

    ax.plot(x_array, Pr_fid[0][0], color='red', label=r'${\rm Fiducial}$', lw='2')

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\rm{r}/\rm{r_{500c}}$', size=20)
    ax.set_ylabel(r'$ P_e \ (\rm{eV \ cm^{-3}})$', size=20)
    legend = ax.legend(loc='upper right', fontsize=20, framealpha=1.0)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_linewidth(0)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    fig.tight_layout()

    if save_plots:
        fig.savefig(plot_dir + 'Pr_relation_' + save_suffix + '.png')

    return fig


def plot_wthetay_samples(theta_array, wthetayg_mat, wthetayg_fid, plot_dir='./', percentiles=None, do_samples=None,
                         xlim=None, ylim=None,
                         save_plots=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    nsamples, numx = wthetayg_mat.shape

    if percentiles is None:
        percentiles = [16., 84.]

    print("percentiles = ", percentiles)

    Pr_low = np.percentile(wthetayg_mat, percentiles[0], axis=0)
    Pr_high = np.percentile(wthetayg_mat, percentiles[1], axis=0)
    ax.fill_between(theta_array, Pr_low, Pr_high, color='blue', alpha=0.4, label=r'${\rm Forecast}$')
    if do_samples:
        for j in range(nsamples):
            ax.plot(theta_array, wthetayg_mat[j, :], color='blue', linewidth=1., alpha=0.6)

    ax.plot(theta_array, wthetayg_fid, color='red', label=r'${\rm Fiducial}$', lw='2')

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\theta$ (arcmin)', size=20)
    ax.set_ylabel(r'$ \xi^{yg} $', size=20)
    legend = ax.legend(loc='upper right', fontsize=20, framealpha=1.0)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_linewidth(0)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    fig.tight_layout()

    if save_plots:
        fig.savefig(plot_dir + 'wthetayg_from_samples_' + save_suffix + '.png')

    return fig


def plot_w_yg_rp_samples_w_errorbars(w_yg_params_dict, w_yg_data_dict, plot_dir='./', percentiles=None, do_samples=None,
                                     xlim=None, ylim=None,
                                     save_plots=False):
    rp_params, wthetayg_mat, wthetayg_fid = w_yg_params_dict['x'], w_yg_params_dict['curves'], w_yg_params_dict['fid']

    rp_data, wthetayg_data, sig_yg_data = w_yg_data_dict['x'], w_yg_data_dict['y'], w_yg_data_dict['sig']

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    nsamples, numx = wthetayg_mat.shape

    if percentiles is None:
        percentiles = [16., 84.]

    print("percentiles = ", percentiles)

    Pr_low = np.percentile(wthetayg_mat, percentiles[0], axis=0)
    Pr_high = np.percentile(wthetayg_mat, percentiles[1], axis=0)
    ax.fill_between(rp_params, Pr_low, Pr_high, color='blue', alpha=0.4, label=r'${\rm Forecast}$')
    if do_samples:
        for j in range(nsamples):
            ax.plot(rp_params, wthetayg_mat[j, :], color='blue', linewidth=1., alpha=0.6)

    ax.plot(rp_params, wthetayg_fid, color='red', label=r'${\rm Fiducial}$', lw='2')

    ax.errorbar(rp_data, wthetayg_data, sig_yg_data, color='k', label=r'${\rm Measurement}$', linestyle='', marker='o')

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r$ (Mpc/h)', size=20)
    ax.set_ylabel(r'$ \xi^{{\rm halo}-y} $', size=20)

    ax.legend(fontsize=16, frameon=False)

    # legend = ax.legend(loc='upper right', fontsize=20, framealpha=1.0)
    # frame = legend.get_frame()
    # frame.set_facecolor('white')
    # frame.set_linewidth(0)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    fig.tight_layout()

    if save_plots:
        fig.savefig(plot_dir + 'w_rpyg_from_samples_' + save_suffix + '.png')

    return fig


def plot_w_yg_rp_samples_w_errorbars_relative(w_yg_params_dict, w_yg_data_dict, plot_dir='./', percentiles=None,
                                              do_samples=None, xlim=None, ylim=None,
                                              save_plots=False):
    rp_params, wthetayg_mat, wthetayg_fid = w_yg_params_dict['x'], w_yg_params_dict['curves'], w_yg_params_dict['fid']

    rp_data, wthetayg_data, sig_yg_data = w_yg_data_dict['x'], w_yg_data_dict['y'], w_yg_data_dict['sig']

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    nsamples, numx = wthetayg_mat.shape

    if percentiles is None:
        percentiles = [16., 84.]

    print("percentiles = ", percentiles)

    Pr_low = np.percentile(wthetayg_mat, percentiles[0], axis=0)
    Pr_high = np.percentile(wthetayg_mat, percentiles[1], axis=0)

    Pr_low_interp = interpolate.interp1d(np.log(rp_params), Pr_low, fill_value='extrapolate')
    Pr_high_interp = interpolate.interp1d(np.log(rp_params), Pr_high, fill_value='extrapolate')

    wthetayg_fid_interp = interpolate.interp1d(np.log(rp_params), wthetayg_fid, fill_value='extrapolate')

    Pr_low_rpdata = (Pr_low_interp(np.log(rp_data)))
    Pr_high_rpdata = (Pr_high_interp(np.log(rp_data)))
    wthetayg_fid_rpdata = (wthetayg_fid_interp(np.log(rp_data)))

    ax.plot(rp_data, wthetayg_fid_rpdata / wthetayg_data, color='red', label=r'${\rm Fiducial}$', lw='2')

    ax.fill_between(rp_data, Pr_low_rpdata / wthetayg_data, Pr_high_rpdata / wthetayg_data, color='blue', alpha=0.4,
                    label=r'${\rm Forecast}$')

    if do_samples:
        for j in range(nsamples):
            wthetayg_mat_interp = interpolate.interp1d(np.log(rp_params), (wthetayg_mat[j, :]),
                                                       fill_value='extrapolate')
            wthetayg_mat_rpdata = (wthetayg_mat_interp(np.log(rp_data)))
            ax.plot(rp_data, wthetayg_mat_rpdata / wthetayg_data, color='blue', linewidth=1., alpha=0.3)

    ax.errorbar(rp_data, wthetayg_data / wthetayg_data, sig_yg_data / wthetayg_data, color='k',
                label=r'${\rm Measurement}$', linestyle='--', marker='o')

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel(r'$r$ (Mpc/h)', size=20)
    ax.set_ylabel(r'$ \xi^{{\rm halo}-y} / \xi_{measure}^{{\rm halo}-y}  $', size=20)

    ax.legend(fontsize=16, frameon=False)

    # legend = ax.legend(loc='upper right', fontsize=20, framealpha=1.0)
    # frame = legend.get_frame()
    # frame.set_facecolor('white')
    # frame.set_linewidth(0)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    fig.tight_layout()

    if save_plots:
        fig.savefig(plot_dir + 'w_rpyg_realtive_from_samples_' + save_suffix + '.png')

    # pdb.set_trace()

    return fig


def weight_array(mat, weights):
    weighted_mat = np.zeros(mat.shape)
    for mj in range(mat.shape[0]):
        ar = mat[:,mj]
        zipped = zip(ar, weights)
        weighted = []
        for i in zipped:
            # pdb.set_trace()
            for j in range(i[1]):
                weighted.append(i[0])

        weighted_mat[:mj] = weighted
    return weighted_mat

def weighted_percentile(data_mat, percents, weights=None):
    weighted_mat = np.zeros(data_mat.shape[1])
    for mj in range(data_mat.shape[1]):
        data = data_mat[:, mj]
        if weights is None:
            return np.percentile(data, percents)
        ind=np.argsort(data)
        d=data[ind]
        w=weights[ind]
        p=1.*w.cumsum()/w.sum()*100
        y=np.interp(percents, p, d)
        weighted_mat[mj] = y
    return weighted_mat

# def weighted_percentile(mat, q=None, w=None):
#     weighted_mat = np.zeros(mat.shape[1])
#     # pdb.set_trace()
#     for mj in range(mat.shape[1]):
#         a = mat[:, mj]
#         q = np.array(q) / 100.0
#         if w is None:
#             w = np.ones(a.size)
#         idx = np.argsort(a)
#         a_sort = a[idx]
#         w_sort = w[idx]
#
#         # Get the cumulative sum of weights
#         ecdf = np.cumsum(w_sort)
#
#         # Find the percentile index positions associated with the percentiles
#         p = q * (w.sum() - 1)
#
#         # Find the bounding indices (both low and high)
#         idx_low = np.searchsorted(ecdf, p, side='right')
#         idx_high = np.searchsorted(ecdf, p + 1, side='right')
#
#         # pdb.set_trace()
#         # if len(q) > 1:
#         #     idx_high[idx_high > ecdf.size - 1] = ecdf.size - 1
#         if idx_high > ecdf.size - 1:
#             idx_high = ecdf.size - 1
#
#         # Calculate the weights
#         weights_high = p - np.floor(p)
#         weights_low = 1.0 - weights_high
#
#         # Extract the low/high indexes and multiply by the corresponding weights
#         x1 = np.take(a_sort, idx_low) * weights_low
#         x2 = np.take(a_sort, idx_high) * weights_high
#
#         weighted_mat[mj] = np.add(x1, x2)
#
#     # Return the average
#     return weighted_mat

def plot_Clyg_samples(l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, l_array_survey, plot_dir='./', percentiles=None,
                      do_plot_relative=False, do_samples=None, xlim=None, ylim=None, save_plots=False, weights=None,
                      weight_Cls=False, Cls_to_plot='yg'):
    nsample = len(Cl_dicts_samples)
    nl = len(l_array)
    Cl_yg_mat = np.zeros((nsample, nl))
    Cl_yg1h_mat = np.zeros((nsample, nl))
    Cl_yg2h_mat = np.zeros((nsample, nl))

    ind_select_survey = np.where((l_array >= l_array_survey[0]) & (l_array <= l_array_survey[-1]))[0]

    for j in range(len(Cl_dicts_samples)):
        Cl_yg_mat[j, :] = Cl_dicts_samples[j][Cls_to_plot]['total']
        Cl_yg1h_mat[j, :] = Cl_dicts_samples[j][Cls_to_plot]['1h']
        Cl_yg2h_mat[j, :] = Cl_dicts_samples[j][Cls_to_plot]['2h']

    # Cl_yg_1h_array_fid = Cl_fid_dict['yg']['1h']
    # Cl_yg_2h_array_fid = Cl_fid_dict['yg']['2h']
    Cl_yg_total_array_fid = Cl_fid_dict[Cls_to_plot]['total']

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
    fig.subplots_adjust(wspace=0.0)

    axi = ax

    nsamples, numx = Cl_yg_mat.shape
    Cl_yg_low = np.percentile(Cl_yg_mat, percentiles[0], axis=0)
    Cl_yg_high = np.percentile(Cl_yg_mat, percentiles[1], axis=0)
    if weight_Cls:
        Cl_yg_low = weighted_percentile(Cl_yg_mat, percentiles[0], weights=weights)
        Cl_yg_high = weighted_percentile(Cl_yg_mat, percentiles[1], weights=weights)
        # pdb.set_trace()

    # Cl_yg_1h_low = np.percentile(Cl_yg1h_mat, percentiles[0], axis=0)
    # Cl_yg_1h_high = np.percentile(Cl_yg1h_mat, percentiles[1], axis=0)
    # Cl_yg_2h_low = np.percentile(Cl_yg2h_mat, percentiles[0], axis=0)
    # Cl_yg_2h_high = np.percentile(Cl_yg2h_mat, percentiles[1], axis=0)

    plotfact = (l_array[ind_select_survey]) * (l_array[ind_select_survey] + 1) / (2 * np.pi)

    if do_samples:
        for j in range(nsamples):
            axi.plot(l_array[ind_select_survey], (plotfact * Cl_yg_mat[j, :][ind_select_survey] / (
                        plotfact * Cl_yg_total_array_fid[ind_select_survey])), linewidth=0.025)

    if do_plot_relative:

        axi.fill_between(l_array[ind_select_survey], (
                plotfact * Cl_yg_low[ind_select_survey] / (plotfact * Cl_yg_total_array_fid[ind_select_survey])), (
                                 plotfact * Cl_yg_high[ind_select_survey] / (
                                 plotfact * Cl_yg_total_array_fid[ind_select_survey])), color='blue', alpha=0.3,
                         label='Total', linestyle='--')

        axi.errorbar(l_array[ind_select_survey], (plotfact * Cl_yg_total_array_fid[ind_select_survey] / (
                plotfact * Cl_yg_total_array_fid[ind_select_survey])),
                     plotfact * np.sqrt(np.diag(cov_fid)) / (plotfact * Cl_yg_total_array_fid[ind_select_survey]),
                     color='black', label=r'${\rm Fiducial Total}$', lw='0.8')
    else:
        axi.fill_between(l_array[ind_select_survey], (
                plotfact * Cl_yg_low[ind_select_survey]), (
                                 plotfact * Cl_yg_high[ind_select_survey]), color='blue', alpha=0.4,
                         label='Total', linestyle='--')

        axi.errorbar(l_array[ind_select_survey], (plotfact * Cl_yg_total_array_fid[ind_select_survey]),
                     plotfact * np.sqrt(np.diag(cov_fid)),
                     color='black', label=r'${\rm Fiducial Total}$', lw='0.8')

    axi.set_xscale('log')

    if not do_plot_relative:
        axi.set_yscale('log')

    if ylim is None:
        ylim = (0.9, 1.1)

    if xlim is not None:
        axi.set_xlim(xlim)

    axi.set_ylim(ylim)
    axi.set_xlabel(r'$\ell$', size=22)


    if Cls_to_plot == 'yg':
        if do_plot_relative:
            axi.set_ylabel(r'$C^{yg}_\ell$ ratio', size=22)
        else:
            axi.set_ylabel(r'$\ell \ (\ell + 1) \ C^{yg}_\ell/ 2 \pi$', size=22)

    if Cls_to_plot == 'yy':
        if do_plot_relative:
            axi.set_ylabel(r'$C^{yy}_\ell$ ratio', size=22)
        else:
            axi.set_ylabel(r'$\ell \ (\ell + 1) \ C^{yy}_\ell/ 2 \pi$', size=22)



    axi.tick_params(axis='both', which='major', labelsize=15)
    axi.tick_params(axis='both', which='minor', labelsize=15)

    fig.tight_layout()

    if save_plots:
        fig.savefig(plot_dir + 'Cl_' + Cls_to_plot + '_' + save_suffix + '.png')

    return fig


def plot_Pr_samples_relative(x_array, Pr_mat, Pr_fid, plot_dir='./', percentiles=None, do_samples=None, xlim=None,
                             ylim=None, save_plots=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    nsamples, numx = Pr_mat.shape

    Pr_relative_mat = np.zeros((nsamples, numx))
    for ii in range(nsamples):
        Pr_relative_mat[ii, :] = Pr_mat[ii, :] / Pr_fid[0][0]

    if percentiles is None:
        percentiles = [16., 84.]

    print("percentiles = ", percentiles)

    Pr_low = np.percentile(Pr_relative_mat, percentiles[0], axis=0)
    Pr_high = np.percentile(Pr_relative_mat, percentiles[1], axis=0)
    ax.fill_between(x_array, Pr_low, Pr_high, color='blue', alpha=0.4, label=r'${\rm Halos}$')

    if do_samples:
        for j in range(nsamples):
            ax.plot(x_array, Pr_relative_mat[j, :], color='blue', linewidth=1., alpha=0.4)

    ax.plot(x_array, Pr_fid[0][0] / Pr_fid[0][0], label=r'${\rm Fiducial}$', color='black', ls='dashed')

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\rm{r}/\rm{r_{500c}}$', size=20)
    ax.set_ylabel(r'$ \Delta P_e / P_e$', size=20)
    legend = ax.legend(loc='upper right', fontsize=20, framealpha=1.0)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_linewidth(0)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    fig.tight_layout()

    if save_plots:
        fig.savefig(plot_dir + 'Pr_relative_' + save_suffix + '.png')

    return fig


def plot_YM_relation_samples(M_array, integratedY_mat, YM_fid, do_split_params_massbins, split_mass_bins_min=None,
                             plot_dir='./', save_suffix='', save_plots=False, xlim=None, do_samples=True,
                             percentiles=None, ylim=None):

    print("xlim = ", xlim)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    nsamples, numM = integratedY_mat.shape

    if percentiles is None:
        percentiles = [16., 84.]
    print("percentiles = ", percentiles)
    YM_low = np.percentile(integratedY_mat, percentiles[0], axis=0)
    YM_high = np.percentile(integratedY_mat, percentiles[1], axis=0)

    # pdb.set_trace()

    ax.fill_between(M_array, YM_low, YM_high, color='blue', alpha=0.4, label=r'${\rm Forecast}$')
    if do_samples:
        for j in range(nsamples):
            ax.plot(M_array, integratedY_mat[j, :], color='blue', linewidth=1., alpha=0.6)

    if np.max(M_array) > 3 * 10 ** 14:
        ind_match = np.where(M_array > 3 * 10 ** 14)[0][0]
    else:
        ind_match = len(M_array) / 2
    prefactor = (YM_fid[ind_match] / (M_array[ind_match] ** (5. / 3.)))

    ax.plot(M_array, prefactor * M_array ** (5. / 3.), color='yellow', label=r'$M^{5/3}$', lw='2')
    ax.plot(M_array, YM_fid, color='red', label=r'${\rm Fiducial}$', lw='2')

    if do_split_params_massbins:
        mass_bin_split = split_mass_bins_min
        for j in range(len(mass_bin_split)):
            ax.axvline(x=mass_bin_split[j], ymin=0., ymax=1., ls='--', color='k')

        if xlim is None:
            ax.set_xlim(np.min(M_array) * 0.9, np.max(M_array) * 1.1)
        ax.set_ylim(10 ** -6, 10 ** -1)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$M_{\rm 500c} \ (M_{\odot}/h)$', size=20)
    ax.set_ylabel(r'$ Y_{500} \ (\rm{arcmin}^2)$', size=20)
    # legend = ax.legend(loc='upper left', fontsize=20, framealpha=1.0)
    # frame = legend.get_frame()
    # frame.set_facecolor('white')
    # frame.set_linewidth(0)

    ax.legend(loc='lower right', fontsize=15, frameon=False)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    # ax.set_ylim((3.0e-4, 1.0e-3))

    fig.tight_layout()

    if save_plots:
        fig.savefig(plot_dir + 'YM_relation_' + save_suffix + '.png')

    return fig


def plot_YM_relation_relative(M_array, integratedY_mat, YM_fid, do_split_params_massbins, split_mass_bins_min=None,
                              plot_dir='./', save_suffix='', save_plots=False, xlim=None, do_samples=True,
                              percentiles=None, labels=None):
    nsamples, numM = integratedY_mat.shape
    integratedY_relative_mat = np.zeros((nsamples, numM))
    for ii in range(nsamples):
        integratedY_relative_mat[ii, :] = integratedY_mat[ii, :] / YM_fid

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if (percentiles == None):
        percentiles = [16., 84.]

    print("percentiles = ", percentiles)
    YM_low = np.percentile(integratedY_relative_mat, percentiles[0], axis=0)
    YM_high = np.percentile(integratedY_relative_mat, percentiles[1], axis=0)
    # pdb.set_trace()
    ax.fill_between(M_array, YM_low, YM_high, color='blue', alpha=0.4, label=labels[0])
    if (do_samples):
        for j in range(nsamples):
            ax.plot(M_array, integratedY_relative_mat[j, :], color='blue', linewidth=1., alpha=0.6)

    # ax.plot(M_array, (prefactor * M_array ** (5. / 3.))/YM_fid, color='green', label=r'$M^{5/3}$', lw='2')
    ax.plot(M_array, YM_fid / YM_fid, label=labels[1], color='black', ls='dashed')

    if do_split_params_massbins:
        mass_bin_split = split_mass_bins_min
        for j in range(len(mass_bin_split)):
            ax.axvline(x=mass_bin_split[j], ymin=0., ymax=1., ls='--', color='k')

        if (xlim == None):
            ax.set_xlim(np.min(M_array) * 0.9, np.max(M_array) * 1.1)

    if (xlim != None):
        ax.set_xlim(xlim)
    # ax.set_ylim((np.min(YM_low) - 0.1, np.max(YM_high) + 0.1))
    ax.set_ylim((0.8, 1.2))
    ax.set_xscale('log')
    ax.set_xlabel(r'$M_{\rm 500c} \ (M_{\odot}/h)$', size=20)
    ax.set_ylabel(r'$ Y_{500} / Y_{500,{\rm fid}} $', size=20)
    # legend = ax.legend(loc='upper left', fontsize=20, framealpha=1.0)
    # frame = legend.get_frame()
    # frame.set_facecolor('white')
    # frame.set_linewidth(0)

    ax.legend(fontsize=15, frameon=False)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    fig.tight_layout()

    if save_plots:
        fig.savefig(plot_dir + 'YM_relation_' + save_suffix + '.png')

    return fig


def plot_Pr_dep(x_array, Pr_mat, pressure_params_array, fisher_params_vary, fisher_params_vary_orig,
                fisher_params_label, plot_dir, save_suffix):
    nparam = len(fisher_params_vary)
    figx = nparam * 5.
    fig, ax = plt.subplots(1, nparam, figsize=(figx, 4), sharey=True)

    j_to_plot = 1

    for j in range(nparam):
        param_values = []
        param_vary = fisher_params_vary[j]

        if param_vary not in fisher_params_vary_orig:
            param_vary_split = list(param_vary)
            addition_index = param_vary_split.index('-')
            param_vary_orig = ''.join(param_vary_split[:addition_index])
        else:
            param_vary_orig = param_vary

        if param_vary_orig in pressure_params_array[0].keys():

            for i in range(len(pressure_params_array)):

                if param_vary not in fisher_params_vary_orig:
                    if str.isdigit(list(param_vary)[-1]):
                        mass_bin_number = int(list(param_vary)[-1])
                    if str.isdigit(list(param_vary)[-2]):
                        mass_bin_number += 10 * int(list(param_vary)[-2])

                    param_vary_split = list(param_vary)
                    addition_index = param_vary_split.index('-')
                    param_vary_orig = ''.join(param_vary_split[:addition_index])

                    param_value = pressure_params_array[i][param_vary_orig][mass_bin_number]
                else:
                    param_value = pressure_params_array[i][param_vary]

                param_values.append(param_value)
            param_values = np.array(param_values)

            ax[j].plot(param_values, Pr_mat[:, j_to_plot], ls='', marker='*')

            if j == 0:
                ax[j].set_ylabel(r'$P_e(x=$' + str(np.around(x_array[j_to_plot], 2)) + ')', size=17)

            ax[j].set_xlabel(fisher_params_label[j], size=17)

            ax[j].tick_params(axis='both', which='major', labelsize=17)
            ax[j].tick_params(axis='both', which='minor', labelsize=17)
        else:
            fig.delaxes(ax[j])

    fig.tight_layout()

    fig.savefig(plot_dir + 'Pr_paramdep_' + save_suffix + '.png')

    # pdb.set_trace()


def plot_dCl_dparam(l_array, dCl_dparam, cov_fid, param_name, param_label, save_suffix, plot_dir,stat = None):
    sig_fid = np.sqrt(np.diag(cov_fid))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(l_array, l_array * (l_array + 1.) * dCl_dparam / (2 * np.pi), color='blue', marker='', linestyle='-',
            label=r'$dC_{\ell}/d$' + param_label)
    ax.plot(l_array, -1 * l_array * (l_array + 1.) * dCl_dparam / (2 * np.pi), color='blue', marker='', linestyle='--')
    ax.plot(l_array, l_array * (l_array + 1.) * sig_fid / (2 * np.pi), color='red', marker='', linestyle=':',
            label=r'$\sigma (C_{\ell})$')

    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_ylim(1e-2, 10)
    if stat in ['yg','gy']:
        ax.set_ylabel(r'$\ell \ (\ell + 1) \ C^{hy}_\ell/ 2 \pi$', size=22)
        savename = 'dCl_dparam_hy_' + param_name + '_' + save_suffix + '.png'

    if stat in ['yy']:
        ax.set_ylabel(r'$\ell \ (\ell + 1) \ C^{yy}_\ell/ 2 \pi$', size=22)
        savename = 'dCl_dparam_yy_' + param_name + '_' + save_suffix + '.png'

    ax.set_xlabel(r'$\ell$', size=22)
    ax.legend(fontsize=20, frameon=False, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    # pdb.set_trace()
    plt.savefig(plot_dir + savename)
    plt.close()


def plot_dlnCl_dlnparam(l_array, dCl_dparam, Cl_fid, param_fid, param_name, param_label, save_suffix, plot_dir,stat = None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(l_array, (param_fid/Cl_fid) * dCl_dparam , color='blue', marker='', linestyle='-')

    # ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_ylim(1e-2, 10)
    if stat in ['yg','gy']:
        ax.set_ylabel(r'$d\log \ C^{hy}_\ell/ d \log$' + param_label, size=22)
        savename = 'dlnCl_dlnparam_hy_' + param_name + '_' + save_suffix + '.png'

    if stat in ['yy']:
        ax.set_ylabel(r'$d\log \ C^{yy}_\ell/ d \log$' + param_label, size=22)
        savename = 'dlnCl_dlnparam_yy_' + param_name + '_' + save_suffix + '.png'

    if stat in ['gg']:
        ax.set_ylabel(r'$d\log \ C^{gg}_\ell/ d \log$' + param_label, size=22)
        savename = 'dlnCl_dlnparam_gg_' + param_name + '_' + save_suffix + '.png'

    ax.set_xlabel(r'$\ell$', size=22)
    ax.legend(fontsize=20, frameon=False, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    # pdb.set_trace()
    plt.savefig(plot_dir + savename)
    plt.close()


def plot_chi2_samples(chi2_samples, chi2_params_array):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    # rows
    for parami in range(2):
        # cols
        for paramj in range(2):
            # ellipses on lower triangle

            ax_handle = ax[parami, paramj]

            if parami > paramj:
                ax_handle.scatter(chi2_samples, chi2_params_array, s=1.4)

                ax_handle.plot(np.linspace(0.01, np.max(chi2_samples), 100),
                               np.linspace(0.01, np.max(chi2_samples), 100))

                ax_handle.set_xlabel(r'$\chi^2_{data}$', fontsize=20)
                ax_handle.set_ylabel(r'$\chi^2_{params}$', fontsize=20)

                # ax_handle.set_xlim(0,10)
                # ax_handle.set_ylim(0, 10)

                ax_handle.set_xscale('log')
                ax_handle.set_yscale('log')

            # Get rid of upper triangle
            if paramj > parami:
                fig.delaxes(ax_handle)
            # 1d gaussian on diagonal
            if parami == paramj:
                if parami == 0:
                    ax_handle.hist(chi2_samples, bins=100, histtype='step')
                    ax_handle.set_xlabel(r'$\chi^2_{data}$', fontsize=20)
                else:
                    ax_handle.hist(chi2_params_array, bins=100, histtype='step')
                    ax_handle.set_xlabel(r'$\chi^2_{params}$', fontsize=20)

            ax_handle.tick_params(axis='both', which='major', labelsize=15)
            ax_handle.tick_params(axis='both', which='minor', labelsize=15)

    return fig
