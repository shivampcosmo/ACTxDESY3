import sys, os

sys.path.insert(0, '../../helper/')
import LSS_funcs as hmf
import numpy as np
from scipy import interpolate
import scipy as sp
import scipy.interpolate as interpolate
from astropy.cosmology import Planck15
from camb import model
from scipy.integrate import simps as _simps
from scipy.interpolate import InterpolatedUnivariateSpline as _spline


def QR_inverse(matrix):
    _Q, _R = np.linalg.qr(matrix)
    return np.dot(_Q, np.linalg.inv(_R.T))


def get_wprp_gt(rp, karr, Pkarr):
    val1 = sp.special.jv(2, karr * rp)
    valf = (sp.integrate.simps(karr * Pkarr * val1, karr)) / (2 * np.pi)
    return valf


out_dict = np.load(
    '/global/cfs/cdirs/des/shivamp/cosmosis/ACTxDESY3/src/results/results_block_ell_kk_ky_desy1.npz')
theta_min = 2.5
theta_max = 250.
ntheta = 21
# import pdb; pdb.set_trace()

l_array = out_dict['theory_ell']
theta_array_all = np.logspace(np.log10(theta_min), np.log10(theta_max), ntheta)
theta_array = (theta_array_all[1:] + theta_array_all[:-1]) / 2.
theta_array_rad = theta_array * (np.pi / 180.) * (1. / 60.)
# l_array_full = np.logspace(0, 4.1, 50000)
l_array_full = np.linspace(0.1, 20000, 500000)

wtheta_yg_dict = {'theta_rad': theta_array_rad, 'theta_arcmin': theta_array}
wtheta_gg_dict = {'theta_rad': theta_array_rad, 'theta_arcmin': theta_array}

xi_gty_dict = {'theta_rad': theta_array_rad, 'theta_arcmin': theta_array}
wtheta_ky_dict = {'theta_rad': theta_array_rad, 'theta_arcmin': theta_array}
wtheta_kk_dict = {'theta_rad': theta_array_rad, 'theta_arcmin': theta_array}

wtheta_yy_dict = {'theta_rad': theta_array_rad, 'theta_arcmin': theta_array}

for jb in range(4):
    binv = jb + 1
    print('proceessing bin-' + str(binv))

    # Cl_yg_1h_array = out_dict['theory_clgy1h_bin_' + str(binv) + '_' + str(binv)]
    # Cl_yg_2h_array = out_dict['theory_clgy2h_bin_' + str(binv) + '_' + str(binv)]
    #
    # Cl_yg_1h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yg_1h_array), fill_value='extrapolate', bounds_error=False)
    # Cl_yg_1h_full = np.exp(Cl_yg_1h_interp(np.log(l_array_full)))
    # Cl_yg_2h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yg_2h_array), fill_value='extrapolate', bounds_error=False)
    # Cl_yg_2h_full = np.exp(Cl_yg_2h_interp(np.log(l_array_full)))
    #
    # wtheta_yg_1h = np.zeros(len(theta_array_rad))
    # wtheta_yg_2h = np.zeros(len(theta_array_rad))
    # wtheta_yg = np.zeros(len(theta_array_rad))
    #
    # for j in range(len(theta_array_rad)):
    #     wtheta_yg_1h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yg_1h_full)
    #     wtheta_yg_2h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yg_2h_full)
    #     wtheta_yg[j] = wtheta_yg_1h[j] + wtheta_yg_2h[j]
    #
    # Cl_gg_1h_array = out_dict['theory_clgg1h_bin_' + str(binv) + '_' + str(binv)]
    # Cl_gg_2h_array = out_dict['theory_clgg2h_bin_' + str(binv) + '_' + str(binv)]
    #
    # Cl_gg_1h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_gg_1h_array), fill_value='extrapolate', bounds_error=False)
    # Cl_gg_1h_full = np.exp(Cl_gg_1h_interp(np.log(l_array_full)))
    # Cl_gg_2h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_gg_2h_array), fill_value='extrapolate', bounds_error=False)
    # Cl_gg_2h_full = np.exp(Cl_gg_2h_interp(np.log(l_array_full)))
    #
    # wtheta_gg_1h = np.zeros(len(theta_array_rad))
    # wtheta_gg_2h = np.zeros(len(theta_array_rad))
    # wtheta_gg = np.zeros(len(theta_array_rad))
    #
    # for j in range(len(theta_array_rad)):
    #     wtheta_gg_1h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_gg_1h_full)
    #     wtheta_gg_2h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_gg_2h_full)
    #     wtheta_gg[j] = wtheta_gg_1h[j] + wtheta_gg_2h[j]
    #
    # wtheta_yg_dict[ str(binv) + '-1h'] = wtheta_yg_1h
    # wtheta_yg_dict[ str(binv) + '-2h'] = wtheta_yg_2h
    # wtheta_yg_dict[ str(binv) + '-tot'] = wtheta_yg
    #
    # wtheta_gg_dict[ str(binv) + '-1h'] = wtheta_gg_1h
    # wtheta_gg_dict[ str(binv) + '-2h'] = wtheta_gg_2h
    # wtheta_gg_dict[ str(binv) + '-tot'] = wtheta_gg

    Cl_ky_1h_array = out_dict['theory_clky1h_bin_' + str(binv) + '_' + str(binv)]
    Cl_ky_2h_array = out_dict['theory_clky2h_bin_' + str(binv) + '_' + str(binv)]

    cov_ky_ky = out_dict['cov_total_ky_ky_bin_' + str(binv) + '_' + str(binv)]
    inv_cov_bin = QR_inverse(cov_ky_ky)
    Cl_ky_bin = Cl_ky_1h_array + Cl_ky_2h_array
    snr_bin = np.sqrt(np.dot(np.array([Cl_ky_bin]), np.dot(inv_cov_bin, np.array([Cl_ky_bin]).T)))
    print('SNR ky bin:' + str(binv) + '=' + str(np.round(snr_bin[0][0], 2)) + ' sigma')
    # import pdb; pdb.set_trace()

    Cl_ky_1h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_ky_1h_array), fill_value='extrapolate',
                                           bounds_error=False)
    Cl_ky_1h_full = np.exp(Cl_ky_1h_interp(np.log(l_array_full)))
    Cl_ky_2h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_ky_2h_array), fill_value='extrapolate',
                                           bounds_error=False)
    Cl_ky_2h_full = np.exp(Cl_ky_2h_interp(np.log(l_array_full)))

    wtheta_ky_1h = np.zeros(len(theta_array_rad))
    wtheta_ky_2h = np.zeros(len(theta_array_rad))
    wtheta_ky = np.zeros(len(theta_array_rad))

    xi_gty_1h = np.zeros(len(theta_array_rad))
    xi_gty_2h = np.zeros(len(theta_array_rad))
    xi_gty = np.zeros(len(theta_array_rad))

    for j in range(len(theta_array_rad)):
        wtheta_ky_1h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_ky_1h_full)
        wtheta_ky_2h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_ky_2h_full)
        wtheta_ky[j] = wtheta_ky_1h[j] + wtheta_ky_2h[j]

        xi_gty_1h[j] = get_wprp_gt(theta_array_rad[j], l_array_full, Cl_ky_1h_full)
        xi_gty_2h[j] = get_wprp_gt(theta_array_rad[j], l_array_full, Cl_ky_2h_full)
        xi_gty[j] = xi_gty_1h[j] + xi_gty_2h[j]

    Cl_kk_1h_array = out_dict['theory_clkk1h_bin_' + str(binv) + '_' + str(binv)]
    Cl_kk_2h_array = out_dict['theory_clkk2h_bin_' + str(binv) + '_' + str(binv)]

    cov_kk_kk = out_dict['cov_total_kk_kk_bin_' + str(binv) + '_' + str(binv)]
    inv_cov_bin = QR_inverse(cov_kk_kk)
    Cl_kk_bin = Cl_kk_1h_array + Cl_kk_2h_array
    snr_bin = np.sqrt(np.dot(np.array([Cl_kk_bin]), np.dot(inv_cov_bin, np.array([Cl_kk_bin]).T)))
    print('SNR kk bin:' + str(binv) + '=' + str(np.round(snr_bin[0][0], 2)) + ' sigma')
    # import pdb; pdb.set_trace()

    Cl_kk_1h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_kk_1h_array), fill_value='extrapolate',
                                           bounds_error=False)
    Cl_kk_1h_full = np.exp(Cl_kk_1h_interp(np.log(l_array_full)))
    Cl_kk_2h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_kk_2h_array), fill_value='extrapolate',
                                           bounds_error=False)
    Cl_kk_2h_full = np.exp(Cl_kk_2h_interp(np.log(l_array_full)))

    wtheta_kk_1h = np.zeros(len(theta_array_rad))
    wtheta_kk_2h = np.zeros(len(theta_array_rad))
    wtheta_kk = np.zeros(len(theta_array_rad))

    for j in range(len(theta_array_rad)):
        wtheta_kk_1h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_kk_1h_full)
        wtheta_kk_2h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_kk_2h_full)
        wtheta_kk[j] = wtheta_kk_1h[j] + wtheta_kk_2h[j]


    if jb ==   0:
        Cl_yy_1h_array = out_dict['theory_clyy1h']
        Cl_yy_2h_array = out_dict['theory_clyy2h']

        cov_yy_yy = out_dict['cov_total_yy_yy']
        inv_cov_bin = QR_inverse(cov_yy_yy)
        Cl_yy_bin = Cl_yy_1h_array + Cl_yy_2h_array
        snr_bin = np.sqrt(np.dot(np.array([Cl_yy_bin]), np.dot(inv_cov_bin, np.array([Cl_yy_bin]).T)))
        print('SNR yy bin:' + str(binv) + '=' + str(np.round(snr_bin[0][0], 2)) + ' sigma')
        # import pdb; pdb.set_trace()

        Cl_yy_1h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yy_1h_array), fill_value='extrapolate',
                                               bounds_error=False)
        Cl_yy_1h_full = np.exp(Cl_yy_1h_interp(np.log(l_array_full)))
        Cl_yy_2h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yy_2h_array), fill_value='extrapolate',
                                               bounds_error=False)
        Cl_yy_2h_full = np.exp(Cl_yy_2h_interp(np.log(l_array_full)))

        wtheta_yy_1h = np.zeros(len(theta_array_rad))
        wtheta_yy_2h = np.zeros(len(theta_array_rad))
        wtheta_yy = np.zeros(len(theta_array_rad))

        for j in range(len(theta_array_rad)):
            wtheta_yy_1h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yy_1h_full)
            wtheta_yy_2h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yy_2h_full)
            wtheta_yy[j] = wtheta_yy_1h[j] + wtheta_yy_2h[j]

        wtheta_yy_dict['1h'] = wtheta_yy_1h
        wtheta_yy_dict['2h'] = wtheta_yy_2h
        wtheta_yy_dict['tot'] = wtheta_yy

    wtheta_ky_dict[str(binv) + '-1h'] = wtheta_ky_1h
    wtheta_ky_dict[str(binv) + '-2h'] = wtheta_ky_2h
    wtheta_ky_dict[str(binv) + '-tot'] = wtheta_ky

    xi_gty_dict[str(binv) + '-1h'] = xi_gty_1h
    xi_gty_dict[str(binv) + '-2h'] = xi_gty_2h
    xi_gty_dict[str(binv) + '-tot'] = xi_gty

    wtheta_kk_dict[str(binv) + '-1h'] = wtheta_kk_1h
    wtheta_kk_dict[str(binv) + '-2h'] = wtheta_kk_2h
    wtheta_kk_dict[str(binv) + '-tot'] = wtheta_kk

    # import pdb; pdb.set_trace()

    # print
# import pdb; pdb.set_trace()

# np.savez('/global/project/projectdirs/des/shivamp/cosmosis/ACTxDESY3/src/results/results_wtheta_gg_maglim.npz', **wtheta_gg_dict)
# np.savez('/global/project/projectdirs/des/shivamp/cosmosis/ACTxDESY3/src/results/results_wtheta_gy_maglim.npz', **wtheta_yg_dict)

np.savez('/global/cfs/cdirs/des/shivamp/cosmosis/ACTxDESY3/src/results/results_wtheta_kk_desy1.npz',
         **wtheta_kk_dict)
np.savez('/global/cfs/cdirs/des/shivamp/cosmosis/ACTxDESY3/src/results/results_wtheta_ky_desy1.npz',
         **wtheta_ky_dict)
np.savez('/global/cfs/cdirs/des/shivamp/cosmosis/ACTxDESY3/src/results/results_xi_gty_desy1.npz',
         **xi_gty_dict)



