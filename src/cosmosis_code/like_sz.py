import sys, os
from cosmosis.datablock import names, option_section, BlockError
sys.path.insert(0, '../../helper/')
sys.path.insert(0, '../')
import numpy as np
import copy
import pdb
import ast
import scipy as sp
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
import multiprocessing
import dill
import pickle as pk
from configobj import ConfigObj
from configparser import ConfigParser
import traceback as tb


def get_corr(cov):
    corr = np.zeros(cov.shape)
    for ii in xrange(0, cov.shape[0]):
        for jj in xrange(0, cov.shape[1]):
            corr[ii, jj] = cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj])
    return corr


def QR_inverse(matrix):
    _Q,_R = np.linalg.qr(matrix)
    return np.dot(_Q,np.linalg.inv(_R.T))

def get_theory_terms(block, xcoord_data, stat_type, bins_array, sec_savename = "save_theory"):
    try:
        stats_array = stat_type.split('_')
    except:
        stats_array = stat_type
    corrf_theory_rdata = []


    # if stat_type == 'gg':
    for stat in stats_array:
        nbins = len(bins_array)
        kbv = 0
        for j in range(nbins):
            bin_j = bins_array[j]
            try:
                if stat == 'gty':
                    xcoord_array = block.get_double_array_1d(sec_savename, "xcoord_" + stat + '_bin_' + str(bin_j) + '_' + str(0))
                    corrf_stat =  block.get_double_array_1d(sec_savename,'theory_corrf_' + stat + '_bin_' + str(bin_j) + '_' + str(0))
                else:
                    xcoord_array = block.get_double_array_1d(sec_savename, "xcoord_" + stat + '_bin_' + str(bin_j) + '_' + str(bin_j))
                    corrf_stat =  block.get_double_array_1d(sec_savename,'theory_corrf_' + stat + '_bin_' + str(bin_j) + '_' + str(bin_j))
            except:
                import ipdb; ipdb.set_trace() # BREAKPOINT

            corrf_stat_temp = intspline(xcoord_array, corrf_stat)
            corrf_stat_f = corrf_stat_temp(xcoord_data[kbv])
            if len(corrf_theory_rdata) == 0:
                corrf_theory_rdata = corrf_stat_f
            else:
                corrf_theory_rdata = np.hstack((corrf_theory_rdata, corrf_stat_f))
            kbv += 1
            # import pdb; pdb.set_trace()

    # if stat_type == 'gy':
    #     nbins = len(bins_array)
    #     for j in range(nbins):
    #         bin_j = bins_array[j]
    #         corrf_gy = block.get_double_array_1d(sec_savename,'theory_corrfgy_bin_'+ str(bin_j) + '_' + str(bin_j))
    #         corrf_gy_temp = intspline(xcoord_array, corrf_gy)
    #         corrf_gy_f = corrf_gy_temp(xcoord_data[j])
    #         if len(corrf_theory_rdata) == 0:
    #             corrf_theory_rdata = corrf_gy_f
    #         else:
    #             corrf_theory_rdata = np.hstack((corrf_theory_rdata, corrf_gy_f))
    #
    # elif stat_type == 'gg_gy':
    #     nbins = len(bins_array)
    #     for j in range(nbins):
    #         bin_j = bins_array[j]
    #         corrf_gg = block.get_double_array_1d(sec_savename,'theory_corrfgg_bin_' + str(bin_j) + '_' + str(bin_j))
    #         corrf_gg_temp = intspline(xcoord_array, corrf_gg)
    #         corrf_gg_f = corrf_gg_temp(xcoord_data[j])
    #         if len(corrf_theory_rdata) == 0:
    #             corrf_theory_rdata = corrf_gg_f
    #         else:
    #             corrf_theory_rdata = np.hstack((corrf_theory_rdata, corrf_gg_f ))
    #
    #     for j in range(nbins):
    #         bin_j = bins_array[j]
    #         corrf_gy = block.get_double_array_1d(sec_savename,'theory_corrfgy_bin_'+ str(bin_j) + '_' + str(bin_j))
    #         corrf_gy_temp = intspline(xcoord_array, corrf_gy)
    #         corrf_gy_f = corrf_gy_temp(xcoord_data[j + nbins])
    #         corrf_theory_rdata = np.hstack((corrf_theory_rdata, corrf_gy_f))

    return corrf_theory_rdata


def lnprob_func(block, xcoord_data, corrf_data_gtcut, incov_obs_comp, stat_type, bins_array,sec_save_name):
    corrf_theory_rdata = get_theory_terms(block, xcoord_data, stat_type, bins_array,sec_savename=sec_save_name)
    # import pdb; pdb.set_trace()
    valf = -0.5 * np.dot(np.dot(np.transpose((corrf_data_gtcut - corrf_theory_rdata)), incov_obs_comp),
                         (corrf_data_gtcut - corrf_theory_rdata))
    return valf, corrf_theory_rdata


def setuplnprob_func(scale_cut_min, scale_cut_max, xcoord_data_array, corrf_data_full, cov_obs, stat_type, bins_array, cov_diag=False,
                     no_cov_zbins_only_auto_cross=False, no_cov_zbins_all=False, no_cov_auto_cross=False):
    xcoord_data_comp_ll = []

    selection = []
    countk = 0

    if stat_type in ['gg_gy','kk_gty']:

        xcoord_data_autocorr_all = np.array([])
        xcoord_data_xcorr_all = np.array([])
        for j in range(len(bins_array)):
            if len(xcoord_data_autocorr_all) == 0:
                xcoord_data_autocorr_all = xcoord_data_array[j]
                xcoord_data_xcorr_all = xcoord_data_array[j + len(bins_array)]
            else:
                xcoord_data_autocorr_all = np.hstack((xcoord_data_autocorr_all, xcoord_data_array[j]))
                xcoord_data_xcorr_all = np.hstack((xcoord_data_xcorr_all, xcoord_data_array[j + len(bins_array)]))

        xcoord_data_all = np.hstack((xcoord_data_autocorr_all, xcoord_data_xcorr_all))

        for j in range(len(bins_array)):
            xcoord_data_j = xcoord_data_array[j]
            selection_j = np.where((xcoord_data_j >= scale_cut_min[j]) & (xcoord_data_j <= scale_cut_max[j]))[0]
            if len(selection) == 0:
                selection = selection_j
            else:
                selection = np.hstack((selection, countk + selection_j))
            xcoord_data_comp_j = xcoord_data_j[selection_j]
            xcoord_data_comp_ll.append(xcoord_data_comp_j)
            countk += len(xcoord_data_j)

        for j in range(len(bins_array)):
            xcoord_data_j = xcoord_data_array[j + len(bins_array)]
            selection_j = \
                np.where((xcoord_data_j >= scale_cut_min[j + len(bins_array)]) & (xcoord_data_j <= scale_cut_max[j + len(bins_array)]))[0]
            selection = np.hstack((selection, countk + selection_j))
            xcoord_data_comp_j = xcoord_data_j[selection_j]
            xcoord_data_comp_ll.append(xcoord_data_comp_j)
            countk += len(xcoord_data_j)

    if stat_type in ['gg','kk']:

        xcoord_data_autocorr_all = np.array([])
        for j in range(len(bins_array)):
            if len(xcoord_data_autocorr_all) == 0:
                xcoord_data_autocorr_all = xcoord_data_array[j]
            else:
                xcoord_data_autocorr_all = np.hstack((xcoord_data_autocorr_all, xcoord_data_array[j]))

        xcoord_data_all = xcoord_data_autocorr_all

        for j in range(len(bins_array)):
            xcoord_data_j = xcoord_data_array[j]
            selection_j = np.where((xcoord_data_j >= scale_cut_min[j]) & (xcoord_data_j <= scale_cut_max[j]))[0]
            if len(selection) == 0:
                selection = selection_j
            else:
                selection = np.hstack((selection, countk + selection_j))

            xcoord_data_comp_j = xcoord_data_j[selection_j]
            xcoord_data_comp_ll.append(xcoord_data_comp_j)
            countk += len(xcoord_data_j)

    if stat_type in ['gy','gty']:

        xcoord_data_xcorr_all = np.array([])

        for j in range(len(bins_array)):
            if len(xcoord_data_xcorr_all) == 0:
                xcoord_data_xcorr_all = xcoord_data_array[j + len(bins_array)]
            else:
                xcoord_data_xcorr_all = np.hstack((xcoord_data_xcorr_all, xcoord_data_array[j + len(bins_array)]))

        xcoord_data_all = xcoord_data_xcorr_all

        for j in range(len(bins_array)):
            xcoord_data_j = xcoord_data_array[j + len(bins_array)]
            selection_j = \
                np.where((xcoord_data_j >= scale_cut_min[j + len(bins_array)]) & (xcoord_data_j <= scale_cut_max[j + len(bins_array)]))[0]
            if len(selection) == 0:
                selection = selection_j
            else:
                selection = np.hstack((selection, countk + selection_j))

            xcoord_data_comp_j = xcoord_data_j[selection_j]
            xcoord_data_comp_ll.append(xcoord_data_comp_j)
            countk += len(xcoord_data_j)

    selection = np.array(selection)
    cov_obs_comp = (cov_obs[:, selection])[selection, :]

    if no_cov_zbins_only_auto_cross or no_cov_zbins_all:
        bins_n_array = np.arange(len(bins_array))
        cov_obs_comp_h = np.copy(cov_obs_comp)

        if len(bins_array) > 1:
            print ('zeroing the covariance between z bins')

            if stat_type in ['gg_gy','kk_gty']:
                z1_0 = []
                for ji in range(len(bins_array)):
                    if len(z1_0) == 0:
                        z1_0 = bins_n_array[ji] * np.ones(len(xcoord_data_comp_ll[ji]))
                    else:
                        z1_0 = np.hstack((z1_0, bins_n_array[ji] * np.ones(len(xcoord_data_comp_ll[ji]))))

                z1_1 = []
                for ji in range(len(bins_array)):
                    if len(z1_1) == 0:
                        z1_1 = bins_n_array[ji] * np.ones(len(xcoord_data_comp_ll[len(bins_array) + ji]))
                    else:
                        z1_1 = np.hstack((z1_1, bins_n_array[ji] * np.ones(len(xcoord_data_comp_ll[len(bins_array) + ji]))))

                z1_mat_0 = np.tile(z1_0, (len(z1_0), 1)).transpose()
                z1_mat_1 = np.tile(z1_1, (len(z1_1), 1)).transpose()

                z1_mat_01 = np.tile(z1_0, (len(z1_1), 1)).transpose()
                z1_mat_10 = np.tile(z1_1, (len(z1_0), 1)).transpose()

                if no_cov_zbins_only_auto_cross:
                    z1_mat_0c = -1 * np.ones(z1_mat_0.shape)
                    z1_mat_1c = -1 * np.ones(z1_mat_1.shape)
                if no_cov_zbins_all:
                    z1_mat_0c = z1_mat_0
                    z1_mat_1c = z1_mat_1

                z1_mat2 = np.concatenate((z1_mat_0c, z1_mat_01), axis=1)
                z1_mat22 = np.concatenate((z1_mat_10, z1_mat_1c), axis=1)
                z1_matf = np.concatenate((z1_mat2, z1_mat22), axis=0)
                z2_matf = np.transpose(z1_matf)
                offdiag = np.where(z1_matf != z2_matf)
                cov_obs_comp_h[offdiag] = 0.0

            if stat_type == 'gg' or stat_type == 'gy' or stat_type == 'kk' or stat_type == 'gty':
                z1 = np.repeat(np.arange(len(bins_array)), len(xcoord_data_comp_ll[0]))
                z1_mat = np.tile(z1, (len(bins_array) * len(xcoord_data_comp_ll[0]), 1)).transpose()
                z2_mat = np.transpose(z1_mat)
                offdiag = np.where(z1_mat != z2_mat)
                cov_obs_comp_h[offdiag] = 0.0

        cov_obs_comp = cov_obs_comp_h

    if no_cov_auto_cross:
        bins_n_array = np.arange(len(bins_array)) + 1
        cov_obs_comp_hf = np.copy(cov_obs_comp)
        print ('zeroing the covariance between auto and cross')
        if stat_type in ['gg_gy','kk_gty']:
            z1_0 = []
            for ji in range(len(bins_array)):
                if len(z1_0) == 0:
                    z1_0 = bins_n_array[ji] * np.ones(len(xcoord_data_comp_ll[ji]))
                else:
                    z1_0 = np.hstack((z1_0, bins_n_array[ji] * np.ones(len(xcoord_data_comp_ll[ji]))))

            z1_1 = []
            for ji in range(len(bins_array)):
                if len(z1_1) == 0:
                    z1_1 = -1. * bins_n_array[ji] * np.ones(len(xcoord_data_comp_ll[len(bins_array) + ji]))
                else:
                    z1_1 = np.hstack((z1_1, -1. * bins_n_array[ji] * np.ones(len(xcoord_data_comp_ll[len(bins_array) + ji]))))

            z1_mat_0 = np.tile(z1_0, (len(z1_0), 1)).transpose()
            z1_mat_1 = np.tile(z1_1, (len(z1_1), 1)).transpose()

            z1_mat_01 = np.tile(z1_0, (len(z1_1), 1)).transpose()
            z1_mat_10 = np.tile(z1_1, (len(z1_0), 1)).transpose()

            z1_mat2 = np.concatenate((z1_mat_0, z1_mat_01), axis=1)
            z1_mat22 = np.concatenate((z1_mat_10, z1_mat_1), axis=1)
            z1_matf = np.concatenate((z1_mat2, z1_mat22), axis=0)
            z2_matf = np.transpose(z1_matf)
            offdiag = np.where(z1_matf != z2_matf)
            cov_obs_comp_hf[offdiag] = 0.0


        cov_obs_comp = cov_obs_comp_hf

    if cov_diag:
        cov_obs_comp = np.diag(np.diag(cov_obs_comp))

    incov_obs_comp = QR_inverse(cov_obs_comp)
    xcoord_data_comp = xcoord_data_all[selection]
    corrf_data_gtcut = corrf_data_full[selection]
    print('total number of data points=' + str(len(corrf_data_gtcut)))
    # import pdb; pdb.set_trace()
    return corrf_data_gtcut, xcoord_data_comp, xcoord_data_comp_ll, incov_obs_comp, cov_obs_comp

def import_data(xcoord_obs, data_obs, cov_obs, bins_to_rem, bins_to_fit, bins_all, stat_type):
    if len(bins_to_rem) > 0:
        cov_obs_rm = np.ones(cov_obs.shape)
        cov_obs_copy = np.copy(cov_obs)

        z1_0 = []
        for ji in range(len(bins_all)):
            if len(z1_0) == 0:
                z1_0 = bins_all[ji] * np.ones(len(xcoord_obs[ji]))
            else:
                z1_0 = np.hstack((z1_0, bins_all[ji] * np.ones(len(xcoord_obs[ji]))))

        z1_1 = []
        for ji in range(len(bins_all)):
            if len(z1_1) == 0:
                z1_1 = bins_all[ji] * np.ones(len(xcoord_obs[len(bins_all) + ji]))
            else:
                z1_1 = np.hstack((z1_1, bins_all[ji] * np.ones(len(xcoord_obs[len(bins_all) + ji]))))

        z1_mat_0 = np.tile(z1_0, (len(z1_0), 1)).transpose()
        z1_mat_1 = np.tile(z1_1, (len(z1_1), 1)).transpose()

        z1_mat_01 = np.tile(z1_0, (len(z1_1), 1)).transpose()
        z1_mat_10 = np.tile(z1_1, (len(z1_0), 1)).transpose()

        z1_mat2 = np.concatenate((z1_mat_0, z1_mat_01), axis=1)
        z1_mat22 = np.concatenate((z1_mat_10, z1_mat_1), axis=1)
        z1_matf = np.concatenate((z1_mat2, z1_mat22), axis=0)
        z2_matf = np.transpose(z1_matf)
        ind_to_select = np.ones(z1_matf.shape)

        z1f = np.concatenate((z1_0, z1_1))

        ind_to_select_robs = []

        for bins in bins_to_rem:
            ax1_ind = np.where(z1_matf == bins)
            ax2_ind = np.where(z2_matf == bins)
            ax1_ind_robs = np.where(z1f == bins)[0]
            ind_to_select_robs.append(ax1_ind_robs)
            ind_to_select[ax1_ind] = 0
            ind_to_select[ax2_ind] = 0

        del_indf = (np.array(ind_to_select_robs)).flatten()

        ind_rm_f = np.where(ind_to_select == 0)
        cov_obs_rm[ind_rm_f] = 0
        non_zero_ind = np.nonzero(cov_obs_rm)

        newcovd = np.count_nonzero(cov_obs_rm[non_zero_ind[0][0], :])
        cov_obs_new = np.zeros((newcovd, newcovd))
        k = 0

        for j in range(len(cov_obs_rm[0, :])):
            cov_rm_j = cov_obs_rm[j, :]
            cov_obs_j = cov_obs_copy[j, :]
            nnzero_cov_obs_j = np.nonzero(cov_rm_j)
            if len(nnzero_cov_obs_j[0]) > 0:
                cov_obs_new[k, :] = cov_obs_j[nnzero_cov_obs_j]
                k += 1

        data_obs_new = np.delete(data_obs, del_indf)

        xcoord_obs_new = []
        for bins in bins_to_fit:
            xcoord_obs_new.append(xcoord_obs[bins - 1])

        for bins in bins_to_fit:
            xcoord_obs_new.append(xcoord_obs[len(bins_all) + bins - 1])

    else:
        cov_obs_new = np.copy(cov_obs)
        data_obs_new = np.copy(data_obs)
        xcoord_obs_new = np.copy(xcoord_obs)

    if stat_type in ['gg','kk']:
        data_obs_new, cov_obs_new = data_obs_new[0:len(bins_to_fit) * len(xcoord_obs[0])], cov_obs_new[
                                                                                      0:len(bins_to_fit) * len(
                                                                                          xcoord_obs[0]),
                                                                                      0:len(bins_to_fit) * len(
                                                                                          xcoord_obs[0])]

    if stat_type in ['gy','gty']:
        data_obs_new, cov_obs_new = data_obs_new[len(bins_to_fit) * len(xcoord_obs[0]):len(data_obs_new)], cov_obs_new[
                                                                                                      len(
                                                                                                          bins_to_fit) * len(
                                                                                                          xcoord_obs[0]):len(
                                                                                                          data_obs_new),
                                                                                                      len(
                                                                                                          bins_to_fit) * len(
                                                                                                          xcoord_obs[0]):len(
                                                                                                          data_obs_new)]

    return xcoord_obs_new, data_obs_new, cov_obs_new


def setup(options):
    bins_all = ast.literal_eval(options.get_string(option_section, "bins_source", "[1, 2, 3, 4, 5]"))
    bins_to_fit = ast.literal_eval(options.get_string(option_section, "bins_to_fit", "[1, 2, 3, 4, 5]"))
    xcoord_comp_min = ast.literal_eval(options.get_string(option_section, "xcoord_comp_min", "[0,0,0,0,0,0,0,0,0,0]"))
    xcoord_comp_max = ast.literal_eval(
        options.get_string(option_section, "xcoord_comp_max", "[10000,10000,10000,10000,10000,10000,10000,10000,10000,10000]"))
    cov_diag = options.get_bool(option_section, "cov_diag", False)
    no_cov_zbins_only_auto_cross = options.get_bool(option_section, "no_cov_zbins_only_auto_cross", False)
    no_cov_zbins_all = options.get_bool(option_section, "no_cov_zbins_all", False)
    no_cov_auto_cross = options.get_bool(option_section, "no_cov_auto_cross", False)
    stat_type = options.get_string(option_section, "stat_analyze", 'gg_gy')
    sec_save_name = options.get_string(option_section, "sec_save_name", 'save_theory')

    bins_to_rem = copy.deepcopy(bins_all)
    for bins in bins_to_fit:
        bins_to_rem.remove(bins)

    twopt_file_processed_file = options.get_string(option_section, "twopt_file")
    try:
        data = pk.load(open(twopt_file_processed_file, 'rb'))
    except:
        data = pk.load(open(twopt_file_processed_file, 'rb'),encoding='latin1')

    xcoord_obs, data_obs, cov_obs = data['xcoord_all'], data['mean'], data['cov_total']

    xcoord_obs_new, data_obs_new, cov_obs_new = import_data(xcoord_obs, data_obs, cov_obs, bins_to_rem, bins_to_fit, bins_all,
                                                       stat_type)

    data_obs_comp, xcoord_obs_comp, xcoord_obs_comp_ll, incov_obs_comp, cov_obs_comp = setuplnprob_func(xcoord_comp_min, xcoord_comp_max,
                                                                                              xcoord_obs_new, data_obs_new,
                                                                                              cov_obs_new, stat_type,
                                                                                              bins_to_fit,
                                                                                              cov_diag=cov_diag,
                                                                                              no_cov_zbins_only_auto_cross=no_cov_zbins_only_auto_cross,
                                                                                              no_cov_zbins_all=no_cov_zbins_all,
                                                                                              no_cov_auto_cross=no_cov_auto_cross)
    # import pdb; pdb.set_trace()

    return data_obs_comp, xcoord_obs_comp, xcoord_obs_comp_ll, incov_obs_comp, cov_obs_comp, stat_type, bins_to_fit, sec_save_name


def execute(block, config):
    data_obs_comp, xcoord_obs_comp, xcoord_obs_comp_ll, incov_obs_comp, cov_obs_comp, stat_type, bins_to_fit, sec_save_name = config
    try:
        like3d, corrf_theory_rdata = lnprob_func(block, xcoord_obs_comp_ll, data_obs_comp, incov_obs_comp, stat_type, bins_to_fit,sec_save_name)
    except:
        print(tb.format_exc())
    chi2 = -2. * like3d

    likes = names.likelihoods
    block[likes, 'SZ_LIKE'] = like3d
    block[likes, 'SZ_CHI2'] = chi2
    # block[likes, 'cov_obs_comp'] = cov_obs_comp
    # block[likes, 'incov_obs_comp'] = incov_obs_comp
    # block[likes, 'corrf_theory_rdata'] = corrf_theory_rdata
    # block[likes, 'corrf_data_gtcut'] = data_obs_comp

    # block["data_vector", '3d_inverse_covariance'] = incov_obs_comp
    # block["data_vector", '3d_theory'] = corrf_theory_rdata

    return 0


def cleanup(config):
    pass
