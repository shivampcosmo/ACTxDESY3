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



def get_corr(cov):
    corr = np.zeros(cov.shape)
    for ii in xrange(0, cov.shape[0]):
        for jj in xrange(0, cov.shape[1]):
            corr[ii, jj] = cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj])
    return corr

def get_theory_terms(block, ell_data, stat_type, bins_array, sec_savename = "save_theory"):
    Cl_theory_rdata = []
    ell_array = block.get_double_array_1d(sec_savename, "ell")
    if stat_type == 'gg':
        nbins = len(bins_array)
        for j in range(nbins):
            bin_j = bins_array[j]
            Cl_gg =  block.get_double_array_1d(sec_savename,'theory_Clgg_bin_' + str(bin_j) + '_' + str(bin_j))
            Cl_gg_temp = intspline(ell_array, Cl_gg)
            Cl_gg_f = Cl_gg_temp(ell_data[j])
            one_nbar =  1./(block[sec_save_name, 'nbar_Clgg_bin_' + str(bin_j)])
            if len(Cl_theory_rdata) == 0:
                Cl_theory_rdata = Cl_gg_f 
            else:
                Cl_theory_rdata = np.hstack((Cl_theory_rdata, Cl_gg_f))

    elif stat_type == 'gy':
        nbins = len(bins_array)
        for j in range(nbins):
            bin_j = bins_array[j]
            Cl_gy = block.get_double_array_1d(sec_savename,'theory_Clgy_bin_'+ str(bin_j) + '_' + str(bin_j))
            Cl_gy_temp = intspline(ell_array, Cl_gy)
            Cl_gy_f = Cl_gy_temp(ell_data[j])
            if len(Cl_theory_rdata) == 0:
                Cl_theory_rdata = Cl_gy_f
            else:
                Cl_theory_rdata = np.hstack((Cl_theory_rdata, Cl_gy_f))

    elif stat_type == 'gg_gy':
        nbins = len(bins_array)
        for j in range(nbins):
            bin_j = bins_array[j]
            Cl_gg = block.get_double_array_1d(sec_savename,'theory_Clgg_bin_' + str(bin_j) + '_' + str(bin_j))
            Cl_gg_temp = intspline(ell_array, Cl_gg)
            Cl_gg_f = Cl_gg_temp(ell_data[j])
            one_nbar =  1./(block[sec_save_name, 'nbar_Clgg_bin_' + str(bin_j)])
            if len(Cl_theory_rdata) == 0:
                Cl_theory_rdata = Cl_gg_f 
            else:
                Cl_theory_rdata = np.hstack((Cl_theory_rdata, Cl_gg_f ))

        for j in range(nbins):
            bin_j = bins_array[j]
            Cl_gy = block.get_double_array_1d(sec_savename,'theory_Clgy_bin_'+ str(bin_j) + '_' + str(bin_j))
            Cl_gy_temp = intspline(ell_array, Cl_gy)
            Cl_gy_f = Cl_gy_temp(ell_data[j + nbins])
            Cl_theory_rdata = np.hstack((Cl_theory_rdata, Cl_gy_f))

    return Cl_theory_rdata


def lnprob_func(block, ell_data, Cl_data_gtcut, incov_obs_comp, stat_type, bins_array,sec_save_name):
    Cl_theory_rdata = get_theory_terms(block, ell_data, stat_type, bins_array,sec_savename=sec_save_name)
    valf = -0.5 * np.dot(np.dot(np.transpose((Cl_data_gtcut - Cl_theory_rdata)), incov_obs_comp),
                         (Cl_data_gtcut - Cl_theory_rdata))
    return valf, Cl_theory_rdata


def setuplnprob_func(scale_cut_min, scale_cut_max, ell_data_array, Cl_data_full, cov_obs, stat_type, bins_array, cov_diag=False,
                     no_cov_zbins_only_gg_gy=False, no_cov_zbins_all=False, no_cov_gg_gy=False):
    ell_data_comp_ll = []

    selection = []
    countk = 0

    if stat_type == 'gg_gy':

        ell_data_gg_all = np.array([])
        ell_data_gy_all = np.array([])
        for j in range(len(bins_array)):
            if len(ell_data_gg_all) == 0:
                ell_data_gg_all = ell_data_array[j]
                ell_data_gy_all = ell_data_array[j + len(bins_array)]
            else:
                ell_data_gg_all = np.hstack((ell_data_gg_all, ell_data_array[j]))
                ell_data_gy_all = np.hstack((ell_data_gy_all, ell_data_array[j + len(bins_array)]))

        ell_data_all = np.hstack((ell_data_gg_all, ell_data_gy_all))

        for j in range(len(bins_array)):
            ell_data_j = ell_data_array[j]
            selection_j = np.where((ell_data_j >= scale_cut_min[j]) & (ell_data_j <= scale_cut_max[j]))[0]
            if len(selection) == 0:
                selection = selection_j
            else:
                selection = np.hstack((selection, countk + selection_j))
            ell_data_comp_j = ell_data_j[selection_j]
            ell_data_comp_ll.append(ell_data_comp_j)
            countk += len(ell_data_j)

        for j in range(len(bins_array)):
            ell_data_j = ell_data_array[j + len(bins_array)]
            selection_j = \
                np.where((ell_data_j >= scale_cut_min[j + len(bins_array)]) & (ell_data_j <= scale_cut_max[j + len(bins_array)]))[0]
            selection = np.hstack((selection, countk + selection_j))
            ell_data_comp_j = ell_data_j[selection_j]
            ell_data_comp_ll.append(ell_data_comp_j)
            countk += len(ell_data_j)

    if stat_type == 'gg':

        ell_data_gg_all = np.array([])
        for j in range(len(bins_array)):
            if len(ell_data_gg_all) == 0:
                ell_data_gg_all = ell_data_array[j]
            else:
                ell_data_gg_all = np.hstack((ell_data_gg_all, ell_data_array[j]))

        ell_data_all = ell_data_gg_all

        for j in range(len(bins_array)):
            ell_data_j = ell_data_array[j]
            selection_j = np.where((ell_data_j >= scale_cut_min[j]) & (ell_data_j <= scale_cut_max[j]))[0]
            if len(selection) == 0:
                selection = selection_j
            else:
                selection = np.hstack((selection, countk + selection_j))

            ell_data_comp_j = ell_data_j[selection_j]
            ell_data_comp_ll.append(ell_data_comp_j)
            countk += len(ell_data_j)

    if stat_type == 'gy':

        ell_data_gy_all = np.array([])

        for j in range(len(bins_array)):
            if len(ell_data_gy_all) == 0:
                ell_data_gy_all = ell_data_array[j + len(bins_array)]
            else:
                ell_data_gy_all = np.hstack((ell_data_gy_all, ell_data_array[j + len(bins_array)]))

        ell_data_all = ell_data_gy_all

        for j in range(len(bins_array)):
            ell_data_j = ell_data_array[j + len(bins_array)]
            selection_j = \
                np.where((ell_data_j >= scale_cut_min[j + len(bins_array)]) & (ell_data_j <= scale_cut_max[j + len(bins_array)]))[0]
            if len(selection) == 0:
                selection = selection_j
            else:
                selection = np.hstack((selection, countk + selection_j))

            ell_data_comp_j = ell_data_j[selection_j]
            ell_data_comp_ll.append(ell_data_comp_j)
            countk += len(ell_data_j)

    selection = np.array(selection)
    cov_obs_comp = (cov_obs[:, selection])[selection, :]

    if no_cov_zbins_only_gg_gy or no_cov_zbins_all:
        bins_n_array = np.arange(len(bins_array))
        cov_obs_comp_h = np.copy(cov_obs_comp)

        if len(bins_array) > 1:
            print 'zeroing the covariance between z bins'

            if stat_type == 'gg_gy':
                z1_0 = []
                for ji in range(len(bins_array)):
                    if len(z1_0) == 0:
                        z1_0 = bins_n_array[ji] * np.ones(len(ell_data_comp_ll[ji]))
                    else:
                        z1_0 = np.hstack((z1_0, bins_n_array[ji] * np.ones(len(ell_data_comp_ll[ji]))))

                z1_1 = []
                for ji in range(len(bins_array)):
                    if len(z1_1) == 0:
                        z1_1 = bins_n_array[ji] * np.ones(len(ell_data_comp_ll[len(bins_array) + ji]))
                    else:
                        z1_1 = np.hstack((z1_1, bins_n_array[ji] * np.ones(len(ell_data_comp_ll[len(bins_array) + ji]))))

                z1_mat_0 = np.tile(z1_0, (len(z1_0), 1)).transpose()
                z1_mat_1 = np.tile(z1_1, (len(z1_1), 1)).transpose()

                z1_mat_01 = np.tile(z1_0, (len(z1_1), 1)).transpose()
                z1_mat_10 = np.tile(z1_1, (len(z1_0), 1)).transpose()

                if no_cov_zbins_only_gg_gy:
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

            if stat_type == 'gg' or stat_type == 'gy':
                z1 = np.repeat(np.arange(len(bins_array)), len(ell_data_comp_ll[0]))
                z1_mat = np.tile(z1, (len(bins_array) * len(ell_data_comp_ll[0]), 1)).transpose()
                z2_mat = np.transpose(z1_mat)
                offdiag = np.where(z1_mat != z2_mat)
                cov_obs_comp_h[offdiag] = 0.0

        cov_obs_comp = cov_obs_comp_h

    if no_cov_gg_gy:
        bins_n_array = np.arange(len(bins_array)) + 1
        cov_obs_comp_hf = np.copy(cov_obs_comp)
        print 'zeroing the covariance between gg and gy'
        if stat_type == 'gg_gy':
            z1_0 = []
            for ji in range(len(bins_array)):
                if len(z1_0) == 0:
                    z1_0 = bins_n_array[ji] * np.ones(len(ell_data_comp_ll[ji]))
                else:
                    z1_0 = np.hstack((z1_0, bins_n_array[ji] * np.ones(len(ell_data_comp_ll[ji]))))

            z1_1 = []
            for ji in range(len(bins_array)):
                if len(z1_1) == 0:
                    z1_1 = -1. * bins_n_array[ji] * np.ones(len(ell_data_comp_ll[len(bins_array) + ji]))
                else:
                    z1_1 = np.hstack((z1_1, -1. * bins_n_array[ji] * np.ones(len(ell_data_comp_ll[len(bins_array) + ji]))))

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

    incov_obs_comp = np.linalg.inv(cov_obs_comp)
    ell_data_comp = ell_data_all[selection]
    Cl_data_gtcut = Cl_data_full[selection]

    return Cl_data_gtcut, ell_data_comp, ell_data_comp_ll, incov_obs_comp, cov_obs_comp

def import_data(ell_obs, data_obs, cov_obs, bins_to_rem, bins_to_fit, bins_all, stat_type):
    if len(bins_to_rem) > 0:
        cov_obs_rm = np.ones(cov_obs.shape)
        cov_obs_copy = np.copy(cov_obs)

        z1_0 = []
        for ji in range(len(bins_all)):
            if len(z1_0) == 0:
                z1_0 = bins_all[ji] * np.ones(len(ell_obs[ji]))
            else:
                z1_0 = np.hstack((z1_0, bins_all[ji] * np.ones(len(ell_obs[ji]))))

        z1_1 = []
        for ji in range(len(bins_all)):
            if len(z1_1) == 0:
                z1_1 = bins_all[ji] * np.ones(len(ell_obs[len(bins_all) + ji]))
            else:
                z1_1 = np.hstack((z1_1, bins_all[ji] * np.ones(len(ell_obs[len(bins_all) + ji]))))

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

        ell_obs_new = []
        for bins in bins_to_fit:
            ell_obs_new.append(ell_obs[bins - 1])

        for bins in bins_to_fit:
            ell_obs_new.append(ell_obs[len(bins_all) + bins - 1])

    else:
        cov_obs_new = np.copy(cov_obs)
        data_obs_new = np.copy(data_obs)
        ell_obs_new = np.copy(ell_obs)

    if stat_type == 'gg':
        data_obs_new, cov_obs_new = data_obs_new[0:len(bins_to_fit) * len(ell_obs[0])], cov_obs_new[
                                                                                      0:len(bins_to_fit) * len(
                                                                                          ell_obs[0]),
                                                                                      0:len(bins_to_fit) * len(
                                                                                          ell_obs[0])]

    if stat_type == 'gy':
        data_obs_new, cov_obs_new = data_obs_new[len(bins_to_fit) * len(ell_obs[0]):len(data_obs_new)], cov_obs_new[
                                                                                                      len(
                                                                                                          bins_to_fit) * len(
                                                                                                          ell_obs[0]):len(
                                                                                                          data_obs_new),
                                                                                                      len(
                                                                                                          bins_to_fit) * len(
                                                                                                          ell_obs[0]):len(
                                                                                                          data_obs_new)]

    return ell_obs_new, data_obs_new, cov_obs_new


def setup(options):
    bins_all = ast.literal_eval(options.get_string(option_section, "bins_all", "[1, 2, 3, 4, 5]"))
    bins_to_fit = ast.literal_eval(options.get_string(option_section, "bins_to_fit", "[1, 2, 3, 4, 5]"))
    ell_comp_min = ast.literal_eval(options.get_string(option_section, "ell_comp_min", "[0,0,0,0,0,0,0,0,0,0]"))
    ell_comp_max = ast.literal_eval(
        options.get_string(option_section, "ell_comp_max", "[10000,10000,10000,10000,10000,10000,10000,10000,10000,10000]"))
    cov_diag = options.get_bool(option_section, "cov_diag", False)
    no_cov_zbins_only_gg_gy = options.get_bool(option_section, "no_cov_zbins_only_gg_gy", False)
    no_cov_zbins_all = options.get_bool(option_section, "no_cov_zbins_all", False)
    no_cov_gg_gy = options.get_bool(option_section, "no_cov_gg_gy", False)
    stat_type = options.get_string(option_section, "stat_type", 'gg_gy')
    sec_save_name = options.get_string(option_section, "sec_save_name", 'save_theory')

    bins_to_rem = copy.deepcopy(bins_all)
    for bins in bins_to_fit:
        bins_to_rem.remove(bins)

    twopt_file_processed_file = options.get_string(option_section, "twopt_file_processed_file")
    data = pk.load(open(twopt_file_processed_file, 'rb'))
    ell_obs, data_obs, cov_obs = data['ell_all'], data['mean'], data['cov_total']

    ell_obs_new, data_obs_new, cov_obs_new = import_data(ell_obs, data_obs, cov_obs, bins_to_rem, bins_to_fit, bins_all,
                                                       stat_type)

    data_obs_comp, ell_obs_comp, ell_obs_comp_ll, incov_obs_comp, cov_obs_comp = setuplnprob_func(ell_comp_min, ell_comp_max,
                                                                                              ell_obs_new, data_obs_new,
                                                                                              cov_obs_new, stat_type,
                                                                                              bins_to_fit,
                                                                                              cov_diag=cov_diag,
                                                                                              no_cov_zbins_only_gg_gy=no_cov_zbins_only_gg_gy,
                                                                                              no_cov_zbins_all=no_cov_zbins_all,
                                                                                              no_cov_gg_gy=no_cov_gg_gy)
    return data_obs_comp, ell_obs_comp, ell_obs_comp_ll, incov_obs_comp, cov_obs_comp, stat_type, bins_to_fit, sec_save_name


def execute(block, config):
    data_obs_comp, ell_obs_comp, ell_obs_comp_ll, incov_obs_comp, cov_obs_comp, stat_type, bins_to_fit, sec_save_name = config
    like3d, Cl_theory_rdata = lnprob_func(block, ell_obs_comp_ll, data_obs_comp, incov_obs_comp, stat_type, bins_to_fit,sec_save_name)
    chi2 = -2. * like3d

    likes = names.likelihoods
    block[likes, 'SZ_LIKE'] = like3d
    block[likes, 'SZ_CHI2'] = chi2
    block[likes, 'cov_obs_comp'] = cov_obs_comp
    block[likes, 'incov_obs_comp'] = incov_obs_comp
    block[likes, 'Cl_theory_rdata'] = Cl_theory_rdata
    block[likes, 'Cl_data_gtcut'] = data_obs_comp

    block["data_vector", '3d_inverse_covariance'] = incov_obs_comp
    block["data_vector", '3d_theory'] = Cl_theory_rdata

    return 0


def cleanup(config):
    pass
