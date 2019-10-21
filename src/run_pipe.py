import sys, os
import pdb
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
from scipy import interpolate
import matplotlib.pyplot as plt
import copy
import configparser
import ast
import dill
import pdb
from configobj import ConfigObj
from configparser import ConfigParser

sys.path.insert(0, '../helper/')
import plot_funcs as pf
from haloyforecast import DataVec
import fisher_plotter as fplot
import get_nz
import LSS_funcs as hmf
import string
import random
import time
pi = np.pi

font = {'size': 18}
matplotlib.rc('font', **font)
plt.rc('text', usetex=False)
plt.rc('font', family='serif')


class Fisher:

    def __init__(self, cosmo_params_fid, hod_params_fid, pressure_params_fid, other_params_fid,
                 fisher_params_vary_dict):

        self.fisher_params_vary_dict = fisher_params_vary_dict

        DV_fid = DataVec(cosmo_params_fid, hod_params_fid, pressure_params_fid, other_params_fid)

        self.verbose = other_params_fid['verbose']

        self.Cl_fid = DV_fid.get_Cl_vector()
        cov_fid_dict_G = DV_fid.get_cov_G()
        cov_fid_dict_NG = DV_fid.get_cov_NG()
        self.Cl_all_dict = DV_fid.Cl_dict
        self.l_array_survey = DV_fid.l_array_survey
        self.dl_array_survey = DV_fid.dl_array_survey

        if not other_params_fid['do_vary_cosmo']:
            other_params_fid['pkzlin_interp'] = DV_fid.PS.pkzlin_interp
            other_params_fid['dndm_array'] = DV_fid.PS.dndm_array
            other_params_fid['bm_array'] = DV_fid.PS.bm_array
            other_params_fid['halo_conc_mdef'] = DV_fid.PS.halo_conc_mdef

        other_params_fid['Mmat_cond'] = DV_fid.PS.M_mat_cond_inbin
        other_params_fid['zmat_cond'] = DV_fid.PS.z_mat_cond_inbin

        self.stats_analyze = other_params_fid['stats_analyze']

        stats_analyze_pairs_all = DV_fid.stats_analyze_pairs_all
        stats_analyze_1, stats_analyze_2 = stats_analyze_pairs_all[0]

        nl = cov_fid_dict_G[stats_analyze_1 + '_' + stats_analyze_2].shape[0]
        nstats = len(other_params_fid['stats_analyze'])

        cov_fid_G = np.zeros((nstats * nl, nstats * nl))
        cov_fid_NG = np.zeros((nstats * nl, nstats * nl))

        self.get_realspace_cov = other_params_fid['do_compare_to_configspace']

        if self.get_realspace_cov:
            l_array_full_all = np.logspace(np.log10(1.0), np.log10(1e4), 16000)
            self.dl_array_full = l_array_full_all[1:] - l_array_full_all[:-1]
            self.l_array_full = l_array_full_all[1:]

            self.nl_full = len(self.l_array_full)

            ng_zarray = other_params_fid['ng_zarray']
            ng_value = other_params_fid['ng_value']
            self.zmean = sp.integrate.simps(ng_value * ng_zarray, ng_zarray) / sp.integrate.simps(ng_value, ng_zarray)

            self.chi_val = hmf.get_Dcom(self.zmean, cosmo_params_fid['Om0'])
            self.rp_min = 0.1
            self.rp_max = 30.0

            self.theta_min_rad = self.rp_min/self.chi_val
            self.theta_max_rad = self.rp_max/self.chi_val
            self.ntheta = 10


            self.theta_min = self.theta_min_rad * (180.0 * 60.0 / np.pi)
            self.theta_max = self.theta_max_rad * (180.0 * 60.0 / np.pi)

            self.cov_G_fid_theta_dict = {}

        k = 0
        for j1 in range(nstats):
            for j2 in range(nstats):
                stats_analyze_1, stats_analyze_2 = stats_analyze_pairs_all[k]
                if self.verbose:
                    print('fitting ', stats_analyze_1 + '_' + stats_analyze_2, ' in block ', k)

                cov_G_j1j2 = cov_fid_dict_G[stats_analyze_1 + '_' + stats_analyze_2]
                cov_NG_j1j2 = cov_fid_dict_NG[stats_analyze_1 + '_' + stats_analyze_2]

                cov_fid_G[j2 * nl:(j2 + 1) * nl, j1 * nl:(j1 + 1) * nl] = cov_G_j1j2
                cov_fid_NG[j2 * nl:(j2 + 1) * nl, j1 * nl:(j1 + 1) * nl] = cov_NG_j1j2

                # pdb.set_trace()

                if self.get_realspace_cov and (stats_analyze_1 == stats_analyze_2):
                    cov_G_diag_interp = interpolate.interp1d(np.log(self.l_array_survey),
                                                             np.log(self.dl_array_survey * np.diag(cov_G_j1j2)),
                                                             fill_value=-100.0, bounds_error=False)
                    cov_G_diag_lfull = (np.exp(cov_G_diag_interp(np.log(self.l_array_full)))) / self.dl_array_full

                    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    # ax.plot(self.l_array_survey, np.diag(cov_G_j1j2), label='Original')
                    # ax.plot(self.l_array_full, cov_G_diag_lfull, label='Extrapolated')
                    # ax.set_yscale('log')
                    # ax.set_xscale('log')
                    # ax.legend(fontsize=15, frameon=False)
                    # ax.set_xlabel(r'$\ell$', size=20)
                    # ax.set_ylabel(r'Cov Gaussian', size=20)
                    # plt.tick_params(axis='both', which='major', labelsize=15)
                    # plt.tick_params(axis='both', which='minor', labelsize=15)
                    # plt.tight_layout()
                    # plot_save_name = '/global/u1/s/spandey/actxdes/sz_forecasts/forecasts/plots/cov_comp.png'
                    # plt.savefig(plot_save_name)
                    # plt.close()

                    theta_array, cov_G_theta_j1j2 = DV_fid.get_covdiag_wtheta(self.theta_min, self.theta_max,
                                                                              self.ntheta, self.l_array_full,
                                                                              cov_G_diag_lfull)

                    # cov_G_fid_theta[j2 * self.ntheta:(j2 + 1) * self.ntheta,j1 * self.ntheta:(j1 + 1) * self.ntheta] = cov_G_theta_j1j2

                    self.cov_G_fid_theta_dict[stats_analyze_1 + '_' + stats_analyze_2] = cov_G_theta_j1j2

                k += 1

        if self.verbose:
            print('cov NG shape ', cov_fid_NG.shape, ' and cov G shape ', cov_fid_G.shape)
            print('larray ', DV_fid.l_array_survey)
            print('diagonal elements of sig G ', np.sqrt(np.diag(cov_fid_G)))
            print('diagonal elements of sig NG ', np.sqrt(np.diag(cov_fid_NG)))

        if self.get_realspace_cov:
            l_array = other_params_fid['l_array']
            theta_array_all = np.logspace(np.log10(self.theta_min), np.log10(self.theta_max), self.ntheta)
            theta_array = (theta_array_all[1:] + theta_array_all[:-1]) / 2.
            theta_array_rad = theta_array * (np.pi / 180.) * (1. / 60.)
            l_array_full = np.logspace(0, 4.1, 14000)
            Cl_yg_1h_array = self.Cl_all_dict['yg']['1h']
            Cl_yg_2h_array = self.Cl_all_dict['yg']['2h']
    
            Cl_yg_1h_interp = interpolate.interp1d(np.log(l_array), Cl_yg_1h_array, fill_value=0.0, bounds_error=False)
            Cl_yg_1h_full = Cl_yg_1h_interp(np.log(l_array_full))
            Cl_yg_2h_interp = interpolate.interp1d(np.log(l_array), Cl_yg_2h_array, fill_value=0.0, bounds_error=False)
            Cl_yg_2h_full = Cl_yg_2h_interp(np.log(l_array_full))

            wtheta_yg_1h = np.zeros(len(theta_array_rad))
            wtheta_yg_2h = np.zeros(len(theta_array_rad))
            wtheta_yg = np.zeros(len(theta_array_rad))
    
            for j in range(len(theta_array_rad)):
                wtheta_yg_1h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yg_1h_full)
                wtheta_yg_2h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yg_2h_full)
                wtheta_yg[j] = wtheta_yg_1h[j] + wtheta_yg_2h[j]

            self.sigG_gy_array = np.sqrt(np.diag(self.cov_G_fid_theta_dict['gy_gy']))
            self.rp_array = theta_array_rad * self.chi_val
            self.wtheta_yg = wtheta_yg

        if other_params_fid['do_plot']:
            DV_fid.plot(plot_dir=other_params_fid['plot_dir'], cov_fid_dict_G=cov_fid_dict_G,
                        cov_fid_dict_NG=cov_fid_dict_NG)
            if self.get_realspace_cov:

                pf.plot_wtheta(other_params_fid['l_array'], self.Cl_all_dict, self.theta_min, self.theta_max,
                               self.ntheta, other_params_fid['save_suffix'], other_params_fid['plot_dir'],
                               cov_wtheta_G_dict=self.cov_G_fid_theta_dict, chi_zeval=self.chi_val)
                # pf.plot_wtheta(other_params_fid['l_array'], self.Cl_all_dict, self.theta_min, self.theta_max,
                #                self.ntheta, other_params_fid['save_suffix'], other_params_fid['plot_dir'])

        # pdb.set_trace()

        self.cov_fid = cov_fid_G + cov_fid_NG
        # pdb.set_trace()
        self.inv_cov_fid = sp.linalg.inv(self.cov_fid)

        self.Cl_noise_yy_l_array = DV_fid.Cl_noise_yy_l_array
        self.Cl_noise_gg_l_array = DV_fid.Cl_noise_gg_l_array

        deltaparam_param = other_params_fid['deltaparam_param']
        self.save_suffix = other_params_fid['save_suffix']
        self.output_save_name = other_params_fid['output_save_name']

        fisher_params_vary_orig = fisher_params_vary_dict['params_to_vary']
        fisher_params_label_orig = fisher_params_vary_dict['params_label']
        self.priors_dict = fisher_params_vary_dict['priors_dict']

        fisher_params_vary = copy.deepcopy(fisher_params_vary_orig)
        fisher_params_label = copy.deepcopy(fisher_params_label_orig)

        if other_params_fid['do_split_params_massbins']:
            fisher_split_params_massbins_vary_names = fisher_params_vary_dict[
                'fisher_split_params_massbins_vary'].keys()
            split_mass_bins_min = other_params_fid['split_mass_bins_min']
            split_mass_bins_max = other_params_fid['split_mass_bins_max']
            for j in range(len(fisher_split_params_massbins_vary_names)):
                if fisher_split_params_massbins_vary_names[j] in fisher_params_vary:
                    ind_of_param = fisher_params_vary.index(fisher_split_params_massbins_vary_names[j])
                    fisher_param_vary_name = fisher_params_vary[ind_of_param]
                    fisher_param_vary_label = fisher_params_label[ind_of_param]
                    mass_bins_to_vary_param_j = fisher_params_vary_dict['fisher_split_params_massbins_vary'][
                        fisher_split_params_massbins_vary_names[j]]
                    k = ind_of_param
                    for i in range(len(mass_bins_to_vary_param_j)):
                        if mass_bins_to_vary_param_j[i] == 1:
                            fisher_param_vary_name_bini = fisher_param_vary_name + '-bin_' + str(i)
                            fisher_param_vary_label_split = list(fisher_param_vary_label)
                            log_mass_bin_i_min = np.round(np.log10(split_mass_bins_min[i]), 1)
                            log_mass_bin_i_max = np.round(np.log10(split_mass_bins_max[i]), 1)

                            if (other_params_fid['pressure_model_type'] in [
                                'broken_powerlaw'] and fisher_param_vary_name in ['alpha_p']) or (
                                    other_params_fid['pressure_model_type'] in [
                                'superbroken_powerlaw'] and fisher_param_vary_name in ['alpha_p', 'beta']) or (
                                    other_params_fid['pressure_model_type'] in [
                                'superPlybroken_powerlaw'] and fisher_param_vary_name in ['alpha_p', 'P0']) or (
                                    other_params_fid['pressure_model_type'] in [
                                'fullybroken_powerlaw'] and fisher_param_vary_name in ['alpha_p', 'beta', 'P0']):
                                if i == 0:
                                    fisher_param_vary_label_bini = ''.join(
                                        fisher_param_vary_label_split[:-1]) + '[(10^{' + str(
                                        log_mass_bin_i_min) + '}) < M_h < (M_{\star}=' + '10^{' + str(
                                        pressure_params_fid['logMstar'][fisher_param_vary_name]) + '})]' + \
                                                                   fisher_param_vary_label_split[-1]
                                if i == 1:
                                    fisher_param_vary_label_bini = ''.join(
                                        fisher_param_vary_label_split[:-1]) + '[(M_{\star}=' + '10^{' + str(
                                        pressure_params_fid['logMstar'][
                                            fisher_param_vary_name]) + '}) < M_h < (10^{' + str(
                                        log_mass_bin_i_max) + '})]' + \
                                                                   fisher_param_vary_label_split[-1]
                            else:
                                fisher_param_vary_label_bini = ''.join(
                                    fisher_param_vary_label_split[:-1]) + '[10^{' + str(
                                    log_mass_bin_i_min) + '} < M_h < 10^{' + str(log_mass_bin_i_max) + '}]' + \
                                                               fisher_param_vary_label_split[-1]
                            fisher_params_vary.insert(k, fisher_param_vary_name_bini)
                            fisher_params_label.insert(k, fisher_param_vary_label_bini)
                            k += 1
                    fisher_params_vary.remove(fisher_param_vary_name)
                    fisher_params_label.remove(fisher_param_vary_label)

        param_vary_new = {}
        param_vary_fid = {}
        deltaparam_vary = {}
        for j in range(len(fisher_params_vary)):
            param_vary = fisher_params_vary[j]
            if param_vary not in fisher_params_vary_orig:
                param_vary_split = list(param_vary)
                addition_index = param_vary_split.index('-')
                param_vary_orig = ''.join(param_vary_split[:addition_index])
            else:
                param_vary_orig = param_vary

            if param_vary_orig in cosmo_params_fid.keys():
                params_new = copy.deepcopy(cosmo_params_fid)
                params_new[param_vary] = cosmo_params_fid[param_vary] * (1. + deltaparam_param)
                param_vary_new[param_vary] = [params_new, hod_params_fid, pressure_params_fid]
                deltaparam_vary[param_vary] = cosmo_params_fid[param_vary] * deltaparam_param
                param_vary_fid[param_vary] = cosmo_params_fid[param_vary]

            if param_vary_orig in hod_params_fid.keys():
                params_new = copy.deepcopy(hod_params_fid)
                params_new[param_vary] = hod_params_fid[param_vary] * (1. + deltaparam_param)
                param_vary_new[param_vary] = [cosmo_params_fid, params_new, pressure_params_fid]
                deltaparam_vary[param_vary] = hod_params_fid[param_vary] * deltaparam_param
                param_vary_fid[param_vary] = hod_params_fid[param_vary]

            if param_vary_orig in pressure_params_fid.keys():
                params_new = copy.deepcopy(pressure_params_fid)
                if other_params_fid['do_split_params_massbins']:
                    if param_vary not in fisher_params_vary_orig:
                        if str.isdigit(list(param_vary)[-1]):
                            mass_bin_number = int(list(param_vary)[-1])
                        if str.isdigit(list(param_vary)[-2]):
                            mass_bin_number += 10 * int(list(param_vary)[-2])

                        params_new_param_vary = copy.deepcopy(pressure_params_fid[param_vary_orig])
                        params_new_param_vary[mass_bin_number] = params_new_param_vary[mass_bin_number] * (
                                1. + deltaparam_param)
                        params_new[param_vary_orig] = params_new_param_vary
                        deltaparam_vary[param_vary] = params_new_param_vary[mass_bin_number] * deltaparam_param
                        param_vary_fid[param_vary] = params_new_param_vary[mass_bin_number]
                        param_vary_new[param_vary] = [cosmo_params_fid, hod_params_fid, params_new]

                        self.priors_dict[param_vary] = self.priors_dict[param_vary_orig][mass_bin_number]

                    else:
                        if pressure_params_fid[param_vary] != 0:
                            params_new[param_vary] = pressure_params_fid[param_vary] * (1. + deltaparam_param)
                            deltaparam_vary[param_vary] = pressure_params_fid[param_vary] * deltaparam_param
                            param_vary_fid[param_vary] = pressure_params_fid[param_vary]
                        else:
                            params_new[param_vary] = deltaparam_param
                            deltaparam_vary[param_vary] = deltaparam_param
                            param_vary_fid[param_vary] = 0.0
                        param_vary_new[param_vary] = [cosmo_params_fid, hod_params_fid, params_new]
                else:
                    if pressure_params_fid[param_vary] != 0:
                        params_new[param_vary] = pressure_params_fid[param_vary] * (1. + deltaparam_param)
                        deltaparam_vary[param_vary] = pressure_params_fid[param_vary] * deltaparam_param
                        param_vary_fid[param_vary] = pressure_params_fid[param_vary]
                    else:
                        params_new[param_vary] = deltaparam_param
                        deltaparam_vary[param_vary] = deltaparam_param
                        param_vary_fid[param_vary] = 0.0
                    param_vary_new[param_vary] = [cosmo_params_fid, hod_params_fid, params_new]

            # pdb.set_trace()

            if param_vary not in self.priors_dict.keys():
                self.priors_dict[param_vary] = 1e20

        self.param_vary_new = param_vary_new
        self.param_vary_fid = param_vary_fid
        self.deltaparam_vary = deltaparam_vary

        self.cosmo_params_fid = cosmo_params_fid
        self.hod_params_fid = hod_params_fid
        self.pressure_params_fid = pressure_params_fid

        self.cov_fid_dict_G = cov_fid_dict_G
        self.cov_fid_dict_NG = cov_fid_dict_NG

        self.other_params_fid = copy.deepcopy(other_params_fid)
        self.fisher_params_vary = copy.deepcopy(fisher_params_vary)
        self.fisher_params_label = fisher_params_label
        self.fisher_params_vary_orig = fisher_params_vary_orig
        self.fisher_params_label_orig = fisher_params_label_orig
        self.do_split_params_massbins = other_params_fid['do_split_params_massbins']
        if other_params_fid['do_split_params_massbins']:
            self.split_mass_bins_min = other_params_fid['split_mass_bins_min']

        if other_params_fid['do_plot']:
            if self.verbose:
                print("plotting fisher stuff")
            pf.plot_cov(cov_fid_G, cov_fid_NG, plot_dir=other_params_fid['plot_dir'],
                        save_suffix=other_params_fid['save_suffix'])

    def get_fisher_mat(self):

        other_params_new = copy.deepcopy(self.other_params_fid)
        dCl_dparam_dict = {}
        nparam = len(self.fisher_params_vary)
        # pdb.set_trace()
        for j in range(nparam):

            param_vary = self.fisher_params_vary[j]
            deltaparam = self.deltaparam_vary[param_vary]
            cosmo_params_new, hod_params_new, pressure_params_new = self.param_vary_new[param_vary]

            if self.verbose:
                print 'setting up param : ', param_vary
                ti = time.time()

            DV_new = DataVec(cosmo_params_new, hod_params_new, pressure_params_new, other_params_new)

            if self.verbose:
                print('that took ', time.time() - ti, 'seconds')

            Cl_new = DV_new.get_Cl_vector()
            dCl_dparam = (Cl_new - self.Cl_fid) / deltaparam
            dCl_dparam_dict[param_vary] = dCl_dparam
            if self.verbose:
                print(param_vary, ' ,dCl_dparam: ', dCl_dparam)
            dlnCl_d_lnparam = (self.param_vary_fid[param_vary] / self.Cl_fid) * dCl_dparam

            M_cen = ((self.other_params_fid['log_M_min_tracer']) + (self.other_params_fid['log_M_max_tracer']))/2.
            Delta_Mcen = ((self.other_params_fid['log_M_max_tracer']) - (self.other_params_fid['log_M_min_tracer']))

            # filesave_name = './output/dlnCl_dlnparam/res_' + self.stats_analyze[0] + '_logM_' + str(M_cen) + '_dlogM_' + str(Delta_Mcen) + '.txt'
            #
            # np.savetxt(filesave_name,np.array([self.l_array_survey,dlnCl_d_lnparam]).T)

            # if self.other_params_fid['do_plot']:
            #     pf.plot_dlnCl_dlnparam(self.l_array_survey, dCl_dparam, self.Cl_fid, self.param_vary_fid[param_vary], param_vary, self.fisher_params_label[j],
            #             self.other_params_fid['save_suffix'],self.other_params_fid['plot_dir'],stat = self.stats_analyze[0])

        F_mat = np.zeros((nparam, nparam))

        for j1 in range(nparam):
            for j2 in range(nparam):
                param_vary_j1 = self.fisher_params_vary[j1]
                param_vary_j2 = self.fisher_params_vary[j2]

                dCl_dparam_j1 = dCl_dparam_dict[param_vary_j1]
                dCl_dparam_j2 = dCl_dparam_dict[param_vary_j2]

                val = np.matmul(np.array([dCl_dparam_j1]), np.matmul(self.inv_cov_fid , np.array([dCl_dparam_j2]).T))

                F_mat[j1, j2] = val



        self.F_mat = F_mat
        self.dCl_dparam_dict = dCl_dparam_dict
        return F_mat

    def get_fid_values(self):
        if hasattr(self, 'F_mat'):
            F_mat = self.F_mat
        else:
            F_mat = self.get_fisher_mat()

        # pdb.set_trace()
        fid_values = np.zeros(F_mat.shape[0])

        for fi in xrange(0, len(fid_values)):

            param_vary = self.fisher_params_vary[fi]
            if param_vary not in self.fisher_params_vary_orig:
                param_vary_split = list(param_vary)
                addition_index = param_vary_split.index('-')
                param_vary_orig = ''.join(param_vary_split[:addition_index])
            else:
                param_vary_orig = param_vary

            if param_vary_orig in self.cosmo_params_fid:
                fid_values[fi] = self.cosmo_params_fid[param_vary_orig]
            if param_vary_orig in self.hod_params_fid:
                fid_values[fi] = self.hod_params_fid[param_vary_orig]
            if param_vary_orig in self.pressure_params_fid:
                if self.do_split_params_massbins:
                    if param_vary not in self.fisher_params_vary_orig:
                        if str.isdigit(list(param_vary)[-1]):
                            mass_bin_number = int(list(param_vary)[-1])
                        if str.isdigit(list(param_vary)[-2]):
                            mass_bin_number += 10 * int(list(param_vary)[-2])
                        fid_values[fi] = self.pressure_params_fid[param_vary_orig][mass_bin_number]
                    else:
                        fid_values[fi] = self.pressure_params_fid[param_vary_orig]
                else:
                    fid_values[fi] = self.pressure_params_fid[param_vary_orig]

        self.fid_values = fid_values
        return fid_values

    def set_values(self, cosmo_params, hod_params, pressure_params, new_values):

        for fi in xrange(0, len(self.fisher_params_vary)):

            param_vary = self.fisher_params_vary[fi]
            if param_vary not in self.fisher_params_vary_orig:
                param_vary_split = list(param_vary)
                addition_index = param_vary_split.index('-')
                param_vary_orig = ''.join(param_vary_split[:addition_index])
            else:
                param_vary_orig = param_vary

            if param_vary_orig in self.cosmo_params_fid:
                cosmo_params[param_vary_orig] = new_values[fi]
            if param_vary_orig in self.hod_params_fid:
                hod_params[param_vary_orig] = new_values[fi]
            if param_vary_orig in self.pressure_params_fid:

                if self.do_split_params_massbins:
                    if param_vary not in self.fisher_params_vary_orig:
                        if str.isdigit(list(param_vary)[-1]):
                            mass_bin_number = int(list(param_vary)[-1])
                        if str.isdigit(list(param_vary)[-2]):
                            mass_bin_number += 10 * int(list(param_vary)[-2])
                        params_new_param_vary = copy.deepcopy(pressure_params[param_vary_orig])
                        params_new_param_vary[mass_bin_number] = new_values[fi]
                        pressure_params[param_vary_orig] = params_new_param_vary
                    else:
                        pressure_params[param_vary_orig] = new_values[fi]
                else:
                    pressure_params[param_vary_orig] = new_values[fi]

        return cosmo_params, hod_params, pressure_params


def get_value(section, value, config_run, config_def):
    if section in config_run.keys() and value in config_run[section].keys():
        val = config_run[section][value]
    else:
        val = config_def[section][value]
    return val


def read_ini(ini_file, ini_def=None, save_params=True):
    config_run = ConfigObj(ini_file, unrepr=True)
    if ini_def is None:
        config_def = ConfigObj(config_run['DEFAULT']['params_default_file'], unrepr=True)
    else:
        config_def = ConfigObj(ini_def, unrepr=True)

    # pdb.set_trace()
    cosmo_params_keys = config_def['COSMO'].keys()
    cosmo_params_dict = {}
    for key in cosmo_params_keys:
        cosmo_params_dict[key] = get_value('COSMO', key, config_run, config_def)

    hod_type = config_run['ANALYSIS']['hod_type']
    hod_params_dict = {'hod_type': hod_type}
    hod_params_keys = config_def['HOD'][hod_type].keys()
    for key in hod_params_keys:
        if 'HOD' in config_run.keys() and key in config_run['HOD'].keys():
            hod_params_dict[key] = config_run['HOD'][key]
        else:
            hod_params_dict[key] = config_def['HOD'][hod_type][key]

    pressure_model_string = config_run['ANALYSIS']['pressure_model']
    ind_splits = [i for i, e in enumerate(list(pressure_model_string)) if e == '-']
    pressure_model_name = ''.join(list(pressure_model_string)[:ind_splits[0]])
    pressure_model_feedback = ''.join(list(pressure_model_string)[ind_splits[0] + 1:ind_splits[1]])
    pressure_model_mdef = ''.join(list(pressure_model_string)[ind_splits[1] + 1:ind_splits[2]])
    pressure_model_type = ''.join(list(pressure_model_string)[ind_splits[2] + 1:])
    pressure_params_dict = {}
    pressure_params_keys = config_def['PRESSURE'][pressure_model_name][
        pressure_model_feedback + '-' + pressure_model_mdef].keys()
    for key in pressure_params_keys:
        if 'PRESSURE' in config_run.keys() and key in config_run['PRESSURE'].keys():
            pressure_params_dict[key] = config_run['PRESSURE'][key]
        else:
            pressure_params_dict[key] = \
                config_def['PRESSURE'][pressure_model_name][pressure_model_feedback + '-' + pressure_model_mdef][key]

    other_params_default_keys = config_def['DEFAULT'].keys()
    other_params_analysis_keys = config_def['ANALYSIS'].keys()
    other_params_dict = {}
    for key in other_params_default_keys:
        if 'DEFAULT' in config_run.keys() and key in config_run['DEFAULT'].keys():
            other_params_dict[key] = config_run['DEFAULT'][key]
        else:
            other_params_dict[key] = config_def['DEFAULT'][key]
    for key in other_params_analysis_keys:
        if 'ANALYSIS' in config_run.keys() and key in config_run['ANALYSIS'].keys():
            other_params_dict[key] = config_run['ANALYSIS'][key]
        else:
            other_params_dict[key] = config_def['ANALYSIS'][key]
    other_params_dict['pressure_model_type'] = pressure_model_type
    other_params_dict['pressure_model_name'] = pressure_model_name
    other_params_dict['pressure_model_mdef'] = pressure_model_mdef
    other_params_dict['pressure_model_delta'] = float(''.join(list(pressure_model_mdef)[:-1]))

    # ,'lowbroken_powerlaw','highbroken_powerlaw','doublybroken_powerlaw'
    if other_params_dict['do_split_params_massbins'] or pressure_model_type in ['broken_powerlaw',
                                                                                'superbroken_powerlaw',
                                                                                'superPlybroken_powerlaw',
                                                                                'fullybroken_powerlaw']:
        if pressure_model_type in ['broken_powerlaw', 'superbroken_powerlaw', 'superPlybroken_powerlaw',
                                   'fullybroken_powerlaw']:
            other_params_dict['do_split_params_massbins'] = True
            other_params_dict['split_log10_mass_bins_min'] = 10.0
            other_params_dict['split_log10_mass_bins_max'] = 16.0
            other_params_dict['num_split_bins'] = 2

            if pressure_model_name == 'Arnaud10':
                if (pressure_model_type in ['broken_powerlaw', 'superbroken_powerlaw', 'superPlybroken_powerlaw',
                                            'fullybroken_powerlaw']) and (
                        'alpha_p' not in other_params_dict['split_params_massbins_and_default'].keys()):
                    other_params_dict['split_params_massbins_and_default']['alpha_p'] = 'default'

                if (pressure_model_type in ['superbroken_powerlaw', 'fullybroken_powerlaw']) and (
                        'beta' not in other_params_dict['split_params_massbins_and_default'].keys()):
                    other_params_dict['split_params_massbins_and_default']['beta'] = 'default'

                if (pressure_model_type in ['superPlybroken_powerlaw', 'fullybroken_powerlaw']) and (
                        'P0' not in other_params_dict['split_params_massbins_and_default'].keys()):
                    other_params_dict['split_params_massbins_and_default']['P0'] = 'default'

            # if pressure_model_name == 'Battaglia12':
            #     if (pressure_model_type in ['lowbroken_powerlaw','doublybroken_powerlaw']) and (
            #             'alpha_p_low' not in other_params_dict['split_params_massbins_and_default'].keys()):
            #         other_params_dict['split_params_massbins_and_default']['alpha_p_low'] = 'default'
            #
            #     if (pressure_model_type in ['highbroken_powerlaw','doublybroken_powerlaw']) and (
            #             'alpha_p_high' not in other_params_dict['split_params_massbins_and_default'].keys()):
            #         other_params_dict['split_params_massbins_and_default']['alpha_p_high'] = 'default'

        if 'split_log10_mass_bins_edges' not in other_params_dict.keys():
            split_log10_mass_bins_edges = np.linspace(other_params_dict['split_log10_mass_bins_min'],
                                                      other_params_dict['split_log10_mass_bins_max'],
                                                      other_params_dict['num_split_bins'] + 1)
            other_params_dict['split_log10_mass_bins_edges'] = split_log10_mass_bins_edges

        else:
            split_log10_mass_bins_edges = other_params_dict['split_log10_mass_bins_edges']
            other_params_dict['num_split_bins'] = len(other_params_dict['split_log10_mass_bins_edges']) - 1

        split_params_massbins_and_default = other_params_dict['split_params_massbins_and_default']
        split_params_massbins_keys = split_params_massbins_and_default.keys()
        split_params_massbins_dict = {}
        for key in split_params_massbins_keys:
            def_val = split_params_massbins_and_default[key]
            if def_val == 'default':
                split_params_massbins_dict[key] = pressure_params_dict[key] * np.ones(
                    other_params_dict['num_split_bins'])
            elif len(def_val) == 1:
                split_params_massbins_dict[key] = def_val[0] * np.ones(other_params_dict['num_split_bins'])
            elif len(def_val) == other_params_dict['num_split_bins']:
                split_params_massbins_dict[key] = def_val
            else:
                print('length of default values not equal to number of bins')
                sys.exit(1)

            if key in pressure_params_dict.keys():
                pressure_params_dict[key] = split_params_massbins_dict[key]

        other_params_dict['split_params_massbins_dict'] = split_params_massbins_dict

        split_params_massbins_names = split_params_massbins_dict.keys()

        split_mass_bins_min = 10 ** np.array(split_log10_mass_bins_edges[:-1])
        split_mass_bins_max = 10 ** np.array(split_log10_mass_bins_edges[1:])

        # how to get the bins centers? center of log bin or center of absolute bin?
        split_mass_bins_centers = 10 ** (
                0.5 * (np.array(split_log10_mass_bins_edges[:-1]) + np.array(split_log10_mass_bins_edges[1:])))
        # split_mass_bins_centers = 0.5*(split_mass_bins_min + split_mass_bins_max)

        other_params_dict['split_mass_bins_min'] = split_mass_bins_min
        other_params_dict['split_mass_bins_max'] = split_mass_bins_max
        other_params_dict['split_mass_bins_centers'] = split_mass_bins_centers
        other_params_dict['split_params_massbins_names'] = split_params_massbins_names

    # pdb.set_trace()

    fisher_params_vary_array = other_params_dict['fisher_params_vary_array']
    fisher_params_vary_array_latex = []

    if pressure_model_type in ['broken_powerlaw', 'superbroken_powerlaw', 'superPlybroken_powerlaw',
                               'fullybroken_powerlaw', 'lowbroken_powerlaw', 'highbroken_powerlaw',
                               'doublybroken_powerlaw']:
        if pressure_model_type == 'broken_powerlaw':
            if 'alpha_p' not in fisher_params_vary_array: fisher_params_vary_array.append('alpha_p')

        if pressure_model_type == 'superbroken_powerlaw':
            if 'alpha_p' not in fisher_params_vary_array: fisher_params_vary_array.append('alpha_p')
            if 'beta' not in fisher_params_vary_array: fisher_params_vary_array.append('beta')

        if pressure_model_type == 'superPlybroken_powerlaw':
            if 'alpha_p' not in fisher_params_vary_array: fisher_params_vary_array.append('alpha_p')
            if 'P0' not in fisher_params_vary_array: fisher_params_vary_array.append('P0')

        if pressure_model_type == 'fullybroken_powerlaw':
            if 'alpha_p' not in fisher_params_vary_array: fisher_params_vary_array.append('alpha_p')
            if 'P0' not in fisher_params_vary_array: fisher_params_vary_array.append('P0')
            if 'beta' not in fisher_params_vary_array: fisher_params_vary_array.append('beta')

        if pressure_model_type == 'lowbroken_powerlaw':
            if 'alpha_p_low' not in fisher_params_vary_array: fisher_params_vary_array.append('alpha_p_low')

        if pressure_model_type == 'highbroken_powerlaw':
            if 'alpha_p_high' not in fisher_params_vary_array: fisher_params_vary_array.append('alpha_p_high')

        if pressure_model_type == 'doublybroken_powerlaw':
            if 'alpha_p_low' not in fisher_params_vary_array: fisher_params_vary_array.append('alpha_p_low')
            if 'alpha_p_high' not in fisher_params_vary_array: fisher_params_vary_array.append('alpha_p_high')

    for param in fisher_params_vary_array:
        fisher_params_vary_array_latex.append(other_params_dict['latex_names'][param])
    fisher_params_vary_array_dict = {'params_to_vary': fisher_params_vary_array,
                                     'params_label': fisher_params_vary_array_latex,
                                     'priors_dict': other_params_dict['priors_dict']}

    do_vary_cosmo = False
    for params in fisher_params_vary_array:
        if params in cosmo_params_dict.keys():
            do_vary_cosmo = True

    other_params_dict['do_vary_cosmo'] = do_vary_cosmo

    if other_params_dict['do_split_params_massbins']:
        fisher_split_params_massbins_vary = other_params_dict['fisher_split_params_massbins_vary']
        fisher_split_params_massbins_vary_keys = fisher_split_params_massbins_vary.keys()

        for key in split_params_massbins_keys:
            if key not in fisher_split_params_massbins_vary_keys:
                fisher_split_params_massbins_vary[key] = np.ones(other_params_dict['num_split_bins'])
            elif len(fisher_split_params_massbins_vary[key]) == 1:
                def_val = fisher_split_params_massbins_vary[key]
                fisher_split_params_massbins_vary[key] = def_val * np.ones(other_params_dict['num_split_bins'])
            elif len(fisher_split_params_massbins_vary[key]) == other_params_dict['num_split_bins']:
                pass
            else:
                print('length of mass bin to vary array is not equal to number of bins')
                sys.exit(1)

        fisher_params_vary_array_dict['fisher_split_params_massbins_vary'] = fisher_split_params_massbins_vary

    rand_save_name = ''.join(
        random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(4))
    save_results_ini_dir = other_params_dict['save_results_ini_dir']
    try:
        os.mkdir(save_results_ini_dir)
    except:
        pass

    ini_save_name = save_results_ini_dir + 'params_' + rand_save_name + '.ini'
    output_save_name = save_results_ini_dir + 'results_' + rand_save_name + '.pkl'

    if save_params:
        with open(ini_save_name, 'w') as configfile:
            config_run.write(configfile)

    # other_params_dict['save_suffix'] = other_params_dict['save_suffix'] + '_' + rand_save_name

    other_params_dict['save_suffix'] = '_M_' + str(other_params_dict['log_M_min_tracer']) + '_' + str(
        other_params_dict['log_M_max_tracer']) + '_z_' + str(other_params_dict['zmin_tracer']) + '_' + str(
        other_params_dict['zmax_tracer']) + other_params_dict['save_suffix'] + '_' + rand_save_name

    # other_params_dict['save_suffix'] = '_params_' + '_'.join(fisher_params_vary_array) + other_params_dict['save_suffix'] + '_' + rand_save_name

    other_params_dict['output_save_name'] = output_save_name

    ini_info = {'cosmo_params_dict': cosmo_params_dict, 'pressure_params_dict': pressure_params_dict,
                'hod_params_dict': hod_params_dict, 'other_params_dict': other_params_dict,
                'fisher_params_dict': fisher_params_vary_array_dict}

    return ini_info


def run_forecast(ini_file):
    ini_info = read_ini(ini_file)

    other_params_dict = ini_info['other_params_dict']
    fisher_params_dict = ini_info['fisher_params_dict']
    cosmo_params_dict = ini_info['cosmo_params_dict']
    pressure_params_dict = ini_info['pressure_params_dict']
    hod_params_dict = ini_info['hod_params_dict']

    other_params_dict['cosmo_fid'] = cosmo_params_dict
    other_params_dict['hod_fid'] = hod_params_dict
    other_params_dict['fisher_fid'] = fisher_params_dict
    other_params_dict['pressure_fid'] = pressure_params_dict

    verbose = other_params_dict['verbose']

    if other_params_dict['do_fast']:
        if 'num_M' not in other_params_dict.keys(): other_params_dict['num_M'] = 800
        if 'num_z' not in other_params_dict.keys(): other_params_dict['num_z'] = 60
        if 'num_x' not in other_params_dict.keys(): other_params_dict['num_x'] = 2 ** 8 + 1
    else:
        if 'num_M' not in other_params_dict.keys(): other_params_dict['num_M'] = 200
        if 'num_z' not in other_params_dict.keys(): other_params_dict['num_z'] = 150
        if 'num_x' not in other_params_dict.keys(): other_params_dict['num_x'] = 2 ** 11 + 1



    other_params_dict['M_array'] = np.logspace(other_params_dict['logM_array_min'], other_params_dict['logM_array_max'],
                                               other_params_dict['num_M'])
    other_params_dict['z_array'] = np.logspace(np.log10(other_params_dict['z_array_min']),
                                               np.log10(other_params_dict['z_array_max']),
                                               other_params_dict['num_z'])
    # other_params_dict['x_array'] = np.linspace(other_params_dict['xmin'], other_params_dict['xmax'],
    #                                            other_params_dict['num_x'])
    other_params_dict['x_array'] = np.logspace(np.log10(other_params_dict['xmin']), np.log10(other_params_dict['xmax']),
                                               other_params_dict['num_x'])

    if other_params_dict['larray_spacing'] == 'log':
        l_array_all = np.logspace(np.log10(other_params_dict['lmin']), np.log10(other_params_dict['lmax']),
                                  other_params_dict['num_l'])

    if other_params_dict['larray_spacing'] == 'lin':
        l_array_all = np.linspace((other_params_dict['lmin']), (other_params_dict['lmax']),
                                  other_params_dict['num_l'])
    other_params_dict['dl_array'] = l_array_all[1:] - l_array_all[:-1]
    other_params_dict['l_array'] = (l_array_all[1:] + l_array_all[:-1])/2.

    plot_dir = other_params_dict['plot_dir']
    try:
        os.mkdir(plot_dir)
    except:
        pass

    save_suffix = other_params_dict['save_suffix']
    do_plot = other_params_dict['do_plot']

    '''
        n(z) of galaxies
    '''

    minz = other_params_dict['z_array_min']
    maxz = other_params_dict['z_array_max'] + 0.1
    numz = 1000
    zz = np.logspace(np.log10(minz), np.log10(maxz), numz)
    nz_z = 0.5 * (zz[:-1] + zz[1:])
    # pdb.set_trace()
    if hod_params_dict['hod_type'] == 'Halos':
        if verbose:
            print("getting halos nz")
        nz_g, nbar = get_nz.get_nz_halo(nz_z, cosmo_params_dict, other_params_dict)
        if verbose:
            print('nbar = ', nbar)
    elif hod_params_dict['hod_type'] == 'DESI':
        if verbose:
            print("getting desi nz")
        nz_g_perdeg2 = get_nz.get_desi_specs('bgs', zz)
        nz_g = get_nz.get_nz_normalized(nz_g_perdeg2, nz_z)
        nbar = get_nz.get_nbar(nz_g_perdeg2, nz_z)

        if verbose:
            print('nbar = ', nbar)

    elif hod_params_dict['hod_type'] == '2MRS':
        if verbose:
            print("getting 2MRS nz")
        nz_g = get_nz.get_nz_g_2mrs(nz_z, 1.31, 1.64, 0.0266)
        Ng_total = 43182
        fsky_gg = other_params_dict['fsky_gg']
        nbar = Ng_total / (fsky_gg * 4 * np.pi)
        if verbose:
            print('nbar = ', nbar)

    # this doesn't seem like the right place for this....
    other_params_dict['ng_zarray'] = nz_z
    other_params_dict['ng_value'] = nz_g  # super confusing renaming
    other_params_dict['nbar'] = nbar

    if do_plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(nz_z, nz_g)
        ax.set_xlabel('z')
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_ylabel('n(z)')
        fig.savefig(plot_dir + 'redshift_distribution' + save_suffix + '.png')

    '''
        Run Fisher matrix computation
    '''
    # Object that will compute fisher matrix
    fisher_obj = Fisher(cosmo_params_dict, hod_params_dict, pressure_params_dict, other_params_dict,
                        fisher_params_dict)

    # Run the Fisher matrix computation
    fisher_obj.get_fisher_mat()
    fisher_obj.get_fid_values()

    with open(other_params_dict['output_save_name'], 'wb') as f:
        dill.dump(fisher_obj, f)

    # do_plot = True
    if do_plot:
        fplot_obj = fplot.FisherPlotter(fisher_obj)

        save_name = other_params_dict['plot_dir'] + 'contours' + other_params_dict['save_suffix'] + '.png'
        fig2 = fplot_obj.plot_contours(save_name=save_name)

        Cls_to_plot = 'yg'
        fig4, fig5 = fplot_obj.plot_Cly_relation(nsample=20,do_plot=True,percentiles=[16., 84.], do_samples=True,
                                                           do_plot_relative=True, do_multiprocess=True, num_pool=5, Cls_to_plot=Cls_to_plot)
        fig4.savefig(other_params_dict['plot_dir'] + 'Cl_' + Cls_to_plot + '_werr_' + other_params_dict['save_suffix'] + '.png')
        fig5.savefig(other_params_dict['plot_dir'] + 'chi2_dist_' + other_params_dict['save_suffix'] + '.png')

        # fig4, fig5 = fplot_obj.plot_w_yg_rp_w_errorbars_relation(rp_min=0.01, rp_max=6.0, nsample=450, n_rp=500, do_plot=True,
        #                                                    percentiles=[16., 84.], do_samples=False,
        #                                                    do_plot_relative=True, do_multiprocess=True, num_pool=40)
        # fig4.savefig(other_params_dict['plot_dir'] + 'xi_yg_werr_' + other_params_dict['save_suffix'] + '.png')

        fig = fplot_obj.plot_YM_relation_relative(M_min=10 ** other_params_dict['logM_array_min'],
                                                  M_max=10 ** other_params_dict['logM_array_max'], numM=400,nsample=30,
                                                  do_samples=True,
                                                  labels=(r'Halos', r'$Y_{500, \rm fid} \propto M^{1.72}$'), mdef_Mmin_Mmax=other_params_dict['mdef_analysis'])
        fig.savefig(other_params_dict['plot_dir'] + 'YM_relation' + other_params_dict['save_suffix'] + '.png')

        # fig3 = fplot_obj.plot_Pr_relation(min_x=1e-3, max_x=6, numx=100, nsample=450, do_samples=False,
        #                                   percentiles=[16., 84.])
        # fig3.savefig(other_params_dict['plot_dir'] + 'Pr_3d_' + other_params_dict['save_suffix'] + '.png')




if __name__ == "__main__":
    ini_file_name = sys.argv[1]
    run_forecast(ini_file_name)
