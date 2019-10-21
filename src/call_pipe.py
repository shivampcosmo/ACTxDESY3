import sys, os
import pdb
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
sys.path.insert(0, path + '/../helper/')
import plot_funcs as pf
import numpy as np
import copy

from cross_corr_funcs import Pressure, general_hm, DataVec
import colossus
from colossus.cosmology import cosmology
from colossus.lss import bias
from colossus.lss import mass_function
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import concentration

import LSS_funcs as hmf
import astropy.units as u
from astropy import constants as const
import scipy as sp
from scipy import interpolate
import multiprocessing


class FisherPlotter:
    def __init__(self, fisher_obj, plot_dir='./', save_suffix='', save_plots=False):
        print "initializing fisher plotter"
        self.F_mat = fisher_obj.F_mat
        # pdb.set_trace()
        # self.fid_values = fisher_obj.get_fid_values()
        self.fid_values = fisher_obj.fid_values
        self.fisher_params_label = fisher_obj.fisher_params_label
        if fisher_obj.do_split_params_massbins:
            self.mass_bin_split = fisher_obj.split_mass_bins_min
        else:
            self.mass_bin_split = None

        self.Cl_fid_dict = fisher_obj.Cl_all_dict

        if hasattr(fisher_obj, 'zmean'): self.zmean = fisher_obj.zmean
        if hasattr(fisher_obj, 'chi_val'): self.chi_val = fisher_obj.chi_val
        if hasattr(fisher_obj, 'cov_G_fid_theta_dict'): self.cov_G_fid_theta_dict = fisher_obj.cov_G_fid_theta_dict
        if hasattr(fisher_obj, 'rp_array'): self.rp_array = fisher_obj.rp_array
        if hasattr(fisher_obj, 'wtheta_yg'): self.wtheta_yg = fisher_obj.wtheta_yg
        if hasattr(fisher_obj, 'sigG_gy_array'): self.sigG_gy_array = fisher_obj.sigG_gy_array

        self.dCl_dparam_dict = fisher_obj.dCl_dparam_dict

        self.cosmo_params_fid = fisher_obj.cosmo_params_fid
        self.pressure_params_fid = fisher_obj.pressure_params_fid
        self.other_params_fid = fisher_obj.other_params_fid
        self.hod_params_fid = fisher_obj.hod_params_fid
        self.fisher_params_vary = fisher_obj.fisher_params_vary
        self.fisher_params_vary_orig = fisher_obj.fisher_params_vary_orig
        self.fisher_params_label = fisher_obj.fisher_params_label

        cosmology.addCosmology('mock_cosmo', self.cosmo_params_fid)
        cosmo_colossus = cosmology.setCosmology('mock_cosmo')

        # self.cov_fid = fisher_obj.cov_fid
        # self.inv_cov_fid = fisher_obj.inv_cov_fid

        self.priors_array = np.zeros(self.F_mat.shape[0])
        if hasattr(fisher_obj, 'priors_dict'):
            self.priors_dict = fisher_obj.priors_dict
            for j in range(len(self.fisher_params_vary)):
                param_vary = self.fisher_params_vary[j]
                self.priors_array[j] = self.priors_dict[param_vary]
        else:
            self.priors_dict = {}
            for j in range(len(self.fisher_params_vary)):
                param_vary = self.fisher_params_vary[j]
                self.priors_dict[param_vary] = 1e20
                self.priors_array[j] = 1e20

        self.F_mat_wpriors = self.F_mat + np.diag(1. / self.priors_array ** 2.)

        self.cov_fid = fisher_obj.cov_fid
        self.inv_cov_fid = fisher_obj.inv_cov_fid
        self.Cl_fid = fisher_obj.Cl_fid
        self.l_array_survey = fisher_obj.l_array_survey

        self.save_suffix = save_suffix
        self.plot_dir = plot_dir

        ng_zarray = self.other_params_fid['ng_zarray']
        ng_value = self.other_params_fid['ng_value']
        self.ng_interp = interpolate.interp1d(ng_zarray, np.log(ng_value + 1e-40), fill_value='extrapolate')

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

                if self.other_params_fid['do_split_params_massbins']:
                    if param_vary not in self.fisher_params_vary_orig:
                        mass_bin_number = int(list(param_vary)[-1])
                        params_new_param_vary = copy.deepcopy(pressure_params[param_vary_orig])
                        params_new_param_vary[mass_bin_number] = new_values[fi]
                        pressure_params[param_vary_orig] = params_new_param_vary
                    else:
                        pressure_params[param_vary_orig] = new_values[fi]
                else:
                    pressure_params[param_vary_orig] = new_values[fi]

        return cosmo_params, hod_params, pressure_params

    def set_values_partial(self, cosmo_params_orig, hod_params_orig, pressure_params_orig, new_values_all):

        cosmo_params_array, hod_params_array, pressure_params_array = [], [], []

        for j in range(len(new_values_all)):
            cosmo_params = copy.deepcopy(cosmo_params_orig)
            hod_params = copy.deepcopy(hod_params_orig)
            pressure_params = copy.deepcopy(pressure_params_orig)
            new_values = copy.deepcopy(self.fid_values)
            new_values[j] = new_values_all[j]
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

                    if self.other_params_fid['do_split_params_massbins']:
                        if param_vary not in self.fisher_params_vary_orig:
                            mass_bin_number = int(list(param_vary)[-1])
                            params_new_param_vary = copy.deepcopy(pressure_params[param_vary_orig])
                            params_new_param_vary[mass_bin_number] = new_values[fi]
                            pressure_params[param_vary_orig] = params_new_param_vary
                        else:
                            pressure_params[param_vary_orig] = new_values[fi]
                    else:
                        pressure_params[param_vary_orig] = new_values[fi]

            cosmo_params_array.append(cosmo_params)
            hod_params_array.append(hod_params)
            pressure_params_array.append(pressure_params)

        # pdb.set_trace()

        return cosmo_params_array, hod_params_array, pressure_params_array

    def plot_contours(self, parameter_indices=None, save_name=None, save_dir='./'):
        sigma_levels = np.array([1.])
        fig = pf.plot_contours(self.F_mat_wpriors, self.fid_values, self.fisher_params_label, sigma_levels,
                               save_name=save_name, parameter_indices=parameter_indices)
        return fig

    def get_chi2_Cl(self, Cl_vec):
        diff = np.array([Cl_vec - self.Cl_fid])
        val = np.dot(np.dot(diff, self.inv_cov_fid), diff.T)
        return val

    def get_chi2_wtheta(self, w_yg_params_dict, w_yg_data_dict):
        rp_params, wthetayg_mat, wthetayg_fid = w_yg_params_dict['x'], w_yg_params_dict['curves'], w_yg_params_dict[
            'fid']
        rp_data, wthetayg_data, sig_yg_data = w_yg_data_dict['x'], w_yg_data_dict['y'], w_yg_data_dict['sig']

        cov = np.diag(sig_yg_data ** 2)
        inv_cov = np.linalg.inv(cov)
        nsamples, numx = wthetayg_mat.shape
        chi2_val = np.zeros(nsamples)
        for j in range(nsamples):
            wthetayg_mat_interp = interpolate.interp1d(np.log(rp_params), (wthetayg_mat[j, :]),
                                                       fill_value='extrapolate')
            wthetayg_mat_rpdata = (wthetayg_mat_interp(np.log(rp_data)))

            diff = np.array([wthetayg_data - wthetayg_mat_rpdata])
            chi2_val[j] = np.dot(np.dot(diff, inv_cov), diff.T)
        return chi2_val

    def get_Pr_fid(self, min_x, max_x, M, z, numx=None):
        x_array = np.linspace(min_x, max_x, numx)
        pressure = Pressure(self.cosmo_params_fid, self.pressure_params_fid, self.other_params_fid)
        rhoDelta = pressure.pressure_model_delta * pressure.cosmo_colossus.rho_c(z) * (1000 ** 3)
        R_mat = np.array([[np.power(3 * M / (4 * np.pi * rhoDelta), 1. / 3.)]])
        Pr_mat = pressure.get_Pe_mat(np.array([[M]]), x_array, np.array([z]), R_mat)
        return x_array, Pr_mat

    def get_Pr_mat(self, min_x, max_x, M, z, cosmo_params_array, pressure_params_array, other_params_array, numx=None):
        x_array = np.linspace(min_x, max_x, numx)
        Pr_mat = np.zeros((len(pressure_params_array), numx))

        print "starting Y(M) samples"
        nprint = np.floor(len(pressure_params_array) / 10.)

        for pi in xrange(0, len(pressure_params_array)):
            if float(pi) / nprint - int(pi / nprint) == 0:
                print "si = ", pi, " out of ", len(pressure_params_array)

            pressure = Pressure(cosmo_params_array[pi], pressure_params_array[pi], other_params_array[pi])
            rhoDelta = pressure.pressure_model_delta * pressure.cosmo_colossus.rho_c(z) * (1000 ** 3)
            R_mat = np.array([[np.power(3 * M / (4 * np.pi * rhoDelta), 1. / 3.)]])
            Pr_mat[pi, :] = pressure.get_Pe_mat(np.array([[M]]), x_array, np.array([z]), R_mat)

        # pf.plot_Pr_dep(x_array, Pr_mat, pressure_params_array, self.fisher_params_vary, self.fisher_params_vary_orig,
        #                self.fisher_params_label, self.other_params_fid['plot_dir'],
        #                self.other_params_fid['save_suffix'])

        return x_array, Pr_mat

    def get_Pr_mat_from_Fisher(self, min_x, max_x, M, z, numx=None, nsample=None):
        # generate parameter sets from Fisher
        nsamps = nsample
        pressure_params_array = []
        cosmo_params_array = []
        other_params_array = []
        nprint = np.floor(nsamps / 10.)
        for si in xrange(0, nsamps):
            pressure_params_si, cosmo_params_si, hod_params_si, other_params_si, _ = self.generate_rand_params_from_Fisher()
            pressure_params_array.append(pressure_params_si)
            cosmo_params_array.append(cosmo_params_si)
            other_params_array.append(other_params_si)

        print "getting Pressure profile"
        x_array, Pr_mat = self.get_Pr_mat(min_x, max_x, M, z, cosmo_params_array, pressure_params_array,
                                          other_params_array, numx=numx)

        return x_array, Pr_mat

    def get_wthetay_mat(self, theta_min, theta_max, M_min, M_max, z_min, z_max, cosmo_params_array,
                        pressure_params_array, other_params_array, num_M=None, num_z=None, num_theta=None, x_min=None,
                        x_max=None, num_x=None):

        nsamps = len(pressure_params_array)
        nprint = np.floor(nsamps / 10.)

        M_array = np.logspace(np.log10(M_min), np.log10(M_max), num_M)
        z_array = np.linspace(z_min, z_max, num_z)
        x_array = np.logspace(np.log10(x_min), np.log10(x_max), num_x)

        theta_array_all = np.logspace(np.log10(theta_min), np.log10(theta_max), ntheta)
        theta_array = (theta_array_all[1:] + theta_array_all[:-1]) / 2.
        theta_array_rad = theta_array * (np.pi / 180.) * (1. / 60.)

        nm, nz = len(M_array), len(z_array)
        M_mat = np.tile(M_array.reshape(1, nm), (nz, 1))
        ghmf = general_hm(cosmo_params_array[0], pressure_params_array[0], other_params_array[0])

        dndm_array_Mz = np.zeros(M_mat.shape)
        for j in range(len(z_array)):
            M_array = M_mat[j, :]
            dndm_array_Mz[j, :] = (1. / M_array) * mass_function.massFunction(M_array, z_array[j],
                                                                              mdef=other_params_array[0][
                                                                                  'pressure_model_mdef'],
                                                                              model=other_params_array[0]['dndm_model'],
                                                                              q_out='dndlnM')

        Delta = other_params_array[0]['pressure_model_delta']
        rhoDelta = Delta * ghmf.cosmo_colossus.rho_c(z_array) * (1000 ** 3)
        rhoDelta_mat = np.tile(rhoDelta.reshape(nz, 1), (1, nm))
        R_mat = np.array(np.power(3 * M_mat / (4 * np.pi * rhoDelta_mat), 1. / 3.))

        y3D_mat = np.zeros((len(pressure_params_array), nz, nm, num_x))

        wthetayg_np_ntheta = np.zeros((len(pressure_params_array), num_theta))

        dndm_mat = np.tile(dndm_array_Mz.reshape(1, nz, nm), (num_theta, 1, 1))

        chi_array = hmf.get_Dcom_array(z_array, cosmo_params_array[0]['Om0'])
        dchi_dz_array = (const.c.to(u.km / u.s)).value / (hmf.get_Hz(z_array, cosmo_params_array[0]['Om0']))
        chi_mat = np.tile(chi_array.reshape(1, nz), (num_theta, 1))
        dchi_dz_mat = np.tile(dchi_dz_array.reshape(1, nz), (num_theta, 1))

        for pi in xrange(0, len(pressure_params_array)):

            wthetayg_nm_nz_ntheta = np.zeros((num_theta, nz, nm))

            # if float(pi) / nprint - int(pi / nprint) == 0:
            #     print "si = ", pi, " out of ", len(pressure_params_array)

            pressure = Pressure(cosmo_params_array[pi], pressure_params_array[pi], other_params_array[pi])

            y3D_mat[pi, :, :, :] = pressure.get_y3d(M_mat, x_array, z_array, R_mat)

            for j1 in range(num_theta):
                for j2 in range(nz):
                    for j3 in range(nm):
                        rp_r500 = theta_array_rad[j1] * (
                            hmf.get_Dang_com(z_array[j2], cosmo_params_array[pi]['Om0'])) / (R_mat[j2, j3])

                        # pdb.set_trace()
                        wthetayg_nm_nz_ntheta[j1, j2, j3] = (R_mat[j2, j3]) * hmf.project_corr(rp_r500, x_array,
                                                                                               y3D_mat[pi, j2, j3, :])

            toint_M = wthetayg_nm_nz_ntheta * dndm_mat
            val_z = (sp.integrate.simps(toint_M, M_array)) / (sp.integrate.simps(dndm_mat, M_array))
            toint_z = val_z * (chi_mat ** 2) * dchi_dz_mat * (1. / z_array)
            val = (sp.integrate.simps(toint_z, z_array)) / (
                sp.integrate.simps((chi_mat ** 2) * dchi_dz_mat, z_array))

            # pdb.set_trace()

            wthetayg_np_ntheta[pi, :] = val

        return theta_array_rad, wthetayg_np_ntheta

    def get_wtheta(self, theta_array_rad, l_array, Cl_dict):

        l_array_full = np.logspace(0, 4.1, 14000)
        Cl_yg_1h_array = Cl_dict['yg']['1h']
        Cl_yg_2h_array = Cl_dict['yg']['2h']
        Cl_yg_total_array = Cl_dict['yg']['total']

        # Cl_yg_1h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yg_1h_array), fill_value='extrapolate')
        # Cl_yg_1h_full = np.exp(Cl_yg_1h_interp(np.log(l_array_full)))
        # Cl_yg_2h_interp = interpolate.interp1d(np.log(l_array), np.log(Cl_yg_2h_array), fill_value='extrapolate')
        # Cl_yg_2h_full = np.exp(Cl_yg_2h_interp(np.log(l_array_full)))

        Cl_yg_1h_interp = interpolate.interp1d(np.log(l_array), Cl_yg_1h_array, fill_value=0.0, bounds_error=False)
        Cl_yg_1h_full = Cl_yg_1h_interp(np.log(l_array_full))
        Cl_yg_2h_interp = interpolate.interp1d(np.log(l_array), Cl_yg_2h_array, fill_value=0.0, bounds_error=False)
        Cl_yg_2h_full = (Cl_yg_2h_interp(np.log(l_array_full)))

        wtheta_yg_1h = np.zeros(len(theta_array_rad))
        wtheta_yg_2h = np.zeros(len(theta_array_rad))
        wtheta_yg = np.zeros(len(theta_array_rad))
        for j in range(len(theta_array_rad)):
            wtheta_yg_1h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yg_1h_full)
            wtheta_yg_2h[j] = hmf.get_wprp(theta_array_rad[j], l_array_full, Cl_yg_2h_full)
            wtheta_yg[j] = wtheta_yg_1h[j] + wtheta_yg_2h[j]

        return wtheta_yg_1h, wtheta_yg_2h, wtheta_yg

    def collect_wthetayg(self, j_array, return_dict, cosmo_params_array, pressure_params_array, hod_params_array,
                         other_params_array, theta_array_rad):
        for pi in j_array:
            cosmo_params, pressure_params, hod_params, other_params = copy.deepcopy(
                cosmo_params_array[pi]), copy.deepcopy(pressure_params_array[pi]), copy.deepcopy(
                hod_params_array[pi]), copy.deepcopy(other_params_array[pi])

            other_params['noise_Cl_filename'] = None
            other_params['do_plot'] = False
            other_params['verbose'] = False
            DV = DataVec(cosmo_params, hod_params, pressure_params, other_params)
            Cl_dict = DV.Cl_dict
            wtheta_yg_1h, wtheta_yg_2h, wtheta_yg_np = self.get_wtheta(theta_array_rad, other_params['l_array'],
                                                                       Cl_dict)
            return_dict[pi] = (wtheta_yg_1h, wtheta_yg_2h, wtheta_yg_np)

    def collect_Clyg(self, j_array, return_dict, cosmo_params_array, pressure_params_array, hod_params_array,
                     other_params_array):
        for pi in j_array:
            cosmo_params, pressure_params, hod_params, other_params = copy.deepcopy(
                cosmo_params_array[pi]), copy.deepcopy(pressure_params_array[pi]), copy.deepcopy(
                hod_params_array[pi]), copy.deepcopy(other_params_array[pi])

            other_params['noise_Cl_filename'] = None
            other_params['do_plot'] = False
            other_params['verbose'] = False
            DV = DataVec(cosmo_params, hod_params, pressure_params, other_params)
            Cl_dict = DV.Cl_dict

            Cl_vec = DV.get_Cl_vector()

            return_dict[pi] = {'Cl_dict': Cl_dict, 'chi2': self.get_chi2_Cl(Cl_vec)}

    def collect_YM(self, j_array, return_dict, cosmo_params_array, pressure_params_array, other_params_array,
                   M_array_500c, M_array_200c, z):

        numM = len(M_array_500c)
        for pi in j_array:
            cosmo_params, pressure_params, other_params = copy.deepcopy(
                cosmo_params_array[pi]), copy.deepcopy(pressure_params_array[pi]), copy.deepcopy(other_params_array[pi])

            other_params['noise_Cl_filename'] = None
            other_params['do_plot'] = False
            other_params['verbose'] = False

            pressure = Pressure(cosmo_params, pressure_params, other_params)

            integratedY_mat = np.zeros(numM)
            for mi in xrange(0, numM):
                integratedY = pressure.get_Y500sph_singleMz(M_array_500c[mi], z, do_fast=False, M200c=M_array_200c[mi])
                integratedY_mat[mi] = integratedY

            return_dict[pi] = integratedY_mat

    def get_wthetay_mat_Makiya18(self, theta_min, theta_max, cosmo_params_array, pressure_params_array,
                                 hod_params_array, other_params_array, num_theta=None, do_multiprocess=False,
                                 num_pool=None):

        nsamps = len(cosmo_params_array)
        nprint = np.floor(nsamps / 10.)

        theta_array_all = np.logspace(np.log10(theta_min), np.log10(theta_max), num_theta)
        theta_array = (theta_array_all[1:] + theta_array_all[:-1]) / 2.
        theta_array_rad = theta_array * (np.pi / 180.) * (1. / 60.)
        ntheta = len(theta_array_rad)

        cosmo_params, pressure_params, hod_params, other_params = copy.deepcopy(self.cosmo_params_fid), copy.deepcopy(
            self.pressure_params_fid), copy.deepcopy(self.hod_params_fid), copy.deepcopy(self.other_params_fid)

        wtheta_yg_1h_fid, wtheta_yg_2h_fid, wtheta_yg_fid = self.get_wtheta(theta_array_rad, other_params['l_array'],
                                                                            self.Cl_fid_dict)

        wtheta_yg_np_ntheta = np.zeros((len(pressure_params_array), ntheta))
        wtheta_yg_1h_np_ntheta = np.zeros((len(pressure_params_array), ntheta))
        wtheta_yg_2h_np_ntheta = np.zeros((len(pressure_params_array), ntheta))

        if do_multiprocess:
            manager = multiprocessing.Manager()
            wthetayg_dict = manager.dict()

            processes = []
            if num_pool is None:
                for pi in xrange(0, len(pressure_params_array)):
                    p = multiprocessing.Process(target=self.collect_wthetayg, args=(
                        [pi], wthetayg_dict, cosmo_params_array, pressure_params_array, hod_params_array,
                        other_params_array, theta_array_rad))
                    processes.append(p)
                    p.start()
            else:
                npool = num_pool
                pi_array = np.arange(len(pressure_params_array))
                pi_array_split = np.array_split(pi_array, npool)
                for j in range(npool):
                    p = multiprocessing.Process(target=self.collect_wthetayg, args=(
                        pi_array_split[j], wthetayg_dict, cosmo_params_array, pressure_params_array, hod_params_array,
                        other_params_array, theta_array_rad))
                    processes.append(p)
                    p.start()

            for process in processes:
                process.join()

            for pi in range(len(pressure_params_array)):
                wtheta_yg_1h_np_ntheta[pi, :], wtheta_yg_2h_np_ntheta[pi, :], wtheta_yg_np_ntheta[pi, :] = \
                    wthetayg_dict[pi]

        else:
            for pi in xrange(0, len(pressure_params_array)):

                if float(pi) / nprint - int(pi / nprint) == 0:
                    print "si = ", pi, " out of ", nsamps

                cosmo_params, pressure_params, hod_params, other_params = copy.deepcopy(
                    cosmo_params_array[pi]), copy.deepcopy(pressure_params_array[pi]), copy.deepcopy(
                    hod_params_array[pi]), copy.deepcopy(other_params_array[pi])

                other_params['noise_Cl_filename'] = None
                other_params['do_plot'] = False
                other_params['verbose'] = False
                DV = DataVec(cosmo_params, hod_params, pressure_params, other_params)
                Cl_dict = DV.Cl_dict
                wtheta_yg_1h_np_ntheta[pi, :], wtheta_yg_2h_np_ntheta[pi, :], wtheta_yg_np_ntheta[pi,
                                                                              :] = self.get_wtheta(theta_array_rad,
                                                                                                   other_params[
                                                                                                       'l_array'],
                                                                                                   Cl_dict)

        return theta_array_rad, wtheta_yg_fid, wtheta_yg_np_ntheta

    def get_Clyg_mat_Makiya18(self, cosmo_params_array, pressure_params_array, hod_params_array, other_params_array,
                              do_multiprocess=False, num_pool=None):

        cosmo_params, pressure_params, hod_params, other_params = copy.deepcopy(self.cosmo_params_fid), copy.deepcopy(
            self.pressure_params_fid), copy.deepcopy(self.hod_params_fid), copy.deepcopy(self.other_params_fid)

        nsamps = len(cosmo_params_array)
        nprint = np.floor(nsamps / 10.)

        l_array = other_params['l_array']

        Cl_dicts_samples = []
        chi2_samples = []

        if do_multiprocess:
            manager = multiprocessing.Manager()
            Clyg_dict = manager.dict()

            processes = []
            if num_pool is None:
                for pi in xrange(0, len(pressure_params_array)):
                    p = multiprocessing.Process(target=self.collect_Clyg, args=(
                        [pi], Clyg_dict, cosmo_params_array, pressure_params_array, hod_params_array,
                        other_params_array))
                    processes.append(p)
                    p.start()
            else:
                npool = num_pool
                pi_array = np.arange(len(pressure_params_array))
                pi_array_split = np.array_split(pi_array, npool)
                for j in range(npool):
                    p = multiprocessing.Process(target=self.collect_Clyg, args=(
                        pi_array_split[j], Clyg_dict, cosmo_params_array, pressure_params_array, hod_params_array,
                        other_params_array))
                    processes.append(p)
                    p.start()

            for process in processes:
                process.join()

            for pi in range(len(pressure_params_array)):
                Cl_dicts_samples.append(Clyg_dict[pi]['Cl_dict'])
                chi2_samples.append(Clyg_dict[pi]['chi2'])

        else:

            for pi in xrange(0, len(pressure_params_array)):
                # if float(pi) / nprint - int(pi / nprint) == 0:
                #     print "si = ", pi, " out of ", nsamps

                cosmo_params, pressure_params, hod_params, other_params = copy.deepcopy(
                    cosmo_params_array[pi]), copy.deepcopy(pressure_params_array[pi]), copy.deepcopy(
                    hod_params_array[pi]), copy.deepcopy(other_params_array[pi])

                other_params['noise_Cl_filename'] = None
                other_params['do_plot'] = False
                other_params['verbose'] = False
                DV = DataVec(cosmo_params, hod_params, pressure_params, other_params)
                Cl_dict = DV.Cl_dict
                # Cl_yg_1h_np_nl[pi,:] = Cl_dict['yg']['1h']
                # Cl_yg_2h_np_nl[pi,:] = Cl_dict['yg']['2h']
                # Cl_yg_total_np_nl[pi,:] = Cl_dict['yg']['total']
                Cl_dicts_samples.append(Cl_dict)

                Cl_vec = DV.get_Cl_vector()

                chi2_samples.append(self.get_chi2_Cl(Cl_vec))

                # pdb.set_trace()

        chi2_samples = np.array(chi2_samples)

        return l_array, self.Cl_fid_dict, Cl_dicts_samples, self.cov_fid, self.inv_cov_fid, chi2_samples

    def get_wthetay_mat_from_Fisher(self, theta_min_arcmin=None, theta_max_arcmin=None, num_theta=None, nsample=None,
                                    use_only_pressure=None, do_multiprocess=False, num_pool=None):
        # generate parameter sets from Fisher
        nsamps = nsample
        nprint = np.floor(nsamps / 10.)
        pressure_params_array = []
        hod_params_array = []
        cosmo_params_array = []
        other_params_array = []
        chi2_params_array = []

        theta_min = theta_min_arcmin
        theta_max = theta_max_arcmin

        for si in xrange(0, nsamps):
            if use_only_pressure:
                pressure_params_si, cosmo_params_si, hod_params_si, other_params_si = self.generate_rand_params_from_Fisher_onlyPressure()
            else:
                pressure_params_si, cosmo_params_si, hod_params_si, other_params_si, chi2_param = self.generate_rand_params_from_Fisher()
            pressure_params_array.append(pressure_params_si)
            cosmo_params_array.append(cosmo_params_si)
            other_params_array.append(other_params_si)
            hod_params_array.append(hod_params_si)
            chi2_params_array.append(chi2_param)

        chi2_params_array = np.array(chi2_params_array)
        print "getting Pressure profile"
        theta_array, wtheta_yg_fid, wthetayg_np_ntheta = self.get_wthetay_mat_Makiya18(theta_min, theta_max,
                                                                                       cosmo_params_array,
                                                                                       pressure_params_array,
                                                                                       hod_params_array,
                                                                                       other_params_array,
                                                                                       num_theta=num_theta,
                                                                                       do_multiprocess=do_multiprocess,
                                                                                       num_pool=num_pool)

        return theta_array, wtheta_yg_fid, wthetayg_np_ntheta, chi2_params_array

    def get_Cly_mat_from_Fisher(self, nsample=None, use_only_pressure=None, do_multiprocess=False, num_pool=None):
        # generate parameter sets from Fisher
        nsamps = nsample
        nprint = np.floor(nsamps / 10.)
        pressure_params_array = []
        hod_params_array = []
        cosmo_params_array = []
        other_params_array = []
        chi2_params_array = []

        for si in xrange(0, nsamps):
            if use_only_pressure:
                pressure_params_si, cosmo_params_si, hod_params_si, other_params_si = self.generate_rand_params_from_Fisher_onlyPressure()
            else:
                pressure_params_si, cosmo_params_si, hod_params_si, other_params_si, chi2_param = self.generate_rand_params_from_Fisher()
            pressure_params_array.append(pressure_params_si)
            cosmo_params_array.append(cosmo_params_si)
            other_params_array.append(other_params_si)
            hod_params_array.append(hod_params_si)
            chi2_params_array.append(chi2_param)

        chi2_params_array = np.array(chi2_params_array)
        print "getting Pressure profile"
        l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples = self.get_Clyg_mat_Makiya18(
            cosmo_params_array, pressure_params_array, hod_params_array, other_params_array,
            do_multiprocess=do_multiprocess, num_pool=num_pool)

        return l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples, chi2_params_array

    def plot_wthetay_relation(self, theta_min_arcmin=0.1, theta_max_arcmin=100, nsample=100, num_theta=20,
                              use_only_pressure=True, do_plot=True, percentiles=None, do_samples=None, xlim=None,
                              ylim=None, do_multiprocess=False, num_pool=None):

        theta_array, wtheta_yg_fid, wthetayg_np_ntheta, chi2_params_array = self.get_wthetay_mat_from_Fisher(
            theta_min_arcmin=theta_min_arcmin, theta_max_arcmin=theta_max_arcmin, nsample=nsample, num_theta=num_theta,
            use_only_pressure=use_only_pressure, do_multiprocess=do_multiprocess, num_pool=num_pool)

        theta_array_arcmin = theta_array * (180. / np.pi) * 60.0

        if do_plot:
            print "Plotting"
            fig = pf.plot_wthetay_samples(theta_array_arcmin, wthetayg_np_ntheta, wtheta_yg_fid,
                                          plot_dir=self.other_params_fid['plot_dir'],
                                          percentiles=percentiles, do_samples=do_samples, xlim=xlim, ylim=ylim)
            return fig
        else:
            return theta_array_arcmin, wthetayg_np_ntheta, wtheta_yg_fid, chi2_params_array

    def plot_w_yg_rp_w_errorbars_relation(self, rp_min=0.08, rp_max=8.0, nsample=100, n_rp=15, z_eval=None,
                                          do_plot=True, percentiles=None, do_samples=None, xlim=None, ylim=None,
                                          do_plot_relative=False, use_only_pressure=False, do_multiprocess=False,
                                          num_pool=None):

        if z_eval is None:
            # z_eval = self.zmean
            chi_val = self.chi_val
        else:
            chi_val = hmf.get_Dcom(z_eval, self.cosmo_params_fid['Om0'])

        rp_array_param = np.logspace(np.log10(rp_min), np.log10(rp_max), n_rp)
        theta_array_rad = rp_array_param / chi_val
        theta_array_arcmin = theta_array_rad * (180. / np.pi) * 60.0
        rp_theta_interp = interpolate.interp1d(np.log(theta_array_arcmin), np.log(rp_array_param),
                                               fill_value='extrapolate')

        theta_min_arcmin = theta_array_arcmin[0]
        theta_max_arcmin = theta_array_arcmin[-1]

        theta_array_arcmin, wthetayg_np_ntheta, wtheta_yg_fid, chi2_params_array = self.plot_wthetay_relation(
            theta_min_arcmin=theta_min_arcmin, theta_max_arcmin=theta_max_arcmin, nsample=nsample, num_theta=n_rp,
            do_plot=False, use_only_pressure=use_only_pressure, do_multiprocess=do_multiprocess, num_pool=num_pool)

        rp_final = np.exp(rp_theta_interp(np.log(theta_array_arcmin)))

        w_yg_params_dict = {'x': rp_final, 'fid': wtheta_yg_fid, 'curves': wthetayg_np_ntheta}
        w_yg_data_dict = {'x': self.rp_array, 'y': self.wtheta_yg, 'sig': self.sigG_gy_array}

        chi2_val = self.get_chi2_wtheta(w_yg_params_dict, w_yg_data_dict)
        w_yg_params_dict['chi2_sample'] = chi2_val
        w_yg_params_dict['chi2_params'] = chi2_params_array
        # pdb.set_trace()

        if do_plot:

            if do_plot_relative:
                print "Plotting"
                fig1 = pf.plot_w_yg_rp_samples_w_errorbars_relative(w_yg_params_dict, w_yg_data_dict,
                                                                    plot_dir=self.other_params_fid['plot_dir'],
                                                                    percentiles=percentiles, do_samples=do_samples,
                                                                    xlim=xlim, ylim=ylim)
            else:
                print "Plotting"
                fig1 = pf.plot_w_yg_rp_samples_w_errorbars(w_yg_params_dict, w_yg_data_dict,
                                                           plot_dir=self.other_params_fid['plot_dir'],
                                                           percentiles=percentiles, do_samples=do_samples, xlim=xlim,
                                                           ylim=ylim)

            fig2 = pf.plot_chi2_samples(chi2_val, chi2_params_array)

            return fig1, fig2
        else:
            return w_yg_params_dict, w_yg_data_dict

    def plot_Cly_relation(self, nsample=100, do_plot=True, percentiles=None, do_samples=None, xlim=None, ylim=None,
                          use_only_pressure=False, do_plot_relative=False, do_multiprocess=False, num_pool=None,
                          weights=None,
                          weight_Cls=False, Cls_to_plot='yg'):

        l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples, chi2_params_array = self.get_Cly_mat_from_Fisher(
            nsample=nsample, use_only_pressure=use_only_pressure, do_multiprocess=do_multiprocess, num_pool=num_pool)

        chi2_samples = chi2_samples[:, 0, 0]
        chi2_params_array = chi2_params_array[:, 0, 0]

        if do_plot:
            print "Plotting"
            fig1 = pf.plot_Clyg_samples(l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, self.l_array_survey,
                                        plot_dir=self.other_params_fid['plot_dir'],
                                        percentiles=percentiles, do_samples=do_samples, xlim=xlim, ylim=ylim,
                                        do_plot_relative=do_plot_relative, weights=weights, weight_Cls=weight_Cls,
                                        Cls_to_plot=Cls_to_plot)

            fig2 = pf.plot_chi2_samples(chi2_samples, chi2_params_array)

            return fig1, fig2
        else:
            return l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples, chi2_params_array

    def plot_Pr_relation(self, z=0.15, M=1e13, numx=100, nsample=2000, min_x=1e-3, max_x=6, xlim=None, do_samples=True,
                         percentiles=None, ylim=None, do_plot=True):
        x_array, Pr_mat = self.get_Pr_mat_from_Fisher(min_x, max_x, M, z, numx=numx, nsample=nsample)

        # Get fiducial relation
        x_array, Pr_fid = self.get_Pr_fid(min_x, max_x, M, z, numx=numx)

        if do_plot:
            # Whether to draw mass bin lines
            print "Plotting"
            fig = pf.plot_Pr_samples(x_array, Pr_mat, Pr_fid, plot_dir=self.other_params_fid['plot_dir'],
                                     percentiles=percentiles, do_samples=do_samples, xlim=xlim, ylim=ylim)
            return fig
        else:
            return x_array, Pr_mat, Pr_fid

    def get_YM_fid(self, min_M, max_M, z, numM=None, mdef_Mmin_Mmax=None):
        # M_array = np.exp(np.linspace(np.log(min_M), np.log(max_M), num=numM))

        M_array_mdef = np.logspace(np.log10(min_M), np.log10(max_M), num=numM)

        mdef_Delta = mdef_Mmin_Mmax
        halo_conc_Delta = concentration.concentration(M_array_mdef, mdef_Delta, z)

        if mdef_Mmin_Mmax == '200c':
            M_array_200c = M_array_mdef
        else:
            M_array_200c, _, _ = mass_defs.changeMassDefinition(M_array_mdef, halo_conc_Delta, z, mdef_Delta, '200c')

        if mdef_Mmin_Mmax == '500c':
            M_array_500c = M_array_mdef
        else:
            M_array_500c, _, _ = mass_defs.changeMassDefinition(M_array_mdef, halo_conc_Delta, z, mdef_Delta, '500c')

        #
        # M_array = np.logspace(np.log10(min_M), np.log10(max_M), num=numM)
        integratedY_mat = np.zeros(numM)
        for mi in xrange(0, numM):
            pressure = Pressure(self.cosmo_params_fid, self.pressure_params_fid, self.other_params_fid)
            integratedY = pressure.get_Y500sph_singleMz(M_array_500c[mi], z, do_fast=False, M200c=M_array_200c[mi])
            integratedY_mat[mi] = integratedY

        return M_array_500c, integratedY_mat

    def get_YMmat(self, min_M, max_M, z, cosmo_params_array, pressure_params_array, other_params_array, numM=None,
                     mdef_Mmin_Mmax=None, do_multiprocess=False, num_pool=None):
        # M_array = np.exp(np.linspace(np.log(min_M), np.log(max_M), num=numM))

        M_array_mdef = np.logspace(np.log10(min_M), np.log10(max_M), num=numM)

        # chi_array = hmf.get_Dcom_array(z_array, cosmo_params['Om0'])
        # DA_array = chi_array / (1. + z_array)
        # ng_array = np.exp(self.ng_interp(z_array))
        # dchi_dz_array = (const.c.to(u.km / u.s)).value / (hmf.get_Hz(z_array, cosmo_params['Om0']))

        mdef_Delta = mdef_Mmin_Mmax
        halo_conc_Delta = concentration.concentration(M_array_mdef, mdef_Delta, z)

        if mdef_Mmin_Mmax == '200c':
            M_array_200c = M_array_mdef
        else:
            M_array_200c, _, _ = mass_defs.changeMassDefinition(M_array_mdef, halo_conc_Delta, z, mdef_Delta, '200c')

        if mdef_Mmin_Mmax == '500c':
            M_array_500c = M_array_mdef
        else:
            M_array_500c, _, _ = mass_defs.changeMassDefinition(M_array_mdef, halo_conc_Delta, z, mdef_Delta, '500c')

        integratedY_mat = np.zeros((len(pressure_params_array), numM))

        if do_multiprocess:
            manager = multiprocessing.Manager()
            integratedY_dict = manager.dict()

            processes = []
            if num_pool is None:
                for pi in xrange(0, len(pressure_params_array)):
                    p = multiprocessing.Process(target=self.collect_YM, args=(
                        [pi], integratedY_dict, cosmo_params_array, pressure_params_array, other_params_array,
                        M_array_500c, M_array_200c, z))
                    processes.append(p)
                    p.start()
            else:
                npool = num_pool
                pi_array = np.arange(len(pressure_params_array))
                pi_array_split = np.array_split(pi_array, npool)
                for j in range(npool):
                    p = multiprocessing.Process(target=self.collect_YM, args=(
                        pi_array_split[j], integratedY_dict, cosmo_params_array, pressure_params_array,
                        other_params_array, M_array_500c, M_array_200c, z))
                    processes.append(p)
                    p.start()

            for process in processes:
                process.join()

            for pi in range(len(pressure_params_array)):
                integratedY_mat[pi, :] = integratedY_dict[pi]

        else:
            print "starting Y(M) samples"
            nprint = np.floor(len(pressure_params_array) / 10.)
            for pi in xrange(0, len(pressure_params_array)):
                if float(pi) / nprint - int(pi / nprint) == 0:
                    print "si = ", pi, " out of ", len(pressure_params_array)

                pressure = Pressure(cosmo_params_array[pi], pressure_params_array[pi], other_params_array[pi])
                for mi in xrange(0, numM):
                    integratedY = pressure.get_Y500sph_singleMz(M_array_500c[mi], z, do_fast=False,
                                                                M200c=M_array_200c[mi])
                    integratedY_mat[pi, mi] = integratedY

        return M_array_500c, integratedY_mat

    def generate_rand_params_from_Fisher(self):
        rand_pressure_params = copy.deepcopy(self.pressure_params_fid)
        rand_cosmo_params = copy.deepcopy(self.cosmo_params_fid)
        rand_hod_params = copy.deepcopy(self.hod_params_fid)
        rand_other_params = copy.deepcopy(self.other_params_fid)

        param_cov = np.linalg.inv(self.F_mat_wpriors)
        new_values = np.random.multivariate_normal(self.fid_values, param_cov)

        diff = np.array([new_values - self.fid_values])
        inv_cov_param = np.linalg.inv(param_cov)
        chi2_params = np.dot(np.dot(diff, inv_cov_param), diff.T)

        # print new_values
        rand_cosmo_params, rand_hod_params, rand_pressure_params = self.set_values(rand_cosmo_params, rand_hod_params,
                                                                                   rand_pressure_params, new_values)

        # pdb.set_trace()

        return rand_pressure_params, rand_cosmo_params, rand_hod_params, rand_other_params, chi2_params

    def get_Cly_mat_from_Fisher_partial_usedCldparam(self, nsample=None, do_multiprocess=False, num_pool=None):
        # generate parameter sets from Fisher
        nsamps = nsample
        nprint = np.floor(nsamps / 10.)
        pressure_params_array_nsamps_nparams = []
        hod_params_array_nsamps_nparams = []
        cosmo_params_array_nsamps_nparams = []
        other_params_array_nsamps_nparams = []
        chi2_params_array_nsamps_nparams = []
        Cl_dicts_samples_nsamps_nparams = []
        chi2_samples_nsamps_nparams = []
        new_values_nsamps_nparams = []

        dCl_fisher_nsamps = []

        pressure_params_array = []
        hod_params_array = []
        cosmo_params_array = []
        other_params_array = []
        chi2_params_total_array_nsamps = []

        for si in xrange(0, nsamps):
            cosmo_params_si, hod_params_si, pressure_params_si, other_params_si, cosmo_params_all_si, hod_params_all_si, pressure_params_all_si, other_params_all_si, chi2_param, chi2_params_total, new_values = self.generate_rand_params_from_Fisher_partial()
            pressure_params_array_nsamps_nparams.append(pressure_params_si)
            cosmo_params_array_nsamps_nparams.append(cosmo_params_si)
            other_params_array_nsamps_nparams.append(other_params_si)
            hod_params_array_nsamps_nparams.append(hod_params_si)
            chi2_params_array_nsamps_nparams.append(chi2_param)
            chi2_params_total_array_nsamps.append(chi2_params_total)
            new_values_nsamps_nparams.append(new_values)

            pressure_params_array.append(pressure_params_all_si)
            cosmo_params_array.append(cosmo_params_all_si)
            other_params_array.append(other_params_all_si)
            hod_params_array.append(hod_params_all_si)

            dCl = np.zeros(len(self.l_array_survey))
            for j in range(len(self.fisher_params_vary)):
                param = self.fisher_params_vary[j]
                dparam = new_values[j] - self.fid_values[j]
                dCl_dparam = self.dCl_dparam_dict[param]
                # pdb.set_trace()
                dCl += dCl_dparam * dparam

            dCl_fisher_nsamps.append(dCl)

        l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples = self.get_Clyg_mat_Makiya18(
            cosmo_params_array, pressure_params_array, hod_params_array, other_params_array,
            do_multiprocess=do_multiprocess, num_pool=num_pool)

        chi2_params_array_nsamps_nparams = np.array(chi2_params_array_nsamps_nparams)
        chi2_params_total_array_nsamps = np.array(chi2_params_total_array_nsamps)

        return l_array, Cl_fid_dict, dCl_fisher_nsamps, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples_nsamps_nparams, chi2_params_array_nsamps_nparams, chi2_params_total_array_nsamps, chi2_samples, new_values_nsamps_nparams, self.fid_values, self.fisher_params_label

    def get_Cly_mat_from_Fisher_partial(self, nsample=None, do_multiprocess=False, num_pool=None):
        # generate parameter sets from Fisher
        nsamps = nsample
        nprint = np.floor(nsamps / 10.)
        pressure_params_array_nsamps_nparams = []
        hod_params_array_nsamps_nparams = []
        cosmo_params_array_nsamps_nparams = []
        other_params_array_nsamps_nparams = []
        chi2_params_array_nsamps_nparams = []
        Cl_dicts_samples_nsamps_nparams = []
        chi2_samples_nsamps_nparams = []
        new_values_nsamps_nparams = []

        pressure_params_array = []
        hod_params_array = []
        cosmo_params_array = []
        other_params_array = []
        chi2_params_total_array_nsamps = []

        for si in xrange(0, nsamps):
            cosmo_params_si, hod_params_si, pressure_params_si, other_params_si, cosmo_params_all_si, hod_params_all_si, pressure_params_all_si, other_params_all_si, chi2_param, chi2_params_total, new_values = self.generate_rand_params_from_Fisher_partial()
            pressure_params_array_nsamps_nparams.append(pressure_params_si)
            cosmo_params_array_nsamps_nparams.append(cosmo_params_si)
            other_params_array_nsamps_nparams.append(other_params_si)
            hod_params_array_nsamps_nparams.append(hod_params_si)
            chi2_params_array_nsamps_nparams.append(chi2_param)
            chi2_params_total_array_nsamps.append(chi2_params_total)
            new_values_nsamps_nparams.append(new_values)

            pressure_params_array.append(pressure_params_all_si)
            cosmo_params_array.append(cosmo_params_all_si)
            other_params_array.append(other_params_all_si)
            hod_params_array.append(hod_params_all_si)

            # print "getting Pressure profile"
            l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples = self.get_Clyg_mat_Makiya18(
                cosmo_params_si, pressure_params_si, hod_params_si, other_params_si,
                do_multiprocess=do_multiprocess)

            Cl_dicts_samples_nsamps_nparams.append(Cl_dicts_samples)
            chi2_samples_nsamps_nparams.append(chi2_samples)

        l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples = self.get_Clyg_mat_Makiya18(
            cosmo_params_array, pressure_params_array, hod_params_array, other_params_array,
            do_multiprocess=do_multiprocess, num_pool=num_pool)

        chi2_params_array_nsamps_nparams = np.array(chi2_params_array_nsamps_nparams)
        chi2_params_total_array_nsamps = np.array(chi2_params_total_array_nsamps)

        return l_array, Cl_fid_dict, Cl_dicts_samples_nsamps_nparams, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples_nsamps_nparams, chi2_params_array_nsamps_nparams, chi2_params_total_array_nsamps, chi2_samples, new_values_nsamps_nparams, self.fid_values, self.fisher_params_label

    def generate_rand_params_from_Fisher_partial(self):
        pressure_params = copy.deepcopy(self.pressure_params_fid)
        cosmo_params = copy.deepcopy(self.cosmo_params_fid)
        hod_params = copy.deepcopy(self.hod_params_fid)

        param_cov = np.linalg.inv(self.F_mat_wpriors)

        new_values = np.random.multivariate_normal(self.fid_values, param_cov)

        diff = np.array([new_values - self.fid_values])
        inv_cov_param = np.linalg.inv(param_cov)
        chi2_params_total = np.dot(np.dot(diff, inv_cov_param), diff.T)

        chi2_params = []
        for j in range(len(new_values)):
            diff = np.array([new_values[j] - self.fid_values[j]])
            inv_cov_param = np.linalg.inv(self.F_mat_wpriors[:, [j]][[j], :])
            chi2_params.append(np.dot(np.dot(diff, inv_cov_param), diff.T))

        chi2_params = np.array(chi2_params)
        # print new_values
        rand_cosmo_params, rand_hod_params, rand_pressure_params = self.set_values_partial(cosmo_params,
                                                                                           hod_params,
                                                                                           pressure_params,
                                                                                           new_values)

        rand_cosmo_params_all, rand_hod_params_all, rand_pressure_params_all = self.set_values(cosmo_params,
                                                                                               hod_params,
                                                                                               pressure_params,
                                                                                               new_values)
        rand_other_params_all = self.other_params_fid

        # pdb.set_trace()

        rand_other_params = []
        for j in range(len(rand_cosmo_params)):
            rand_other_params.append(self.other_params_fid)

        return rand_cosmo_params, rand_hod_params, rand_pressure_params, rand_other_params, rand_cosmo_params_all, rand_hod_params_all, rand_pressure_params_all, rand_other_params_all, chi2_params, chi2_params_total, new_values

    def set_values_onlyPressure(self, cosmo_params, hod_params, pressure_params, new_values, ind_in_pressure):

        k = 0
        for fi in ind_in_pressure:

            param_vary = self.fisher_params_vary[fi]
            if param_vary not in self.fisher_params_vary_orig:
                param_vary_split = list(param_vary)
                addition_index = param_vary_split.index('-')
                param_vary_orig = ''.join(param_vary_split[:addition_index])
            else:
                param_vary_orig = param_vary

            if param_vary_orig in self.pressure_params_fid:

                if self.other_params_fid['do_split_params_massbins']:
                    if param_vary not in self.fisher_params_vary_orig:
                        mass_bin_number = int(list(param_vary)[-1])
                        params_new_param_vary = copy.deepcopy(pressure_params[param_vary_orig])
                        params_new_param_vary[mass_bin_number] = new_values[k]
                        pressure_params[param_vary_orig] = params_new_param_vary
                    else:
                        pressure_params[param_vary_orig] = new_values[k]
                else:
                    pressure_params[param_vary_orig] = new_values[k]
            k += 1

        return cosmo_params, hod_params, pressure_params

    def generate_rand_params_from_Fisher_onlyPressure(self):
        rand_pressure_params = copy.deepcopy(self.pressure_params_fid)
        rand_cosmo_params = copy.deepcopy(self.cosmo_params_fid)
        rand_hod_params = copy.deepcopy(self.hod_params_fid)
        rand_other_params = copy.deepcopy(self.other_params_fid)

        param_cov = np.linalg.inv(self.F_mat_wpriors)

        ind_in_pressure = []
        nparam = len(self.fisher_params_vary)
        fid_values_onlyp = []

        for j in range(nparam):
            param_vary = self.fisher_params_vary[j]

            if param_vary not in self.fisher_params_vary_orig:
                param_vary_split = list(param_vary)
                addition_index = param_vary_split.index('-')
                param_vary_orig = ''.join(param_vary_split[:addition_index])
            else:
                param_vary_orig = param_vary

            if param_vary_orig in self.pressure_params_fid.keys():
                ind_in_pressure.append(j)
                fid_values_onlyp.append(self.fid_values[j])

        ind_in_pressure = np.array(ind_in_pressure)
        fid_values_onlyp = np.array(fid_values_onlyp)

        # pdb.set_trace()
        param_cov_onlyp = param_cov[:, ind_in_pressure][ind_in_pressure, :]

        new_values = np.random.multivariate_normal(fid_values_onlyp, param_cov_onlyp)

        # diff = np.array([new_values - fid_values_onlyp])
        # inv_cov_param = np.linalg.inv(param_cov_onlyp)
        # chi2_params = np.dot(np.dot(diff,inv_cov_param),diff.T)

        rand_cosmo_params, rand_hod_params, rand_pressure_params = self.set_values_onlyPressure(rand_cosmo_params,
                                                                                                rand_hod_params,
                                                                                                rand_pressure_params,
                                                                                                new_values,
                                                                                                ind_in_pressure)

        return rand_pressure_params, rand_cosmo_params, rand_hod_params, rand_other_params

    def get_Cly_mat_func_param(self, nsample=None, do_multiprocess=False, num_pool=None, param_minmax_dict=None):
        nsamps = nsample
        nprint = np.floor(nsamps / 10.)

        Cl_dicts_samples_nsamps_nparams = {}
        chi2_samples_nsamps_nparams = {}
        value_dict = {}

        nparams = len(self.fisher_params_vary)
        for pi in range(nparams):

            param_vary = self.fisher_params_vary[pi]
            if param_vary not in self.fisher_params_vary_orig:
                param_vary_split = list(param_vary)
                addition_index = param_vary_split.index('-')
                param_vary_orig = ''.join(param_vary_split[:addition_index])
            else:
                param_vary_orig = param_vary

            print 'doing ', param_vary
            param_fid = self.fid_values[pi]

            dparam = param_minmax_dict[param_vary]
            new_values = np.linspace(param_fid - dparam, param_fid + dparam, nsamps)

            pressure_params_pi, cosmo_params_pi, hod_params_pi, other_params_pi = [], [], [], []

            for si in range(len(new_values)):
                pressure_params = copy.deepcopy(self.pressure_params_fid)
                cosmo_params = copy.deepcopy(self.cosmo_params_fid)
                hod_params = copy.deepcopy(self.hod_params_fid)
                other_params = copy.deepcopy(self.other_params_fid)

                if param_vary_orig in self.cosmo_params_fid:
                    cosmo_params[param_vary_orig] = new_values[si]
                if param_vary_orig in self.hod_params_fid:
                    hod_params[param_vary_orig] = new_values[si]
                if param_vary_orig in self.pressure_params_fid:

                    if self.other_params_fid['do_split_params_massbins']:
                        if param_vary not in self.fisher_params_vary_orig:
                            mass_bin_number = int(list(param_vary)[-1])
                            params_new_param_vary = copy.deepcopy(pressure_params[param_vary_orig])
                            params_new_param_vary[mass_bin_number] = new_values[si]
                            pressure_params[param_vary_orig] = params_new_param_vary
                        else:
                            pressure_params[param_vary_orig] = new_values[si]
                    else:
                        pressure_params[param_vary_orig] = new_values[si]

                pressure_params_pi.append(pressure_params)
                cosmo_params_pi.append(cosmo_params)
                hod_params_pi.append(hod_params)
                other_params_pi.append(other_params)

            l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples = self.get_Clyg_mat_Makiya18(
                cosmo_params_pi, pressure_params_pi, hod_params_pi, other_params_pi,
                do_multiprocess=do_multiprocess, num_pool=num_pool)

            Cl_dicts_samples_nsamps_nparams[param_vary] = Cl_dicts_samples
            chi2_samples_nsamps_nparams[param_vary] = chi2_samples
            value_dict[param_vary] = new_values

        return l_array, Cl_fid_dict, Cl_dicts_samples_nsamps_nparams, cov_fid, inv_cov_fid, chi2_samples_nsamps_nparams, value_dict, self.fisher_params_vary, self.fid_values

    def get_YMmat_from_Fisher(self, min_M, max_M, z, numM=None, nsample=None, mdef_Mmin_Mmax=None):
        # generate parameter sets from Fisher
        nsamps = nsample
        pressure_params_array = []
        cosmo_params_array = []
        other_params_array = []
        nprint = np.floor(nsamps / 10.)
        for si in xrange(0, nsamps):
            pressure_params_si, cosmo_params_si, hod_params_si, other_params_si, chi2_params = self.generate_rand_params_from_Fisher()
            pressure_params_array.append(pressure_params_si)
            cosmo_params_array.append(cosmo_params_si)
            other_params_array.append(other_params_si)

        print "getting YM limits"
        M_array, integratedY_mat = self.get_YMmat(min_M, max_M, z, cosmo_params_array, pressure_params_array,
                                                  other_params_array, numM=numM, mdef_Mmin_Mmax=mdef_Mmin_Mmax)

        return M_array, integratedY_mat

    def plot_YM_relation(self, z=0.15, numM=100, nsample=2000, M_min=1.0e12, M_max=1.0e16, xlim=None, do_samples=True,
                         percentiles=None, ylim=None, do_plot=True, mdef_Mmin_Mmax=None):
        M_array, integratedY_mat = self.get_YMmat_from_Fisher(M_min, M_max, z, numM=numM, nsample=nsample,
                                                              mdef_Mmin_Mmax=mdef_Mmin_Mmax)

        # Get fiducial relation
        M_array, YM_fid = self.get_YM_fid(M_min, M_max, z, numM=numM, mdef_Mmin_Mmax=mdef_Mmin_Mmax)

        # Only generate plot if requested

        if do_plot:
            # Whether to draw mass bin lines
            if self.other_params_fid['do_split_params_massbins'] and (
                    self.other_params_fid['pressure_model_type'] != 'broken_powerlaw'):
                split_mass_bins_min = self.other_params_fid['split_mass_bins_min']
                do_split_params_massbins = True
            else:
                split_mass_bins_min = None
                do_split_params_massbins = False

            print "Plotting"
            fig = pf.plot_YM_relation_samples(M_array, integratedY_mat, YM_fid,
                                              do_split_params_massbins,
                                              split_mass_bins_min=split_mass_bins_min,
                                              plot_dir=self.other_params_fid['plot_dir'],
                                              percentiles=percentiles, do_samples=do_samples, xlim=xlim, ylim=ylim)
            return fig
        else:
            return M_array, integratedY_mat, YM_fid

    def plot_YM_relation_relative(self, z=0.15, numM=100, nsample=200, M_min=1.0e12, M_max=1.0e16, xlim=None,
                                  do_samples=True, percentiles=None, labels=None, mdef_Mmin_Mmax=None):
        # min_M = np.max([M_min, 10 ** self.other_params_fid['log_M_min_tracer']])
        # max_M = np.min([M_max, 10 ** self.other_params_fid['log_M_max_tracer']])
        M_array, integratedY_mat = self.get_YMmat_from_Fisher(M_min, M_max, z, numM=numM, nsample=nsample,
                                                              mdef_Mmin_Mmax=mdef_Mmin_Mmax)

        # Get fiducial relation
        M_array, YM_fid = self.get_YM_fid(M_min, M_max, z, numM=numM, mdef_Mmin_Mmax=mdef_Mmin_Mmax)

        # Whether to draw mass bin lines
        if self.other_params_fid['do_split_params_massbins'] and (
                self.other_params_fid['pressure_model_type'] not in ['broken_powerlaw', 'superbroken_powerlaw']):
            split_mass_bins_min = self.other_params_fid['split_mass_bins_min']
            do_split_params_massbins = True
        else:
            split_mass_bins_min = None
            do_split_params_massbins = False

        print "Plotting"
        fig = pf.plot_YM_relation_relative(M_array, integratedY_mat, YM_fid,
                                           do_split_params_massbins,
                                           split_mass_bins_min=split_mass_bins_min,
                                           plot_dir=self.other_params_fid['plot_dir'],
                                           percentiles=percentiles, do_samples=do_samples, xlim=xlim, labels=labels)
        return fig
