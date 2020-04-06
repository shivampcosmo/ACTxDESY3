import sys, os
import pdb
import numpy as np
import scipy as sp
import scipy.special as spsp
from scipy import interpolate
import astropy.units as u
from astropy import constants as const
import colossus
from colossus.cosmology import cosmology
from colossus.lss import bias
from colossus.lss import mass_function
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import concentration
import copy
import itertools
sys.path.insert(0, os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/helper/')
# sys.path.insert(0, '../../helper/')
import mycosmo as cosmodef
from twobessel import *
import LSS_funcs as hmf
import plot_funcs as pf
import multiprocessing
import time
import pdb
from mcfit import Hankel

pi = np.pi


class HOD:
    """
    Sets up the HOD class for the galaxies.
    """

    def __init__(self, hod_params):
        self.hod_params = hod_params
        self.hod_type = hod_params['hod_type']

    # Average number of central galaxies
    def get_Nc(self, M_val):
        if self.hod_type == 'Halos':
            Ncm = 0.5 * (np.sign((M_val - 10 ** self.hod_params['logMmin'])) + 1.)
        elif self.hod_type in ['DESI', '2MRS']:
            erfval = sp.special.erf((np.log10(M_val) - self.hod_params['logMmin']) / self.hod_params['sig_logM'])
            Ncm = 0.5 * (1 + erfval)
        elif self.hod_type == 'DES_MICE':
            erfval = sp.special.erf((np.log10(M_val) - self.hod_params['logMmin']) / self.hod_params['sig_logM'])
            Ncm = self.hod_params['fmaxcen'] * (
                    1.0 - (1.0 - self.hod_params['fmincen'] / self.hod_params['fmaxcen']) / (1.0 + 10 ** (
                    (2.0 / self.hod_params['fcen_k']) * (
                    np.log10(M_val) - self.hod_params['log_mdrop'])))) * 0.5 * (1 + erfval)
        elif self.hod_type == 'DES_GGL':
            erfval = sp.special.erf((np.log10(M_val) - self.hod_params['logMmin']) / self.hod_params['sig_logM'])
            Ncm = self.hod_params['fcen'] * 0.5 * (1 + erfval)
        else:
            print('give correct HOD type')
            sys.exit(1)
        return Ncm

    # Average number of satellite galaxies
    def get_Ns(self, M_val):
        if self.hod_type == 'Halos':
            Nsm = np.zeros(M_val.shape)
        elif self.hod_type in ['DESI', '2MRS']:
            M0 = 10 ** (self.hod_params['logM0'])
            M1 = 10 ** (self.hod_params['logM1'])
            val = 0.5 * (np.sign((M_val - M0)) + 1.) * ((M_val - M0) / M1)
            Nsm = np.power(val, self.hod_params['alpha_g'])
        elif self.hod_type == 'DES_MICE':
            erfval = sp.special.erf((np.log10(M_val) - self.hod_params['logMmin']) / self.hod_params['sig_logM'])
            Ncerf = 0.5 * (1 + erfval)
            Ncm = get_Nc(self, M_val)
            M1 = 10 ** (self.hod_params['logM1'])
            Nsm = (Ncerf / Ncm) * ((M_val / M1) ** self.hod_params['alpha_g'])
        elif self.hod_type == 'DES_GGL':
            M1 = 10 ** (self.hod_params['logM1'])
            Nsm = ((M_val / M1) ** self.hod_params['alpha_g'])
        else:
            print('give correct HOD type')
            sys.exit(1)
        return Nsm

    # total number of galaxies = (Ncm*(1+Nsm))
    def get_Ntotal(self, M_val):
        Ncm = self.get_Nc(M_val)
        Nsm = self.get_Ns(M_val)
        ntm = Ncm * (1 + Nsm)
        return ntm


class Pressure:
    """
    Sets up the pressure profile functions.
    """
    def __init__(self, cosmo_params, pressure_params, other_params):
        cosmology.addCosmology('mock_cosmo', cosmo_params)
        self.cosmo_colossus = cosmology.setCosmology('mock_cosmo')
        h = cosmo_params['H0'] / 100.
        cosmo_func = cosmodef.mynew_cosmo(h, cosmo_params['Om0'], cosmo_params['Ob0'], cosmo_params['ns'],
                                          cosmo_params['sigma8'])
        self.cosmo = cosmo_func
        self.verbose = other_params['verbose']

        self.z_array = np.linspace(other_params['z_array_min'], other_params['z_array_max'], other_params['num_z'])

        self.do_split_params_massbins = other_params['do_split_params_massbins']
        self.pressure_params_dict = pressure_params
        self.use_multiprocess = other_params['use_multiprocess']

        self.pressure_model_type = other_params['pressure_model_type']
        self.pressure_model_name = other_params['pressure_model_name']
        self.pressure_model_mdef = other_params['pressure_model_mdef']
        self.mdef_analysis = other_params['mdef_analysis']
        self.pressure_model_delta = other_params['pressure_model_delta']

        self.other_params = other_params

        if self.do_split_params_massbins:
            self.split_params_massbins_names = other_params['split_params_massbins_names']
            self.split_mass_bins_min = other_params['split_mass_bins_min']
            self.split_mass_bins_max = other_params['split_mass_bins_max']
            self.split_mass_bins_centers = other_params['split_mass_bins_centers']

    # get generalized nfw profile for the pressure profile from Arnard et al 2010 paper
    def get_gnfwp_Arnaud10(self, x, M_mat_shape_nz_nm):

        params_names_gnfwp = copy.deepcopy(self.pressure_params_dict.keys())
        params_names_gnfwp.remove('alpha_p')
        if 'Mstar' in params_names_gnfwp:
            params_names_gnfwp.remove('Mstar')

        nz, nm = M_mat_shape_nz_nm.shape
        nx = len(x)
        xmat = np.tile(x.reshape(1, 1, nx), (nz, nm, 1))

        M_mat_cond_dict = {}
        if self.do_split_params_massbins:
            for i in range(len(params_names_gnfwp)):
                param_i = params_names_gnfwp[i]
                if self.do_split_params_massbins:
                    if param_i in self.split_params_massbins_names:
                        M_mat_cond_shape_nz_nm_nx = []

                        if self.pressure_model_type in ['fullybroken_powerlaw', 'superbroken_powerlaw',
                                                        'superPlybroken_powerlaw']:
                            for j in range(len(self.split_mass_bins_centers)):
                                if j == 0:
                                    M_mat_cond_j = (M_mat_shape_nz_nm <= (
                                            10 ** self.pressure_params_dict['logMstar'][param_i]) * (
                                                            0.7 / self.cosmo.h))
                                    M_mat_cond_j_reshape = np.tile(M_mat_cond_j.reshape(nz, nm, 1), (1, 1, nx))
                                    M_mat_cond_shape_nz_nm_nx.append(M_mat_cond_j_reshape)

                                if j == 1:
                                    M_mat_cond_j = (M_mat_shape_nz_nm >= (
                                            10 ** self.pressure_params_dict['logMstar'][param_i]) * (
                                                            0.7 / self.cosmo.h))
                                    M_mat_cond_j_reshape = np.tile(M_mat_cond_j.reshape(nz, nm, 1), (1, 1, nx))
                                    M_mat_cond_shape_nz_nm_nx.append(M_mat_cond_j_reshape)

                        else:
                            for j in range(len(self.split_mass_bins_centers)):
                                M_mat_cond_j = np.logical_and(M_mat_shape_nz_nm >= self.split_mass_bins_min[j],
                                                              M_mat_shape_nz_nm <= self.split_mass_bins_max[j])
                                M_mat_cond_j_reshape = np.tile(M_mat_cond_j.reshape(nz, nm, 1), (1, 1, nx))
                                M_mat_cond_shape_nz_nm_nx.append(M_mat_cond_j_reshape)

                        M_mat_cond_dict[param_i] = M_mat_cond_shape_nz_nm_nx

        params_array_value_gnfwp = {}
        for j in range(len(params_names_gnfwp)):
            param_j = params_names_gnfwp[j]
            if self.do_split_params_massbins:
                if param_j in self.split_params_massbins_names:
                    param_j_array = np.zeros((nz, nm, nx))
                    for i in range(len(self.split_mass_bins_centers)):
                        param_j_array += self.pressure_params_dict[param_j][i] * M_mat_cond_dict[param_j][i]
                else:
                    param_j_array = self.pressure_params_dict[param_j]
            else:
                param_j_array = self.pressure_params_dict[param_j]
            params_array_value_gnfwp[param_j] = param_j_array

        val_num = params_array_value_gnfwp['P0'] * ((0.7 / self.cosmo.h) ** (3. / 2.))
        val_denom_1 = np.power(params_array_value_gnfwp['c500'] * xmat, params_array_value_gnfwp['gamma'])
        val_denom_2 = 1. + np.power(params_array_value_gnfwp['c500'] * xmat, params_array_value_gnfwp['alpha'])
        val_denom_3 = (params_array_value_gnfwp['beta'] - params_array_value_gnfwp['gamma']) / \
                      params_array_value_gnfwp['alpha']
        val_denom = val_denom_1 * np.power(val_denom_2, val_denom_3)
        del val_denom_1, val_denom_2, val_denom_3, xmat
        valf = val_num / val_denom

        return valf

    def get_Pe_mat_Arnaud10(self, M_mat, x_array, z_array, Mmat_cond=None, zmat_cond=None):
        nz, nm = M_mat.shape
        nx = len(x_array)

        Ez_array_coeff = (hmf.get_Ez(z_array, self.cosmo.Om0)) ** (8. / 3.)
        Ez_mat_coeff = np.tile(Ez_array_coeff.reshape(nz, 1, 1), (1, nm, nx))

        gnfwp_mat_coeff = self.get_gnfwp_Arnaud10(x_array, M_mat)

        if self.do_split_params_massbins:
            M_mat_cond_shape_nz_nm = []

            if self.pressure_model_type in ['broken_powerlaw', 'fullybroken_powerlaw', 'superbroken_powerlaw',
                                            'superPlybroken_powerlaw']:
                M_mat_cond_shape_nz_nm.append(
                    M_mat <= (10 ** self.pressure_params_dict['logMstar']['alpha_p']) * (0.7 / self.cosmo.h))
                M_mat_cond_shape_nz_nm.append(
                    M_mat >= (10 ** self.pressure_params_dict['logMstar']['alpha_p']) * (0.7 / self.cosmo.h))

                alpha_p_array = np.zeros((nz, nm))
                for i in range(2):
                    alpha_p_array += self.pressure_params_dict['alpha_p'][i] * M_mat_cond_shape_nz_nm[i]

            else:
                for j in range(len(self.split_mass_bins_centers)):
                    M_mat_cond_j = np.logical_and(M_mat > self.split_mass_bins_min[j],
                                                  M_mat < self.split_mass_bins_max[j])
                    M_mat_cond_shape_nz_nm.append(M_mat_cond_j)

                if 'alpha_p' in self.split_params_massbins_names:
                    alpha_p_array = np.zeros((nz, nm))
                    for i in range(len(self.split_mass_bins_centers)):
                        alpha_p_array += self.pressure_params_dict['alpha_p'][i] * M_mat_cond_shape_nz_nm[i]
                else:
                    alpha_p_array = self.pressure_params_dict['alpha_p'] * np.ones((nz, nm))
        else:
            alpha_p_array = self.pressure_params_dict['alpha_p'] * np.ones((nz, nm))

        M_array_coeff = np.power(
            (M_mat / ((10 ** self.pressure_params_dict['logMstar']['alpha_p']) * (0.7 / self.cosmo.h))),
            (2. / 3. + alpha_p_array))
        M_mat_coeff = np.tile(M_array_coeff.reshape(nz, nm, 1), (1, 1, nx))

        Pe_mat = (1.65 * (self.cosmo.h / 0.7) ** 2) * Ez_mat_coeff * gnfwp_mat_coeff * M_mat_coeff

        return Pe_mat

    def get_gnfwp_LeBrun15(self, x, M_mat_shape_nz_nm):
        # import pdb; pdb.set_trace()
        params_names_gnfwp = copy.deepcopy([*self.pressure_params_dict.keys()])
        # params_names_gnfwp.remove('alpha_p')
        # if 'Mstar' in params_names_gnfwp:
        #     params_names_gnfwp.remove('Mstar')

        nz, nm = M_mat_shape_nz_nm.shape
        nx = len(x)
        xmat = np.tile(x.reshape(1, 1, nx), (nz, nm, 1))
        M_mat_Delta_no_h_nz_nm_nx = (np.tile(M_mat_shape_nz_nm.reshape(nz, nm, 1), (1, 1, nx))) / (self.cosmo.h)

        # M_mat_cond_dict = {}
        # if self.do_split_params_massbins:
        #     for i in range(len(params_names_gnfwp)):
        #         param_i = params_names_gnfwp[i]
        #         if self.do_split_params_massbins:
        #             if param_i in self.split_params_massbins_names:
        #                 M_mat_cond_shape_nz_nm_nx = []
        #
        #                 if self.pressure_model_type in ['fullybroken_powerlaw', 'superbroken_powerlaw',
        #                                                 'superPlybroken_powerlaw']:
        #                     for j in range(len(self.split_mass_bins_centers)):
        #                         if j == 0:
        #                             M_mat_cond_j = (M_mat_shape_nz_nm <= (
        #                                     10 ** self.pressure_params_dict['logMstar'][param_i]) * (
        #                                                     0.7 / self.cosmo.h))
        #                             M_mat_cond_j_reshape = np.tile(M_mat_cond_j.reshape(nz, nm, 1), (1, 1, nx))
        #                             M_mat_cond_shape_nz_nm_nx.append(M_mat_cond_j_reshape)
        #
        #                         if j == 1:
        #                             M_mat_cond_j = (M_mat_shape_nz_nm >= (
        #                                     10 ** self.pressure_params_dict['logMstar'][param_i]) * (
        #                                                     0.7 / self.cosmo.h))
        #                             M_mat_cond_j_reshape = np.tile(M_mat_cond_j.reshape(nz, nm, 1), (1, 1, nx))
        #                             M_mat_cond_shape_nz_nm_nx.append(M_mat_cond_j_reshape)
        #
        #                 else:
        #                     for j in range(len(self.split_mass_bins_centers)):
        #                         M_mat_cond_j = np.logical_and(M_mat_shape_nz_nm >= self.split_mass_bins_min[j],
        #                                                       M_mat_shape_nz_nm <= self.split_mass_bins_max[j])
        #                         M_mat_cond_j_reshape = np.tile(M_mat_cond_j.reshape(nz, nm, 1), (1, 1, nx))
        #                         M_mat_cond_shape_nz_nm_nx.append(M_mat_cond_j_reshape)
        #
        #                 M_mat_cond_dict[param_i] = M_mat_cond_shape_nz_nm_nx

        # params_array_value_gnfwp = {}
        # for j in range(len(params_names_gnfwp)):
        #     param_j = params_names_gnfwp[j]
        #     if self.do_split_params_massbins:
        #         if param_j in self.split_params_massbins_names:
        #             param_j_array = np.zeros((nz, nm, nx))
        #             for i in range(len(self.split_mass_bins_centers)):
        #                 param_j_array += self.pressure_params_dict[param_j][i] * M_mat_cond_dict[param_j][i]
        #         else:
        #             param_j_array = self.pressure_params_dict[param_j]
        #     else:
        #         param_j_array = self.pressure_params_dict[param_j]
        #     params_array_value_gnfwp[param_j] = param_j_array

        params_array_value_gnfwp = {}
        for j in range(len(params_names_gnfwp)):
            param_j = params_names_gnfwp[j]
            if param_j == 'c500':
                param_j_array = self.pressure_params_dict[param_j] * np.power((M_mat_Delta_no_h_nz_nm_nx / (10 ** 14)),
                                                                              self.pressure_params_dict['delta'])
            else:
                param_j_array = self.pressure_params_dict[param_j]
            params_array_value_gnfwp[param_j] = param_j_array

        val_num = params_array_value_gnfwp['P0']
        val_denom_1 = np.power(params_array_value_gnfwp['c500'] * xmat, params_array_value_gnfwp['gamma'])
        val_denom_2 = 1. + np.power(params_array_value_gnfwp['c500'] * xmat, params_array_value_gnfwp['alpha'])
        val_denom_3 = (params_array_value_gnfwp['beta'] - params_array_value_gnfwp['gamma']) / \
                      params_array_value_gnfwp['alpha']
        val_denom = val_denom_1 * np.power(val_denom_2, val_denom_3)
        del val_denom_1, val_denom_2, val_denom_3, xmat
        valf = val_num / val_denom

        return valf

    def get_Pe_mat_LeBrun15(self, M_mat_Delta, x_array, z_array, R_mat_Delta, mdef_Delta=None,
                            Mmat_cond=None, zmat_cond=None):
        nz, nm = M_mat_Delta.shape

        if mdef_Delta is None:
            mdef_Delta = self.other_params['pressure_model_mdef']

        # if M200c_mat is None:
        #     if self.verbose:
        #         print('changing mdef to 200c for battaglia profiles in function get_Pe_mat_Battaglia12')
        # 
        #         ti = time.time()
        # 
        #     halo_conc_Delta = np.zeros(M_mat_Delta.shape)
        #     # pdb.set_trace()
        #     for j in range(len(z_array)):
        #         M_array = M_mat_Delta[j, :]
        #         halo_conc_Delta[j, :] = concentration.concentration(M_array, mdef_Delta, z_array[j])
        # 
        #     M200c_mat, R200c_mat_kpc_h = np.zeros(M_mat_Delta.shape), np.zeros(M_mat_Delta.shape)
        #     for j in range(nz):
        #         M200c_mat[j, :], R200c_mat_kpc_h[j, :], _ = mass_defs.changeMassDefinition(M_mat_Delta[j, :],
        #                                                                                    halo_conc_Delta[j, :],
        #                                                                                    z_array[j], mdef_Delta,
        #                                                                                    '200c')
        # 
        #     hydro_B = self.pressure_params_dict['hydro_mb']
        #     M200c_mat = M200c_mat / hydro_B
        # 
        #     if self.verbose:
        #         print('that took ', time.time() - ti, 'seconds')

        # print('M200c=' + str(np.log10(M200c_mat)))

        nz, nm = M_mat_Delta.shape
        nx = len(x_array)
        # units (Msun) / (Mpc ** 3)
        rho_crit_array = self.cosmo_colossus.rho_c(z_array) * (1000 ** 3) * (self.cosmo.h ** 2)
        rho_crit_mat = np.tile(rho_crit_array.reshape(nz, 1, 1), (1, nm, nx))
        M_mat_Delta_no_h = (M_mat_Delta / self.cosmo.h)
        R_mat_Delta_no_h = (R_mat_Delta / self.cosmo.h)
        M_mat_Delta_no_h_nz_nm_nx = np.tile(M_mat_Delta_no_h.reshape(nz, nm, 1), (1, 1, nx))
        R_mat_Delta_no_h_nz_nm_nx = np.tile(R_mat_Delta_no_h.reshape(nz, nm, 1), (1, 1, nx))
        coeff = (const.G * (const.M_sun ** 2) / ((1.0 * u.Mpc) ** 4)).to((u.eV / (u.cm ** 3))).value

        pressure_model_delta = float(''.join(list(mdef_Delta)[:-1]))
        P_Delta = (coeff / 2.) * pressure_model_delta * (
                self.cosmo.Ob0 / self.cosmo.Om0) * M_mat_Delta_no_h_nz_nm_nx * rho_crit_mat / R_mat_Delta_no_h_nz_nm_nx

        gnfw_P = self.get_gnfwp_LeBrun15(x_array, M_mat_Delta)
        M_mat_Delta_no_h = (M_mat_Delta / self.cosmo.h)
        M_mat_Delta_no_h_nz_nm_nx = np.tile(M_mat_Delta_no_h.reshape(nz, nm, 1), (1, 1, nx))

        # if self.pressure_model_type == 'lowbroken_powerlaw':
        #     M_mat_cond_shape_nz_nm = []
        #     M_mat_cond_shape_nz_nm.append(
        #         M_mat_Delta_no_h < (10 ** self.pressure_params_dict['logMstar']['alpha_p_low']))
        #     M_mat_cond_shape_nz_nm.append(
        #         M_mat_Delta_no_h > (10 ** self.pressure_params_dict['logMstar']['alpha_p_low']))
        #
        #     alpha_p_array = np.zeros((nz, nm))
        #     for i in range(2):
        #         if i == 0:
        #             alpha_p_array += self.pressure_params_dict['alpha_p_low'] * M_mat_cond_shape_nz_nm[i]
        #         else:
        #             alpha_p_array += 0.0 * M_mat_cond_shape_nz_nm[i]
        #
        #     alpha_p_mat = np.tile(alpha_p_array.reshape(nz, nm, 1), (1, 1, nx))
        #     M_mat_coeff = np.power(
        #         (M_mat_Delta_no_h_nz_nm_nx / (10 ** self.pressure_params_dict['logMstar']['alpha_p_low'])),
        #         alpha_p_mat)
        #
        # elif self.pressure_model_type == 'highbroken_powerlaw':
        #     M_mat_cond_shape_nz_nm = []
        #     M_mat_cond_shape_nz_nm.append(
        #         M_mat_Delta_no_h < (10 ** self.pressure_params_dict['logMstar']['alpha_p_high']))
        #     M_mat_cond_shape_nz_nm.append(
        #         M_mat_Delta_no_h > (10 ** self.pressure_params_dict['logMstar']['alpha_p_high']))
        #
        #     alpha_p_array = np.zeros((nz, nm))
        #     for i in range(2):
        #         if i == 0:
        #             alpha_p_array += self.pressure_params_dict['alpha_p_high'] * M_mat_cond_shape_nz_nm[i]
        #         else:
        #             alpha_p_array += 0.0 * M_mat_cond_shape_nz_nm[i]
        #
        #     alpha_p_mat = np.tile(alpha_p_array.reshape(nz, nm, 1), (1, 1, nx))
        #     M_mat_coeff = np.power(
        #         (M_mat_Delta_no_h_nz_nm_nx / (10 ** self.pressure_params_dict['logMstar']['alpha_p_high'])),
        #         alpha_p_mat)
        #
        # elif self.pressure_model_type == 'doublybroken_powerlaw':
        #     M_mat_cond_shape_nz_nm = []
        #     M_mat_cond_shape_nz_nm.append(
        #         M_mat_Delta_no_h < (10 ** self.pressure_params_dict['logMstar']['alpha_p_low']))
        #     M_mat_cond_shape_nz_nm.append(
        #         M_mat_Delta_no_h > (10 ** self.pressure_params_dict['logMstar']['alpha_p_low']))
        #
        #     alpha_p_array = np.zeros((nz, nm))
        #     for i in range(2):
        #         if i == 0:
        #             alpha_p_array += self.pressure_params_dict['alpha_p_low'] * M_mat_cond_shape_nz_nm[i]
        #         else:
        #             alpha_p_array += 0.0 * M_mat_cond_shape_nz_nm[i]
        #
        #     alpha_p_mat = np.tile(alpha_p_array.reshape(nz, nm, 1), (1, 1, nx))
        #     M_mat_coeff1 = np.power(
        #         (M_mat_Delta_no_h_nz_nm_nx / (10 ** self.pressure_params_dict['logMstar']['alpha_p_low'])),
        #         alpha_p_mat)
        #
        #     M_mat_cond_shape_nz_nm = []
        #     M_mat_cond_shape_nz_nm.append(
        #         M_mat_Delta_no_h < (10 ** self.pressure_params_dict['logMstar']['alpha_p_high']))
        #     M_mat_cond_shape_nz_nm.append(
        #         M_mat_Delta_no_h > (10 ** self.pressure_params_dict['logMstar']['alpha_p_high']))
        #
        #     alpha_p_array = np.zeros((nz, nm))
        #     for i in range(2):
        #         if i == 0:
        #             alpha_p_array += self.pressure_params_dict['alpha_p_high'] * M_mat_cond_shape_nz_nm[i]
        #         else:
        #             alpha_p_array += 0.0 * M_mat_cond_shape_nz_nm[i]
        #
        #     alpha_p_mat = np.tile(alpha_p_array.reshape(nz, nm, 1), (1, 1, nx))
        #     M_mat_coeff2 = np.power(
        #         (M_mat_Delta_no_h_nz_nm_nx / (10 ** self.pressure_params_dict['logMstar']['alpha_p_high'])),
        #         alpha_p_mat)
        #
        #     M_mat_coeff = M_mat_coeff1 * M_mat_coeff2
        #
        #     # pdb.set_trace()
        #
        # else:
        #     M_mat_coeff = np.ones((nz, nm, nx))

        M_mat_coeff = np.power((M_mat_Delta_no_h_nz_nm_nx / (10 ** 14)), self.pressure_params_dict['epsilon'])
        Pe_mat = 0.518 * P_Delta * gnfw_P * M_mat_coeff

        # pdb.set_trace()

        return Pe_mat

    def get_gnfwp_Battaglia12(self, M_mat_shape_nz_nm, x_array, z_array, Mmat_cond=None, zmat_cond=None):
        # pdb.set_trace()
        pressure_fid = self.other_params['pressure_fid']
        nz, nm = M_mat_shape_nz_nm.shape
        nx = len(x_array)
        xmat = np.tile(x_array.reshape(1, 1, nx), (nz, nm, 1))

        if 'B12new' in self.pressure_params_dict['logMstar'].keys():
            pivot_B12 = 10 ** self.pressure_params_dict['logMstar']['B12']
            pivot_new = 10 ** self.pressure_params_dict['logMstar']['B12new']
            pivot_ratio = pivot_new / pivot_B12
            M_mat_no_h__Mstar = (np.tile(M_mat_shape_nz_nm.reshape(nz, nm, 1), (1, 1, nx))) / (
                    self.cosmo.h * (10 ** self.pressure_params_dict['logMstar']['B12new']))
        else:
            pivot_ratio = 1.
            M_mat_no_h__Mstar = (np.tile(M_mat_shape_nz_nm.reshape(nz, nm, 1), (1, 1, nx))) / (
                    self.cosmo.h * (10 ** self.pressure_params_dict['logMstar']['B12']))

        #######################################
        # BEWARE!! ONLY WORKS WHEN M_MAT_SHAPE_NZ_NM IS IN SAME MDEF AS LOG_M_MIN_TRACER
        ######################################
        Mmat_cond = np.logical_and(M_mat_shape_nz_nm >= (10 ** self.other_params['log_M_min_tracer']),
                                   M_mat_shape_nz_nm <= (10 ** self.other_params['log_M_max_tracer']))
        zmat = np.tile(z_array.reshape(nz, 1), (1, nm))
        zmat_cond = np.logical_and(zmat >= (self.other_params['zmin_tracer']),
                                   zmat <= (self.other_params['zmax_tracer']))

        Mmat_cond_nx = (np.tile(Mmat_cond.reshape(nz, nm, 1), (1, 1, nx)))
        zmat_cond_nx = (np.tile(zmat_cond.reshape(nz, nm, 1), (1, 1, nx)))

        def get_cond_var(var_name, new_val_dict=self.pressure_params_dict, fid_val_dict=pressure_fid,
                         cond_mat_m=Mmat_cond_nx, cond_mat_z=zmat_cond_nx):
            new_val = new_val_dict[var_name]
            fid_val = fid_val_dict[var_name]
            cond_mat_f = np.logical_and(cond_mat_m, cond_mat_z)
            negcond_mat_f = np.logical_not(cond_mat_f)

            result = new_val * cond_mat_f + fid_val * negcond_mat_f
            # pdb.set_trace()

            return result

        gamma_Am = get_cond_var('gamma-A_m') * (pivot_ratio ** get_cond_var('gamma-alpha_m'))
        alpha_Am = get_cond_var('alpha-A_m') * (pivot_ratio ** get_cond_var('alpha-alpha_m'))
        beta_Am = get_cond_var('beta-A_m') * (pivot_ratio ** get_cond_var('beta-alpha_m'))
        xc_Am = get_cond_var('xc-A_m') * (pivot_ratio ** get_cond_var('xc-alpha_m'))
        P0_Am = get_cond_var('P0-A_m') * (pivot_ratio ** get_cond_var('P0-alpha_m'))

        one_plus_z_mat = 1 + np.tile(z_array.reshape(nz, 1, 1), (1, nm, nx))

        gamma_mat = gamma_Am * (M_mat_no_h__Mstar ** get_cond_var('gamma-alpha_m')) * (
                one_plus_z_mat ** get_cond_var('gamma-alpha_z'))

        alpha_mat = alpha_Am * (M_mat_no_h__Mstar ** get_cond_var('alpha-alpha_m')) * (
                one_plus_z_mat ** get_cond_var('alpha-alpha_z'))

        beta_mat = beta_Am * (M_mat_no_h__Mstar ** get_cond_var('beta-alpha_m')) * (
                one_plus_z_mat ** get_cond_var('beta-alpha_z'))

        xc_mat = xc_Am * (M_mat_no_h__Mstar ** get_cond_var('xc-alpha_m')) * (
                one_plus_z_mat ** get_cond_var('xc-alpha_z'))

        P0_mat = P0_Am * (M_mat_no_h__Mstar ** get_cond_var('P0-alpha_m')) * (
                one_plus_z_mat ** get_cond_var('P0-alpha_z'))

        del one_plus_z_mat, M_mat_no_h__Mstar

        val1 = np.power(xmat / xc_mat, gamma_mat)
        del gamma_mat

        val2 = np.power(1. + np.power(xmat / xc_mat, alpha_mat), -1.0 * beta_mat)
        del alpha_mat, beta_mat, xc_mat, xmat,

        valf = P0_mat * val1 * val2

        # pdb.set_trace()
        return valf

    def get_Pe_mat_Battaglia12(self, M_mat_Delta, x_array, z_array, R_mat_Delta, M200c_mat=None, mdef_Delta=None,
                               Mmat_cond=None, zmat_cond=None):
        nz, nm = M_mat_Delta.shape

        if mdef_Delta is None:
            mdef_Delta = self.other_params['pressure_model_mdef']

        if M200c_mat is None:
            if self.verbose:
                print('changing mdef to 200c for battaglia profiles in function get_Pe_mat_Battaglia12')

                ti = time.time()

            halo_conc_Delta = np.zeros(M_mat_Delta.shape)
            # pdb.set_trace()
            for j in range(len(z_array)):
                M_array = M_mat_Delta[j, :]
                halo_conc_Delta[j, :] = concentration.concentration(M_array, mdef_Delta, z_array[j])

            M200c_mat, R200c_mat_kpc_h = np.zeros(M_mat_Delta.shape), np.zeros(M_mat_Delta.shape)
            for j in range(nz):
                M200c_mat[j, :], R200c_mat_kpc_h[j, :], _ = mass_defs.changeMassDefinition(M_mat_Delta[j, :],
                                                                                           halo_conc_Delta[j, :],
                                                                                           z_array[j], mdef_Delta,
                                                                                           '200c')

            hydro_B = self.pressure_params_dict['hydro_mb']
            M200c_mat = M200c_mat / hydro_B

            if self.verbose:
                print('that took ', time.time() - ti, 'seconds')

        # print('M200c=' + str(np.log10(M200c_mat)))

        nz, nm = M_mat_Delta.shape
        nx = len(x_array)
        # units (Msun) / (Mpc ** 3)
        rho_crit_array = self.cosmo_colossus.rho_c(z_array) * (1000 ** 3) * (self.cosmo.h ** 2)
        rho_crit_mat = np.tile(rho_crit_array.reshape(nz, 1, 1), (1, nm, nx))
        M_mat_Delta_no_h = (M_mat_Delta / self.cosmo.h)
        R_mat_Delta_no_h = (R_mat_Delta / self.cosmo.h)
        M_mat_Delta_no_h_nz_nm_nx = np.tile(M_mat_Delta_no_h.reshape(nz, nm, 1), (1, 1, nx))
        R_mat_Delta_no_h_nz_nm_nx = np.tile(R_mat_Delta_no_h.reshape(nz, nm, 1), (1, 1, nx))
        coeff = (const.G * (const.M_sun ** 2) / ((1.0 * u.Mpc) ** 4)).to((u.eV / (u.cm ** 3))).value

        pressure_model_delta = float(''.join(list(mdef_Delta)[:-1]))
        P_Delta = (coeff / 2.) * pressure_model_delta * (
                self.cosmo.Ob0 / self.cosmo.Om0) * M_mat_Delta_no_h_nz_nm_nx * rho_crit_mat / R_mat_Delta_no_h_nz_nm_nx

        gnfw_P = self.get_gnfwp_Battaglia12(M200c_mat, x_array, z_array, Mmat_cond=Mmat_cond, zmat_cond=zmat_cond)
        M200c_mat_no_h = (M200c_mat / self.cosmo.h)
        M200c_mat_no_h_nz_nm_nx = np.tile(M200c_mat_no_h.reshape(nz, nm, 1), (1, 1, nx))

        # import pdb; pdb.set_trace()
        if self.pressure_model_type == 'lowbroken_powerlaw':
            M_mat_cond_shape_nz_nm = []
            M_mat_cond_shape_nz_nm.append(
                M200c_mat_no_h < (10 ** self.pressure_params_dict['logMstar']['alpha_p_low']))
            M_mat_cond_shape_nz_nm.append(
                M200c_mat_no_h > (10 ** self.pressure_params_dict['logMstar']['alpha_p_low']))

            alpha_p_array = np.zeros((nz, nm))
            for i in range(2):
                if i == 0:
                    alpha_p_array += self.pressure_params_dict['alpha_p_low'] * M_mat_cond_shape_nz_nm[i]
                else:
                    alpha_p_array += 0.0 * M_mat_cond_shape_nz_nm[i]

            alpha_p_mat = np.tile(alpha_p_array.reshape(nz, nm, 1), (1, 1, nx))
            M_mat_coeff = np.power(
                (M200c_mat_no_h_nz_nm_nx / (10 ** self.pressure_params_dict['logMstar']['alpha_p_low'])),
                alpha_p_mat)

        elif self.pressure_model_type == 'highbroken_powerlaw':
            M_mat_cond_shape_nz_nm = []
            M_mat_cond_shape_nz_nm.append(
                M200c_mat_no_h < (10 ** self.pressure_params_dict['logMstar']['alpha_p_high']))
            M_mat_cond_shape_nz_nm.append(
                M200c_mat_no_h > (10 ** self.pressure_params_dict['logMstar']['alpha_p_high']))

            alpha_p_array = np.zeros((nz, nm))
            for i in range(2):
                if i == 0:
                    alpha_p_array += self.pressure_params_dict['alpha_p_high'] * M_mat_cond_shape_nz_nm[i]
                else:
                    alpha_p_array += 0.0 * M_mat_cond_shape_nz_nm[i]

            alpha_p_mat = np.tile(alpha_p_array.reshape(nz, nm, 1), (1, 1, nx))
            M_mat_coeff = np.power(
                (M200c_mat_no_h_nz_nm_nx / (10 ** self.pressure_params_dict['logMstar']['alpha_p_high'])),
                alpha_p_mat)

        elif self.pressure_model_type == 'doublybroken_powerlaw':
            M_mat_cond_shape_nz_nm = []
            M_mat_cond_shape_nz_nm.append(
                M200c_mat_no_h < (10 ** self.pressure_params_dict['logMstar']['alpha_p_low']))
            M_mat_cond_shape_nz_nm.append(
                M200c_mat_no_h > (10 ** self.pressure_params_dict['logMstar']['alpha_p_low']))

            alpha_p_array = np.zeros((nz, nm))
            for i in range(2):
                if i == 0:
                    alpha_p_array += self.pressure_params_dict['alpha_p_low'] * M_mat_cond_shape_nz_nm[i]
                else:
                    alpha_p_array += 0.0 * M_mat_cond_shape_nz_nm[i]

            alpha_p_mat = np.tile(alpha_p_array.reshape(nz, nm, 1), (1, 1, nx))
            M_mat_coeff1 = np.power(
                (M200c_mat_no_h_nz_nm_nx / (10 ** self.pressure_params_dict['logMstar']['alpha_p_low'])),
                alpha_p_mat)

            M_mat_cond_shape_nz_nm = []
            M_mat_cond_shape_nz_nm.append(
                M200c_mat_no_h < (10 ** self.pressure_params_dict['logMstar']['alpha_p_high']))
            M_mat_cond_shape_nz_nm.append(
                M200c_mat_no_h > (10 ** self.pressure_params_dict['logMstar']['alpha_p_high']))

            alpha_p_array = np.zeros((nz, nm))
            for i in range(2):
                if i == 0:
                    alpha_p_array += self.pressure_params_dict['alpha_p_high'] * M_mat_cond_shape_nz_nm[i]
                else:
                    alpha_p_array += 0.0 * M_mat_cond_shape_nz_nm[i]

            alpha_p_mat = np.tile(alpha_p_array.reshape(nz, nm, 1), (1, 1, nx))
            M_mat_coeff2 = np.power(
                (M200c_mat_no_h_nz_nm_nx / (10 ** self.pressure_params_dict['logMstar']['alpha_p_high'])),
                alpha_p_mat)

            M_mat_coeff = M_mat_coeff1 * M_mat_coeff2

            # pdb.set_trace()

        else:
            M_mat_coeff = np.ones((nz, nm, nx))

        Pe_mat = 0.518 * P_Delta * gnfw_P * M_mat_coeff

        # np.savez('Pe_mat_comp_B12_zmax1.npz',Pe=Pe_mat,M200c_mat=M200c_mat,x=x_array,z=z_array,r500c=R_mat_Delta)
        # pdb.set_trace()
        return Pe_mat

    def get_Pe_mat(self, M_mat, x_array, z_array, R_mat, M200c_mat=None, mdef_M_mat=None, Mmat_cond=None,
                   zmat_cond=None):
        if self.pressure_model_name == 'Arnaud10':
            Pe_mat = self.get_Pe_mat_Arnaud10(M_mat, x_array, z_array, Mmat_cond=Mmat_cond, zmat_cond=zmat_cond)
        elif self.pressure_model_name == 'Battaglia12':
            Pe_mat = self.get_Pe_mat_Battaglia12(M_mat, x_array, z_array, R_mat, M200c_mat=M200c_mat,
                                                 mdef_Delta=mdef_M_mat, Mmat_cond=Mmat_cond, zmat_cond=zmat_cond)
        elif self.pressure_model_name == 'LeBrun15':
            Pe_mat = self.get_Pe_mat_LeBrun15(M_mat, x_array, z_array, R_mat,
                                              mdef_Delta=mdef_M_mat, Mmat_cond=Mmat_cond, zmat_cond=zmat_cond)
        else:
            print('provide correct Pe model')
            sys.exit(1)
        return Pe_mat

    # Get y3d as defined in Makiya et alpha_g eq 13
    # dimensions of output are: ?????
    def get_y3d(self, M_mat, x_array, z_array, R_mat, M200c_mat=None, mdef_M_mat=None, Mmat_cond=None, zmat_cond=None):
        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        coeff = sigmat / (m_e * (c ** 2))
        oneMpc_h = (((10 ** 6) / self.cosmo.h) * (u.pc).to(u.m)) * (u.m)
        const_coeff = ((coeff * oneMpc_h).to(((u.cm ** 3) / u.eV))).value

        Pe_mat = self.get_Pe_mat(M_mat, x_array, z_array, R_mat, M200c_mat=M200c_mat, mdef_M_mat=mdef_M_mat,
                                 Mmat_cond=Mmat_cond, zmat_cond=zmat_cond)

        y3d_mat_loc = const_coeff * Pe_mat

        return y3d_mat_loc

    # Integrate 3d y profile along line of sight to get 2d y profile
    def get_y2d_singleMz(self, M, rperp_array, z, M200c_mat=None, do_fast=False):
        y2d = np.zeros(len(rperp_array))
        M_mat = np.array([[M]])
        z_array = np.array([z])
        rhoDelta = self.pressure_model_delta * self.cosmo_colossus.rho_c(z) * (1000 ** 3)
        R_mat = [[np.power(3 * M / (4 * np.pi * rhoDelta), 1. / 3.)]]
        rpi_min = 0.0
        rpi_max = 10.0
        num_rpi = 100
        if do_fast:
            num_rpi = 20

        rpi_array = np.linspace(rpi_min, rpi_max, num_rpi)
        for ri in range(len(rperp_array)):
            x_array = np.sqrt(rperp_array[ri] ** 2. + rpi_array ** 2.)
            y3d = self.get_y3d(M_mat, x_array, z_array, R_mat, M200c_mat=M200c_mat)

            # factor of 2 since only doing half integral
            y2d[ri] = 2. * sp.integrate.simps(y3d, rpi_array)
        return y2d

    # Get integrated Y as function of mass and redshift
    def get_integratedY_singleMz(self, M, z, M200c_mat=None, do_fast=False):
        min_rperp = 0.0001
        max_rperp = 1.  # to get Y500
        num_rperp = 500
        if do_fast:
            num_rperp = 100

        rperp_array = np.linspace(min_rperp, max_rperp, num_rperp)

        y2d = self.get_y2d_singleMz(M, rperp_array, z, M200c_mat=M200c_mat, do_fast=do_fast)
        integratedY = 2. * np.pi * sp.integrate.simps(rperp_array * y2d, rperp_array)

        return integratedY

    # Get integrated Y as function of mass and redshift
    def get_Y500sph_singleMz(self, M500c, z, M200c=None, do_fast=False):
        minx = 0.001
        maxx = 1.  # to get Y500
        numx = 500
        if do_fast:
            numx = 100
        x_array = np.linspace(minx, maxx, numx)
        rho_crit = self.cosmo_colossus.rho_c(z) * (1000 ** 3)
        rmdefP = hmf.get_R_from_M(M500c, 500. * rho_crit)
        dang = hmf.get_Dang_com(z, self.cosmo.Om0)
        Ez = hmf.get_Ez(z, self.cosmo.Om0)
        M_mat = np.array([[M500c]])
        M200c_mat = np.array([[M200c]])
        z_array = np.array([z])
        rhoDelta = 500.0 * self.cosmo_colossus.rho_c(z) * (1000 ** 3)
        R_mat = np.array([[np.power(3 * M500c / (4 * np.pi * rhoDelta), 1. / 3.)]])
        # if M200c_mat is None:
        mdef_Delta = '500c'
        halo_conc_Delta = np.zeros(M_mat.shape)
        for j in range(len(z_array)):
            M_array = M_mat[j, :]
            halo_conc_Delta[j, :] = concentration.concentration(M_array, mdef_Delta, z_array[j])

        M200c_mat, R200c_mat_kpc_h = np.zeros(M_mat.shape), np.zeros(M_mat.shape)
        for j in range(len(z_array)):
            M200c_mat[j, :], R200c_mat_kpc_h[j, :], _ = mass_defs.changeMassDefinition(M_mat[j, :],
                                                                                       halo_conc_Delta[j, :],
                                                                                       z_array[j], mdef_Delta,
                                                                                       '200c')

        R200c_mat = R200c_mat_kpc_h / 1000.

        if self.pressure_model_name in ['LeBrun15', 'Arnaud10']:
            to_integrate = 4. * np.pi * (x_array ** 2.) * self.get_y3d(M_mat, x_array, z_array, R_mat,
                                                                       M200c_mat=M200c_mat, mdef_M_mat='500c')

        elif self.pressure_model_name == 'Battaglia12':
            to_integrate = 4. * np.pi * (x_array ** 2.) * self.get_y3d(M200c_mat, x_array, z_array, R200c_mat,
                                                                       M200c_mat=M200c_mat,
                                                                       mdef_M_mat='200c')
        else:
            print('provide correct Pe model')
            sys.exit(1)

        coeff = (((dang / (self.cosmo.h * 500.)) ** 2) / (Ez ** (2. / 3.))) * (180. * 60. / np.pi) ** 2
        # integrating dx so need extra R500**3.
        Y500sph = coeff * (rmdefP ** 3) * sp.integrate.simps(to_integrate, x_array) * (1. / dang ** 2)
        return Y500sph


class general_hm:

    def __init__(self, cosmo_params, pressure_params, other_params):
        cosmology.addCosmology('mock_cosmo', cosmo_params)
        self.cosmo_colossus = cosmology.setCosmology('mock_cosmo')
        h = cosmo_params['H0'] / 100.

        cosmo_func = cosmodef.mynew_cosmo(h, cosmo_params['Om0'], cosmo_params['Ob0'], cosmo_params['ns'],
                                          cosmo_params['sigma8'])
        self.cosmo = cosmo_func

        self.z_array = other_params['z_array']

        self.dndm_model = other_params['dndm_model']
        self.bias_model = other_params['bias_model']
        self.use_multiprocess = other_params['use_multiprocess']

        self.verbose = other_params['verbose']
        H0 = 100. * (u.km / (u.s * u.Mpc))
        G_new = const.G.to(u.Mpc ** 3 / ((u.s ** 2) * u.M_sun))
        self.rho_m_bar = ((cosmo_params['Om0'] * 3 * (H0 ** 2) / (8 * np.pi * G_new)).to(u.M_sun / (u.Mpc ** 3))).value
        # import pdb; pdb.set_trace()

    # Get the interpolated object of linear power spectrum at given cosmology
    def get_Pklin_zk_interp(self):
        k_array = np.logspace(-5, 2, 400)
        Pklinz_z0_test = hmf.get_Pklinz(0.0, k_array, current_cosmo=self.cosmo)
        sig8h = hmf.sigRz0(8., k_array, Pklinz_z0_test, window='tophat')
        sig8_ratio = ((self.cosmo.sig8 / sig8h) ** 2)

        zmin, zmax = np.min(self.z_array), np.max(self.z_array)
        z_array = np.logspace(np.log10(zmin), np.log10(zmax), 100)

        Pklinz_2d_mat = sig8_ratio * (hmf.get_Pklinzarray(z_array, k_array, current_cosmo=self.cosmo))
        # Pklinz_2d_mat = np.zeros((len(self.z_array), len(k_array)))
        # Pklinz0 = self.cosmo_colossus.matterPowerSpectrum(k_array, 0.0)
        # for j in range(len(self.z_array)):
        #     growthz = self.cosmo_colossus.growthFactor(self.z_array[j])
        #     Pklinz_2d_mat[j, :] = Pklinz0 * (growthz ** 2)

        Pklinz_2d_mat_interp = interpolate.RectBivariateSpline(np.log(z_array), np.log(k_array),
                                                               np.log(Pklinz_2d_mat))
        return Pklinz_2d_mat_interp

    # get the halo mass function and halo bias using the colossus module
    def get_dndm_bias(self, M_mat, mdef):

        dndm_array_Mz, bm_array_Mz = np.zeros(M_mat.shape), np.zeros(M_mat.shape)

        for j in range(len(self.z_array)):
            M_array = M_mat[j, :]
            dndm_array_Mz[j, :] = (1. / M_array) * mass_function.massFunction(M_array, self.z_array[j],
                                                                              mdef=mdef, model=self.dndm_model,
                                                                              q_out='dndlnM')

            bm_array_Mz[j, :] = bias.haloBias(M_array, self.z_array[j], model=self.bias_model, mdef=mdef)

        return dndm_array_Mz, bm_array_Mz

    # get the halo concentration using the colossus module
    def get_halo_conc_Mz(self, M_mat, mdef):
        halo_conc_array_Mz = np.zeros(M_mat.shape)

        if self.use_multiprocess:
            def get_halo_conc(ind, output_dict):
                M_array = M_mat[ind, :]
                output_dict[self.z_array[ind]] = concentration.concentration(M_array, mdef, self.z_array[ind])

            manager = multiprocessing.Manager()
            halo_conc_dict = manager.dict()
            if self.verbose:
                print('getting halo_conc for each z and M')
            starttime = time.time()
            processes = []
            for j in range(len(self.z_array)):
                p = multiprocessing.Process(target=get_halo_conc, args=(j, halo_conc_dict))
                processes.append(p)
                p.start()

            for process in processes:
                process.join()
            if self.verbose:
                print('That took {} seconds'.format(time.time() - starttime))

            for j in range(len(self.z_array)):
                halo_conc_array_Mz[j, :] = halo_conc_dict[self.z_array[j]]
        else:
            for j in range(len(self.z_array)):
                M_array = M_mat[j, :]
                halo_conc_array_Mz[j, :] = concentration.concentration(M_array, mdef, self.z_array[j])
                # import pdb;
                # pdb.set_trace()

        return halo_conc_array_Mz

    def get_wplin_interp(self, nu, pkzlin_interp):
        k_array = np.logspace(-5, 2.5, 30000)
        z_array = np.logspace(-3, 1, 100)
        Pklinz0 = np.exp(pkzlin_interp.ev(np.log(z_array[0]), np.log(k_array)))
        theta_out, xi_out = Hankel(k_array, nu=nu, q=1.0)(Pklinz0)
        xi_out *= (1 / (2 * np.pi))
        theta_out_arcmin = theta_out * (180. / np.pi) * 60.
        xi_mat = np.zeros((len(z_array), len(theta_out)))

        for j in range(len(z_array)):
            Pklinz = np.exp(pkzlin_interp.ev(np.log(z_array[j]), np.log(k_array)))
            theta_out, xi_out = Hankel(k_array, nu=nu, q=1.0)(Pklinz)
            xi_out *= (1 / (2 * np.pi))
            xi_mat[j, :] = xi_out

        wplin_mat_interp = interpolate.RectBivariateSpline(np.log(z_array), np.log(theta_out),
                                                           np.log(xi_mat))
        # import pdb; pdb.set_trace()
        return wplin_mat_interp


class Powerspec:

    def __init__(self, cosmo_params, hod_params, pressure_params, other_params):
        cosmology.addCosmology('mock_cosmo', cosmo_params)
        self.cosmo_colossus = cosmology.setCosmology('mock_cosmo')
        h = cosmo_params['H0'] / 100.
        cosmo_func = cosmodef.mynew_cosmo(h, cosmo_params['Om0'], cosmo_params['Ob0'], cosmo_params['ns'],
                                          cosmo_params['sigma8'])
        self.cosmo = cosmo_func

        self.M_array, self.z_array, self.x_array = other_params['M_array'], other_params['z_array'], other_params[
            'x_array']

        self.verbose = other_params['verbose']

        self.nm, self.nz = len(self.M_array), len(self.z_array)
        M_mat_mdef = np.tile(self.M_array.reshape(1, self.nm), (self.nz, 1))
        self.M_mat = M_mat_mdef

        if hod_params['hod_type'] == 'Halos':
            self.use_only_halos = True
        else:
            self.use_only_halos = False

        self.cmis = np.exp(hod_params['lncmis'])
        self.fmis = hod_params['fmis']

        self.mdef_analysis = other_params['mdef_analysis']
        self.dndm_model = other_params['dndm_model']
        self.bias_model = other_params['bias_model']

        self.eta_mb = hod_params['eta_mb']
        self.sig_lnM = hod_params['sig_lnM']
        self.rho_crit_array = self.cosmo_colossus.rho_c(self.z_array) * (1000 ** 3)
        # default output is in (Msun * h ** 2) / (kpc ** 3)
        # rho_crit converted to (Msun * h ** 2) / (Mpc ** 3)
        self.rho_vir_array = (mass_so.deltaVir(self.z_array)) * self.rho_crit_array

        if self.verbose:
            print('setting up hod')
        self.hod = HOD(hod_params)

        if self.verbose:
            print('setting up general hmf')
        self.ghmf = general_hm(cosmo_params, pressure_params, other_params)
        self.Pressure = Pressure(cosmo_params, pressure_params, other_params)

        if 'dndm_array' not in other_params.keys():
            if self.verbose:
                print('getting dndm and bm at', self.mdef_analysis)
            self.dndm_array, self.bm_array = self.ghmf.get_dndm_bias(M_mat_mdef, self.mdef_analysis)
            if self.verbose:
                print('getting halo conc at ', self.mdef_analysis)
            self.halo_conc_mdef = self.ghmf.get_halo_conc_Mz(M_mat_mdef, self.mdef_analysis)
        else:
            self.dndm_array, self.bm_array = other_params['dndm_array'], other_params['bm_array']
            self.halo_conc_mdef = other_params['halo_conc_mdef']

        hydro_B = pressure_params['hydro_mb']
        if self.mdef_analysis == other_params['pressure_model_mdef']:
            self.M_mat_mdefP = M_mat_mdef / hydro_B
            self.rmdefP_mat = hmf.get_R_from_M_mat(self.M_mat_mdefP,
                                                   other_params['pressure_model_delta'] * self.rho_crit_array)
        else:
            if self.verbose:
                print('changing mdef to mdefP for pressure')
            M_mat_mdefP, R_mat_mdefP_kpc_h = np.zeros(M_mat_mdef.shape), np.zeros(M_mat_mdef.shape)
            for j in range(self.nz):
                M_mat_mdefP[j, :], R_mat_mdefP_kpc_h[j, :], _ = mass_defs.changeMassDefinition(M_mat_mdef[j, :],
                                                                                               self.halo_conc_mdef[j,
                                                                                               :], self.z_array[j],
                                                                                               self.mdef_analysis,
                                                                                               other_params[
                                                                                                   'pressure_model_mdef'])

            self.M_mat_mdefP = M_mat_mdefP / hydro_B
            self.rmdefP_mat = hmf.get_R_from_M_mat(self.M_mat_mdefP,
                                                   other_params['pressure_model_delta'] * self.rho_crit_array)

        # if other_params['pressure_model_name'] == 'Battaglia12':
        if self.verbose:
            print('changing mdef to 200c for battaglia profiles')
        M_mat_200cP, R_mat_200cP_kpc_h = np.zeros(M_mat_mdef.shape), np.zeros(M_mat_mdef.shape)
        for j in range(self.nz):
            M_mat_200cP[j, :], R_mat_200cP_kpc_h[j, :], _ = mass_defs.changeMassDefinition(M_mat_mdef[j, :],
                                                                                           self.halo_conc_mdef[j,
                                                                                           :], self.z_array[j],
                                                                                           self.mdef_analysis,
                                                                                           '200c')

        self.M_mat_200cP = M_mat_200cP / hydro_B
        self.r200cP_mat = hmf.get_R_from_M_mat(self.M_mat_200cP, 200 * self.rho_crit_array)

        #
        # self.rmdefP_mat = R_mat_mdefP_kpc_h / 1000.

        # print 'getting y3d matrix'
        # self.y3d_mat = self.ghmf.get_y3d(M_mat_mdefP, self.x_array, self.z_array)

        if self.mdef_analysis == 'vir':
            M_mat_vir = M_mat_mdef
            self.r_vir_mat = hmf.get_R_from_M_mat(M_mat_mdef, self.rho_vir_array)
            self.halo_conc_vir = self.halo_conc_mdef
        else:
            if self.verbose:
                print('changing mdef to vir for nfw')
            M_mat_vir, R_mat_vir_kpc_h, halo_conc_vir = np.zeros(M_mat_mdef.shape), np.zeros(
                M_mat_mdef.shape), np.zeros(M_mat_mdef.shape)
            for j in range(self.nz):
                M_mat_vir[j, :], R_mat_vir_kpc_h[j, :], halo_conc_vir[j, :] = mass_defs.changeMassDefinition(
                    M_mat_mdef[j, :], self.halo_conc_mdef[j, :], self.z_array[j], self.mdef_analysis, 'vir')
            self.r_vir_mat = R_mat_vir_kpc_h / 1000.
            self.halo_conc_vir = halo_conc_vir

        if not self.use_only_halos:
            self.rmax_r200c = hod_params['rmax_r200c']
            self.rmax_rvir = hod_params['rmax_rvir']
            self.rsg_rs = hod_params['rsg_rs']
            self.mass_def_for_rmax = hod_params['mass_def_for_rmax']
            if self.mass_def_for_rmax == 'vir':
                self.r_max_mat = self.r_vir_mat * self.rmax_rvir
            elif self.mass_def_for_rmax == '200c':
                if self.verbose:
                    print('changing mdef to 200c for extent')
                M_mat_200c, R_mat_200c_kpc_h, halo_conc_200c = np.zeros(M_mat_mdef.shape), np.zeros(
                    M_mat_mdef.shape), np.zeros(M_mat_mdef.shape)
                for j in range(self.nz):
                    M_mat_200c[j, :], R_mat_200c_kpc_h[j, :], halo_conc_200c[j, :] = mass_defs.changeMassDefinition(
                        M_mat_mdef[j, :], self.halo_conc_mdef[j, :], self.z_array[j], self.mdef_analysis, '200c')
                self.r_200c_mat = R_mat_200c_kpc_h / 1000.
                self.halo_conc_200c = halo_conc_200c
                self.r_max_mat = self.r_200c_mat * self.rmax_r200c
            else:
                self.r_max_mat = None

        self.Nc_mat = self.hod.get_Nc(M_mat_vir)
        self.Ns_mat = self.hod.get_Ns(M_mat_vir)
        self.Ntotal_mat = self.hod.get_Ntotal(M_mat_vir)
        self.M_mat_vir = M_mat_vir
        # pdb.set_trace()

        self.mass_bias_type = other_params['mass_bias_type']

        z_mat = np.tile(self.z_array.reshape(self.nz, 1), (1, self.nm))
        self.z_mat_cond_inbin = np.logical_and(z_mat >= other_params['zmin_tracer'],
                                               z_mat <= other_params['zmax_tracer'])

        self.z_array_cond_inbin = np.logical_and(self.z_array >= other_params['zmin_tracer'],
                                                 self.z_array <= other_params['zmax_tracer'])

        if self.mass_bias_type == 'limits':
            self.M_mat_cond_inbin = np.logical_and(
                self.M_mat_mdefP >= self.eta_mb * (10 ** other_params['log_M_min_tracer']),
                self.M_mat_mdefP <= self.eta_mb * (10 ** other_params['log_M_max_tracer']))

        else:
            self.M_mat_cond_inbin = np.logical_and(self.M_mat_mdefP >= (10 ** other_params['log_M_min_tracer']),
                                                   self.M_mat_mdefP <= (10 ** other_params['log_M_max_tracer']))

        if self.mass_bias_type == 'weighted':
            xL = (np.log(self.eta_mb * (10 ** other_params['log_M_min_tracer'])) - np.log(M_mat_vir)) / (
                    2 * (self.sig_lnM ** 2))
            xH = (np.log(self.eta_mb * (10 ** other_params['log_M_max_tracer'])) - np.log(M_mat_vir)) / (
                    2 * (self.sig_lnM ** 2))
            self.int_prob = 0.5 * (spsp.erf(xL) - spsp.erf(xH))
            self.M_mat_cond_inbin = np.ones(self.M_mat_cond_inbin.shape)
            # pdb.set_trace()
        else:
            self.int_prob = np.ones(self.Ntotal_mat.shape)
            # M_mat_cond_inbin_nbar = self.M_mat_cond_inbin

        # pdb.set_trace()
        if self.use_only_halos:
            self.nbar = hmf.get_nbar_z(self.M_array, self.dndm_array, self.int_prob, self.M_mat_cond_inbin)
        else:
            self.nbar = hmf.get_nbar_z(self.M_array, self.dndm_array, self.Ntotal_mat, self.M_mat_cond_inbin)
        self.nbar_mat = np.tile(self.nbar.reshape(self.nz, 1), (1, self.nm))

        ng_zarray = other_params['ng_zarray']
        ng_value = other_params['ng_value']

        ng_interp = interpolate.interp1d(ng_zarray, np.log(ng_value + 1e-100), fill_value='extrapolate')

        self.chi_array = hmf.get_Dcom_array(self.z_array, cosmo_params['Om0'])
        self.DA_array = self.chi_array / (1. + self.z_array)
        self.ng_array = np.exp(ng_interp(self.z_array))
        self.dchi_dz_array = (const.c.to(u.km / u.s)).value / (hmf.get_Hz(self.z_array, cosmo_params['Om0']))

        ng_zarray_source = other_params['ng_zarray_source']
        ng_value_source = other_params['ng_value_source']
        # import pdb; pdb.set_trace()
        if np.any(ng_value_source < 0.0):
            ng_interp_source = interpolate.interp1d(ng_zarray_source, ng_value_source,
                                                    fill_value=0.0, bounds_error=False)
            self.ng_array_source = ng_interp_source(self.z_array)
        else:
            ng_interp_source = interpolate.interp1d(ng_zarray_source, np.log(ng_value_source + 1e-40),
                                                    fill_value='extrapolate')
            self.ng_array_source = np.exp(ng_interp_source(self.z_array))

        chi_lmat = np.tile(self.chi_array.reshape(len(self.z_array), 1), (1, len(self.z_array)))
        chi_smat = np.tile(self.chi_array.reshape(1, len(self.z_array)), (len(self.z_array), 1))
        num = chi_smat - chi_lmat
        ind_lzero = np.where(num <= 0)
        num[ind_lzero] = 0
        ng_array_source_rep = np.tile(self.ng_array_source.reshape(1, len(self.z_array)), (len(self.z_array), 1))
        int_sourcez = sp.integrate.simps(ng_array_source_rep * (num / chi_smat), self.z_array)
        coeff_ints = 3 * (100 ** 2) * cosmo_params['Om0'] / (2. * ((3 * 10 ** 5) ** 2))
        self.Wk_array = coeff_ints * self.chi_array * (1. + self.z_array) * int_sourcez
        H0 = 100. * (u.km / (u.s * u.Mpc))
        G_new = const.G.to(u.Mpc ** 3 / ((u.s ** 2) * u.M_sun))
        self.rho_m_bar = ((cosmo_params['Om0'] * 3 * (H0 ** 2) / (8 * np.pi * G_new)).to(u.M_sun / (u.Mpc ** 3))).value
        # self.binv = other_params['binv']
        if 'pkzlin_interp' not in other_params.keys():
            if self.verbose:
                print('getting pkzlin interp')
            self.pkzlin_interp = self.ghmf.get_Pklin_zk_interp()
        else:
            self.pkzlin_interp = other_params['pkzlin_interp']

        self.add_beam_to_theory = other_params['add_beam_to_theory']
        self.beam_fwhm_arcmin = other_params['beam_fwhm_arcmin']
        if 'um_block_allinterp' in other_params.keys():
            self.um_block_allinterp = other_params['um_block_allinterp']
            self.bkm_block_allinterp = other_params['bkm_block_allinterp']

        if other_params['get_bp']:
            self.wplin_interp = other_params['wplin_interp']

        # pdb.set_trace()

    def get_xi_kappy_2h(self, theta_arcmin, bpz_keVcm3=1e-7, bpz0_keVcm3=1e-7,bpalpha=3.0, bp_model='const'):
        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        coeff = sigmat / (m_e * (c ** 2))
        oneMpc_h = (((10 ** 6) / self.cosmo.h) * (u.pc).to(u.m)) * (u.m)
        const_coeff = ((coeff * oneMpc_h).to(((u.cm ** 3) / u.keV))).value
        theta_rad = theta_arcmin * (np.pi / 180.) * (1. / 60.)
        wp_chiz = np.zeros_like(self.z_array)

        for j in range(len(self.z_array)):
            wp_chiz[j] = np.exp(
                self.wplin_interp.ev(np.log(self.z_array[j]), np.log(self.chi_array[j] * theta_rad)))

        if bp_model == 'linear':
            bp_keVcm3 = bpz0_keVcm3 + bpalpha * self.z_array
        if bp_model == 'const':
            bp_keVcm3 = bpz_keVcm3
        int_val = bp_keVcm3 * self.dchi_dz_array * (self.Wk_array / (1. + self.z_array)) * wp_chiz
        value = const_coeff * sp.integrate.simps(int_val, self.z_array)
        # print('zmean=' + str(sp.integrate.simps(self.dchi_dz_array * self.Wk_array, self.z_array)))
        # import pdb; pdb.set_trace()
        return value

    # get spherical harmonic transform of the galaxy distribution, eq 18 of Makiya et al
    def get_ug_l_zM(self, l):
        k_array = (l + 1. / 2.) / self.chi_array
        if self.use_only_halos:
            val = np.ones(self.Nc_mat.shape)
        else:
            ukzm_mat = hmf.get_ukmz_g_mat(self.r_max_mat, k_array, self.halo_conc_vir, self.rsg_rs)
            val = np.sqrt(
                (2 * self.Nc_mat * self.Ns_mat * ukzm_mat + self.Nc_mat * (self.Ns_mat ** 2) * (ukzm_mat ** 2)))

        coeff_mat = np.tile(
            (self.ng_array / ((self.chi_array ** 2) * self.dchi_dz_array * self.nbar)).reshape(self.nz, 1),
            (1, self.nm))
        # import pdb; pdb.set_trace()
        return coeff_mat * val

    # get spherical harmonic transform of the effective galaxy bias, eq 23 of Makiya et al
    def get_bg_l_z(self, l):
        k_array = (l + 1. / 2.) / self.chi_array

        if self.use_only_halos:
            toint = self.dndm_array * self.bm_array * self.M_mat_cond_inbin * self.int_prob
        else:
            ukzm_mat = hmf.get_ukmz_g_mat(self.r_max_mat, k_array, self.halo_conc_vir, self.rsg_rs)
            toint = self.dndm_array * (self.Nc_mat + self.Ns_mat * ukzm_mat) * self.bm_array * self.M_mat_cond_inbin

        val = sp.integrate.simps(toint, self.M_array)

        coeff = self.ng_array / ((self.chi_array ** 2) * self.dchi_dz_array * self.nbar)
        return coeff * val

    # # get spherical harmonic transform of the hot gas distribution, eq 12 of Makiya et al
    def get_uy_l_zM(self, l):
        temp_mat = l * x_mat_lmdefP_mat
        # val = sp.integrate.romb(x_mat2_y3d_mat * np.sin(temp_mat) / temp_mat,
        #                         self.x_array[1] - self.x_array[0])
        val = sp.integrate.simps(x_mat2_y3d_mat * np.sin(temp_mat) / temp_mat, self.x_array)

        if self.add_beam_to_theory and (self.beam_fwhm_arcmin > 0):
            sig_beam = self.beam_fwhm_arcmin * (1. / 60.) * (np.pi / 180.) * (1. / np.sqrt(8. * np.log(2)))
            Bl = np.exp(-1. * l * (l + 1) * (sig_beam ** 2) / 2.)
            val = val * Bl

        return coeff_mat_y * val

        # get spherical harmonic transform of the effective hot gas bias, eq 16 of Makiya et al

    def get_by_l_z(self, l, uyl_zM_dict):
        uyl_zM = uyl_zM_dict[round(l, 1)]
        toint = uyl_zM * self.dndm_array * self.bm_array
        val = sp.integrate.simps(toint, self.M_array)
        return val

    # get spherical harmonic transform of the matter distribution
    def get_uk_l_zM(self, l, uml_zM_dict):
        # k_array = (l + 1. / 2.) / self.chi_array
        # uk_mat_normed = np.zeros((self.nz, self.nm))
        # if hasattr(self,'um_block_kinterp'):
        #     for j in range(len(k_array)):
        #         kv = k_array[j]
        #         ukmz = self.um_block_kinterp(np.log(kv))
        #         uk_mat_normed[j,:] = ukmz[j,:]
        # else:
        #     ukzm_mat = hmf.get_ukmz_g_mat(self.r_max_mat, k_array, self.halo_conc_vir, self.rsg_rs)
        #     uk_mat_normed = ukzm_mat * self.M_mat / self.rho_m_bar
        um_mat_normed = uml_zM_dict[round(l, 1)]
        coeff_mat = np.tile((self.Wk_array / self.chi_array ** 2).reshape(self.nz, 1), (1, self.nm))
        return coeff_mat * um_mat_normed

    # get spherical harmonic transform of the matter distribution
    def get_um_l_zM(self, l):
        k_array = (l + 1. / 2.) / self.chi_array
        if hasattr(self, 'um_block_allinterp'):
            ukzm_mat = np.zeros((self.nz, self.nm))
            marray_rs = np.log(np.reshape(self.M_array, (1, self.nm, 1)))
            for j in range(len(k_array)):
                kv = k_array[j]
                # print(j,kv,self.z_array[j])
                marray_insz = np.insert(marray_rs, 0, (self.z_array[j]), axis=-1)
                marray_insk = np.insert(marray_insz, 2, np.log(kv), axis=-1)[0]
                ukzm_mat[j, :] = np.exp(self.um_block_allinterp(marray_insk))
                # if l > 200:
                #     import pdb;
                #     pdb.set_trace()
            # print(l)

            # uk_mat_normed[j,:] = ukmz[j,:]
        else:
            ukzm_mat = hmf.get_ukmz_g_mat(self.r_max_mat, k_array, self.halo_conc_vir, self.rsg_rs)

        # if l > 1000:
        #     import pdb; pdb.set_trace()
        uk_mat_normed = ukzm_mat * self.M_mat / self.rho_m_bar

        return uk_mat_normed

    # # get spherical harmonic transform of the effective shear bias
    # def get_bk_l_z(self, l, ukl_zM_dict):
    #     return self.Wk_array / self.chi_array ** 2

    # get spherical harmonic transform of the effective shear bias
    def get_bk_l_z(self, l, bm_l_z_dict):
        bm_l_z = bm_l_z_dict[round(l, 1)]
        coeff = (self.Wk_array / self.chi_array ** 2)
        return coeff * bm_l_z

    # get spherical harmonic transform of the effective shear bias
    # def get_bm_l_z(self, l, uml_zM_dict):
    #     uml_zM = uml_zM_dict[round(l, 1)]
    #     toint = uml_zM * self.dndm_array * self.bm_array
    #     val = sp.integrate.simps(toint, self.M_array)
    #     if hasattr(self, 'bkm_block_allinterp'):
    #         k_array = (l + 1. / 2.) / self.chi_array
    #         import pdb; pdb.set_trace()
    #         val *= np.exp(self.bkm_block_allinterp(np.stack(((self.z_array),np.log(k_array)), axis=-1)))
    #
    #     return val

    def get_bm_l_z(self, l):

        val = np.ones_like(self.z_array)
        if hasattr(self, 'bkm_block_allinterp'):
            k_array = (l + 1. / 2.) / self.chi_array
            # import pdb; pdb.set_trace()
            val *= np.exp(self.bkm_block_allinterp(np.stack(((self.z_array), np.log(k_array)), axis=-1)))

        return val

    def collect_ug(self, l_array, return_dict):
        for l in l_array:
            return_dict[round(l, 1)] = self.get_ug_l_zM(l)

    def collect_uy(self, l_array, return_dict):
        for l in l_array:
            return_dict[round(l, 1)] = self.get_uy_l_zM(l)

    def collect_uk(self, l_array, return_dict):
        for l in l_array:
            return_dict[round(l, 1)] = self.get_uk_l_zM(l)


class PrepDataVec:

    def __init__(self, cosmo_params, hod_params, pressure_params, other_params):

        self.PS = Powerspec(cosmo_params, hod_params, pressure_params, other_params)

        self.save_suffix = other_params['save_suffix']
        self.verbose = other_params['verbose']

        dl_array = other_params['dl_array']
        l_array = other_params['l_array']

        self.l_array = l_array
        self.nl = len(l_array)
        self.dl_array = dl_array

        self.ind_select_survey = np.where(
            (l_array >= other_params['lmin_survey']) & (l_array <= other_params['lmax_survey']))
        self.l_array_survey = l_array[self.ind_select_survey]
        self.nl_survey = len(self.l_array_survey)
        self.dl_array_survey = dl_array[self.ind_select_survey]

        # pdb.set_trace()

        self.fsky_gg = other_params['fsky_gg']
        self.fsky_yy = other_params['fsky_yy']
        # self.fsky_yg = np.sqrt(self.fsky_gg * self.fsky_yy)
        self.fsky_yg = other_params['fsky_yg']
        self.fsky_yk = other_params['fsky_yk']
        self.fsky_kk = other_params['fsky_kk']

        self.fsky = {'gg': self.fsky_gg, 'yy': self.fsky_yy, 'yg': self.fsky_yg, 'gy': self.fsky_yg, 'yk': self.fsky_yk,
                     'ky': self.fsky_yk, 'kk': self.fsky_kk}

        self.stats_analyze = other_params['stats_analyze']
        stats_analyze_pairs = []

        stats_analyze_pairs_all = []

        index_params = range(len(self.stats_analyze))
        for j1 in index_params:
            for j2 in index_params:
                if j2 >= j1:
                    stats_analyze_pairs.append([self.stats_analyze[j1], self.stats_analyze[j2]])

                stats_analyze_pairs_all.append([self.stats_analyze[j1], self.stats_analyze[j2]])

        self.stats_analyze_pairs = stats_analyze_pairs
        self.stats_analyze_pairs_all = stats_analyze_pairs_all

        stat_analyze_temp = ''
        for j in range(len(self.stats_analyze)):
            stat_analyze_temp += self.stats_analyze[j]
        self.lss_probes_analyze = list(set(list(stat_analyze_temp)))
        self.lss_probes_allcomb = []
        for js1 in range(len(self.lss_probes_analyze)):
            for js2 in range(len(self.lss_probes_analyze)):
                self.lss_probes_allcomb.append(self.lss_probes_analyze[js1] + self.lss_probes_analyze[js2])

        if 'uyl_zM_dict' in other_params.keys():
            uyl_zM_dict = other_params['uyl_zM_dict']
        else:
            ti = time.time()
            if self.verbose:
                print('getting y3d matrix')
            global x_mat2_y3d_mat, x_mat_lmdefP_mat, coeff_mat_y
            y3d_mat = self.PS.Pressure.get_y3d(self.PS.M_mat_mdefP, self.PS.x_array, self.PS.z_array,
                                               self.PS.rmdefP_mat,
                                               M200c_mat=self.PS.M_mat_200cP, Mmat_cond=self.PS.M_mat_cond_inbin,
                                               zmat_cond=self.PS.z_mat_cond_inbin)

            # Get YM relation from the above class
            # M500_array = np.logspace(12,15,20)
            # Y500_array = np.zeros_like(M500_array)
            # for jM in range(len(M500_array)):
            #     Y500_array[jM] = self.PS.Pressure.get_Y500sph_singleMz(M500_array[jM], 0.001, do_fast=False)
            # np.savez('YM_LeBrun_Ref.npz',M=M500_array,Y500=Y500_array)
            # import pdb; pdb.set_trace()

            if self.verbose:
                print('getting x matrix')
            nz, nm, nx = len(self.PS.z_array), len(self.PS.M_array), len(self.PS.x_array)
            x_mat = np.tile(self.PS.x_array.reshape(1, 1, nx), (nz, nm, 1))

            if self.verbose:
                print('lmdefP matrix')
            DA_mat_coeff = np.tile(self.PS.DA_array.reshape(nz, 1), (1, nm))
            rmdefP_mat_coeff = self.PS.rmdefP_mat
            lmdefP_mat_coeff = DA_mat_coeff / rmdefP_mat_coeff

            x_mat_lmdefP_mat = x_mat / np.tile(lmdefP_mat_coeff.reshape(nz, nm, 1), (1, 1, nx))
            coeff_mat_y = 4 * np.pi * rmdefP_mat_coeff / (lmdefP_mat_coeff ** 2)

            x_mat2_y3d_mat = (x_mat ** 2) * y3d_mat

            del x_mat, y3d_mat

            if self.verbose:
                print('that took ', time.time() - ti, 'seconds')

        if 'uml_zM_dict' in other_params.keys():
            uml_zM_dict = other_params['uml_zM_dict']

        if other_params['use_multiprocess']:

            manager = multiprocessing.Manager()
            # pdb.set_trace()

            ugl_zM_dict = manager.dict()

            if self.verbose:
                print('getting ugl matrix for each z and M')
            starttime = time.time()
            processes = []
            if other_params['num_pool'] is None:
                for j in range(len(l_array)):
                    p = multiprocessing.Process(target=self.PS.collect_ug, args=([l_array[j]], ugl_zM_dict))
                    processes.append(p)
                    p.start()
            else:
                npool = other_params['num_pool']
                l_array_split = np.array_split(l_array, npool)
                for j in range(npool):
                    p = multiprocessing.Process(target=self.PS.collect_ug, args=(l_array_split[j], ugl_zM_dict))
                    processes.append(p)
                    p.start()

            for process in processes:
                process.join()

            if self.verbose:
                print('That took {} seconds'.format(time.time() - starttime))

            if 'uyl_zM_dict' not in other_params.keys():
                uyl_zM_dict = manager.dict()

                if self.verbose:
                    print('getting uyl matrix for each z and M')
                starttime = time.time()
                processes = []

                if other_params['num_pool'] is None:
                    for j in range(len(l_array)):
                        p = multiprocessing.Process(target=self.PS.collect_uy, args=([l_array[j]], uyl_zM_dict))
                        processes.append(p)
                        p.start()
                else:
                    npool = other_params['num_pool']
                    l_array_split = np.array_split(l_array, npool)
                    for j in range(npool):
                        p = multiprocessing.Process(target=self.PS.collect_uy, args=(l_array_split[j], uyl_zM_dict))
                        processes.append(p)
                        p.start()

                for process in processes:
                    process.join()
                print('That took {} seconds'.format(time.time() - starttime))

            ukl_zM_dict = manager.dict()

            if self.verbose:
                print('getting ukl matrix for each z and M')
            starttime = time.time()
            processes = []

            if other_params['num_pool'] is None:
                for j in range(len(l_array)):
                    p = multiprocessing.Process(target=self.PS.collect_uk, args=([l_array[j]], ukl_zM_dict))
                    processes.append(p)
                    p.start()
            else:
                npool = other_params['num_pool']
                l_array_split = np.array_split(l_array, npool)
                for j in range(npool):
                    p = multiprocessing.Process(target=self.PS.collect_uk, args=(l_array_split[j], ukl_zM_dict))
                    processes.append(p)
                    p.start()

            for process in processes:
                process.join()
            print('That took {} seconds'.format(time.time() - starttime))


        else:
            if self.verbose:
                print('getting uyl and ugl matrix')
                ti = time.time()

            if 'uyl_zM_dict' not in other_params.keys():
                uyl_zM_dict = {}
                for j in range(len(l_array)):
                    uyl_zM_dict[round(l_array[j], 1)] = self.PS.get_uy_l_zM(l_array[j])

            if 'uml_zM_dict' not in other_params.keys():
                uml_zM_dict = {}
                if 'k' in self.lss_probes_analyze:
                    for j in range(len(l_array)):
                        uml_zM_dict[round(l_array[j], 1)] = self.PS.get_um_l_zM(l_array[j])

            ugl_zM_dict, ukl_zM_dict = {}, {}
            for j in range(len(l_array)):
                if 'g' in self.lss_probes_analyze:
                    ugl_zM_dict[round(l_array[j], 1)] = self.PS.get_ug_l_zM(l_array[j])
                if 'k' in self.lss_probes_analyze:
                    ukl_zM_dict[round(l_array[j], 1)] = self.PS.get_uk_l_zM(l_array[j], uml_zM_dict)

            if self.verbose:
                print('that took ', time.time() - ti, 'seconds')

            if 'byl_z_dict' in other_params.keys():
                self.byl_z_dict = other_params['byl_z_dict']
            else:
                self.byl_z_dict = {}
                for j in range(len(l_array)):
                    self.byl_z_dict[round(l_array[j], 1)] = self.PS.get_by_l_z(l_array[j], uyl_zM_dict)

            if 'bml_z_dict' in other_params.keys():
                self.bml_z_dict = other_params['bml_z_dict']
            else:
                self.bml_z_dict = {}
                for j in range(len(l_array)):
                    if 'k' in self.lss_probes_analyze:
                        self.bml_z_dict[round(l_array[j], 1)] = self.PS.get_bm_l_z(l_array[j])

            self.bgl_z_dict, self.bkl_z_dict = {}, {}
            for j in range(len(l_array)):
                if 'g' in self.lss_probes_analyze:
                    self.bgl_z_dict[round(l_array[j], 1)] = self.PS.get_bg_l_z(l_array[j])
                if 'k' in self.lss_probes_analyze:
                    self.bkl_z_dict[round(l_array[j], 1)] = self.PS.get_bk_l_z(l_array[j], self.bml_z_dict)
            if self.verbose:
                print('done getting all the uk and bk')

        self.ugl_zM_dict = ugl_zM_dict
        self.uyl_zM_dict = uyl_zM_dict
        self.ukl_zM_dict = ukl_zM_dict
        self.uml_zM_dict = uml_zM_dict
        if other_params['noise_Cl_filename'] is not None:

            self.noise_yy_Cl_file = np.loadtxt(other_params['noise_Cl_filename'])
            if 'S4' in other_params['noise_Cl_filename'].split('/'):
                l_noise_yy_file, Cl_noise_yy_file = self.noise_yy_Cl_file[:, 0], self.noise_yy_Cl_file[:, 2]
                log_Cl_noise_yy_interp = interpolate.interp1d(np.log(l_noise_yy_file), np.log(Cl_noise_yy_file),
                                                              fill_value='extrapolate')
            if ('SO' in other_params['noise_Cl_filename'].split('/')) or (
                    'Planck' in other_params['noise_Cl_filename'].split('/') or
                    'ACT' in other_params['noise_Cl_filename'].split('/')):
                l_noise_yy_file, Cl_noise_yy_file = self.noise_yy_Cl_file[:, 0], self.noise_yy_Cl_file[:, 1]
                log_Cl_noise_yy_interp = interpolate.interp1d(np.log(l_noise_yy_file), np.log(Cl_noise_yy_file),
                                                              fill_value='extrapolate')
            # pdb.set_trace()
            self.Cl_noise_yy_l_array = np.exp(log_Cl_noise_yy_interp(np.log(self.l_array_survey)))
            self.Cl_noise_gg_l_array = (1. / other_params['nbar']) * np.ones(self.nl_survey)
            self.Cl_noise_kk_l_array = (other_params['noise_kappa']) * np.ones(self.nl_survey)
        else:
            self.ell_measured = other_params['ell_measured']
            if len(self.Cl_dict['yy']['total']) == len(self.ell_measured):
                self.Cl_noise_yy_l_array = other_params['Clyy_measured'] - self.Cl_dict['yy']['total']
                self.Cl_noise_gg_l_array = other_params['Clgg_measured'] - self.Cl_dict['gg']['total']
            else:
                Cl_yy_interp = interpolate.interp1d(np.log(self.ell_measured), other_params['Clyy_measured'],
                                                    fill_value=0.0,
                                                    bounds_error=False)
                self.Cl_noise_yy_l_array = Cl_yy_interp(np.log(self.l_array)) - self.Cl_dict['yy']['total']
                Cl_gg_interp = interpolate.interp1d(np.log(self.ell_measured), other_params['Clgg_measured'],
                                                    fill_value=0.0,
                                                    bounds_error=False)
                self.Cl_noise_gg_l_array = Cl_gg_interp(np.log(self.l_array)) - self.Cl_dict['gg']['total']

        # import pdb; pdb.set_trace()
        if 'uyl_zM_dict' not in other_params.keys():
            del x_mat2_y3d_mat, x_mat_lmdefP_mat, coeff_mat_y



        if self.verbose:
            print('finished prep of DV')



class CalcDataVec:

    def __init__(self, PrepDV_params):
        self.PS_prepDV = PrepDV_params['PrepDV_fid'].PS

    def get_Cl_AB_1h(self, A, B, l_array, uAl_zM_dict, uBl_zM_dict):
        g_sum = (A == 'g') + (B == 'g')
        if g_sum == 2:
            if self.PS_prepDV.use_only_halos:
                return 0
            else:
                toint_M_multfac = self.PS_prepDV.M_mat_cond_inbin
                toint_z_multfac = self.PS_prepDV.z_array_cond_inbin
        elif g_sum == 1:
            toint_M_multfac = self.PS_prepDV.M_mat_cond_inbin * self.PS_prepDV.int_prob
            toint_z_multfac = self.PS_prepDV.z_array_cond_inbin
        else:
            toint_M_multfac = 1.
            toint_z_multfac = 1.
        Cl_1h = np.zeros_like(l_array)
        for j in range(len(l_array)):
            l = l_array[j]
            uAl_zM = uAl_zM_dict[round(l, 1)]
            uBl_zM = uBl_zM_dict[round(l, 1)]
            toint_M = (uAl_zM * uBl_zM) * self.PS_prepDV.dndm_array * toint_M_multfac
            val_z = sp.integrate.simps(toint_M, self.PS_prepDV.M_array)
            toint_z = val_z * (self.PS_prepDV.chi_array ** 2) * self.PS_prepDV.dchi_dz_array * toint_z_multfac
            val = sp.integrate.simps(toint_z, self.PS_prepDV.z_array)
            Cl_1h[j] = val
        return Cl_1h

    def get_Cl_AB_2h(self, A, B, l_array, bAl_z_dict, bBl_z_dict):
        g_sum = (A == 'g') + (B == 'g')
        if g_sum > 0:
            toint_z_multfac = self.PS_prepDV.z_array_cond_inbin
        else:
            toint_z_multfac = 1.
        Cl_2h = np.zeros_like(l_array)
        for j in range(len(l_array)):
            l = l_array[j]
            k_array = (l + 1. / 2.) / self.PS_prepDV.chi_array
            bgl_z1 = bAl_z_dict[round(l, 1)]
            bgl_z2 = bBl_z_dict[round(l, 1)]
            toint_z = (bgl_z1 * bgl_z2) * (self.PS_prepDV.chi_array ** 2) * self.PS_prepDV.dchi_dz_array * np.exp(
                self.PS_prepDV.pkzlin_interp.ev(np.log(self.PS_prepDV.z_array), np.log(k_array))) * toint_z_multfac
            val = sp.integrate.simps(toint_z, self.PS_prepDV.z_array)
            Cl_2h[j] = val
        return Cl_2h

    def get_Cl_AB_tot(self, A, B, ClAB_1h, ClAB_2h):
        g_sum = (A == 'g') + (B == 'g')
        if (g_sum == 1) and self.PS_prepDV.use_only_halos and (self.PS_prepDV.fmis > 0):
            Cl_AB_tot = self.get_Cl_yg_miscentered(l_array, ClAB_1h + ClAB_2h)
        else:
            Cl_AB_tot = ClAB_1h + ClAB_2h
        return Cl_AB_tot

    # See Makiya paper
    def get_T_ABCD_NG(self, l_array_all, A, B, C, D, uAl_zM_dict, uBl_zM_dict, uCl_zM_dict, uDl_zM_dict):
        nl = len(l_array_all)

        ul_A_mat, ul_B_mat, ul_C_mat, ul_D_mat = np.zeros((nl, self.PS_prepDV.nz, self.PS_prepDV.nm)), np.zeros(
            (nl, self.PS_prepDV.nz, self.PS_prepDV.nm)), np.zeros((nl, self.PS_prepDV.nz, self.PS_prepDV.nm)), np.zeros(
            (nl, self.PS_prepDV.nz, self.PS_prepDV.nm))
        for j in range(nl):
            ul_A_mat[j, :, :] = uAl_zM_dict[round(l_array_all[j], 1)]
            ul_B_mat[j, :, :] = uBl_zM_dict[round(l_array_all[j], 1)]
            ul_C_mat[j, :, :] = uCl_zM_dict[round(l_array_all[j], 1)]
            ul_D_mat[j, :, :] = uDl_zM_dict[round(l_array_all[j], 1)]

        uAl1_uBl1 = ul_A_mat * ul_B_mat
        uCl2_uDl2 = ul_C_mat * ul_D_mat
        uAl1_uBl1_mat = np.tile(uAl1_uBl1.reshape(1, nl, self.PS_prepDV.nz, self.PS_prepDV.nm), (nl, 1, 1, 1))
        uCl2_uDl2_mat = np.tile(uCl2_uDl2.reshape(nl, 1, self.PS_prepDV.nz, self.PS_prepDV.nm), (1, nl, 1, 1))
        dndm_array_mat = np.tile(self.PS_prepDV.dndm_array.reshape(1, 1, self.PS_prepDV.nz, self.PS_prepDV.nm),
                                 (nl, nl, 1, 1))
        if 'g' in [A, B, C, D]:
            toint_M = (uAl1_uBl1_mat * uCl2_uDl2_mat) * dndm_array_mat * self.M_mat_cond_inbin * self.z_mat_cond_inbin
        else:
            toint_M = (uAl1_uBl1_mat * uCl2_uDl2_mat) * dndm_array_mat
        val_z = sp.integrate.simps(toint_M, self.PS_prepDV.M_array)
        chi2_array_mat = np.tile((self.PS_prepDV.chi_array ** 2).reshape(1, 1, self.PS_prepDV.nz), (nl, nl, 1))
        dchi_dz_array_mat = np.tile(self.PS_prepDV.dchi_dz_array.reshape(1, 1, self.PS_prepDV.nz), (nl, nl, 1))
        toint_z = val_z * chi2_array_mat * dchi_dz_array_mat
        val = sp.integrate.simps(toint_z, self.PS_prepDV.z_array)

        if self.PS_prepDV.use_only_halos:
            if (A + B + C + D not in [''.join(elem) for elem in list(set(list(itertools.permutations('gyyy'))))]) and (
                    A + B + C + D != 'yyyy'):
                val = np.zeros(val.shape)

        return val

    def get_dlnCl1h_AB_dlnM(self, l, uA_zM_dict, uB_zM_dict):
        uAl_zM = uA_zM_dict[round(l, 1)]
        uBl_zM = uB_zM_dict[round(l, 1)]
        toint_M = (uAl_zM * uBl_zM) * self.PS_prepDV.dndm_array
        val_z = sp.integrate.simps(toint_M, self.PS_prepDV.M_array)
        toint_z = val_z * (self.chi_array ** 2) * self.PS_prepDV.dchi_dz_array
        denom = sp.integrate.simps(toint_z, self.PS_prepDV.z_array)

        num_volfac = np.tile(((self.PS_prepDV.chi_array ** 2) * self.PS_prepDV.dchi_dz_array).reshape(1, self.nz),
                             (self.nm, 1))
        toint_z_num = self.PS_prepDV.dndm_array.T * num_volfac * (uAl_zM.T * uBl_zM.T)
        num = self.PS_prepDV.M_array * sp.integrate.simps(toint_z_num, self.PS_prepDV.z_array)
        return num / denom

    def get_dlnCl1h_AB_dlnz(self, l, uA_zM_dict, uB_zM_dict):
        uAl_zM = uA_zM_dict[round(l, 1)]
        uBl_zM = uB_zM_dict[round(l, 1)]
        toint_M = (uAl_zM * uBl_zM) * self.PS_prepDV.dndm_array
        val_z = sp.integrate.simps(toint_M, self.PS_prepDV.M_array)
        toint_z = val_z * (self.PS_prepDV.chi_array ** 2) * self.PS_prepDV.dchi_dz_array
        denom = sp.integrate.simps(toint_z, self.PS_prepDV.z_array)

        num_volfac = self.PS_prepDV.z_array * (self.PS_prepDV.chi_array ** 2) * self.PS_prepDV.dchi_dz_array
        num = num_volfac * sp.integrate.simps(self.PS_prepDV.dndm_array * (uAl_zM * uBl_zM), self.PS_prepDV.M_array)
        return num / denom

    def get_Cl_yg_miscentered(self, l_array, Cl_yg):
        if self.PS_prepDV.verbose:
            print('doing miscentering....')
        nl = len(l_array)
        # pdb.set_trace()
        # l_array_full = np.logspace(np.log10(1.0), np.log10(1000), 16000)
        l_array_full = np.linspace(np.min(l_array), np.max(l_array), 12000)
        # l_array_full = l_array
        nl_full = len(l_array_full)

        Cl_yg_interp = interpolate.interp1d(np.log(l_array), Cl_yg)
        Cl_yg_full = Cl_yg_interp(np.log(l_array_full))

        # theta_min = np.pi/(np.max(l_array))
        # theta_max = np.pi/(np.min(l_array))

        theta_min = 1e-5
        theta_max = 0.1

        theta_array_rad = np.logspace(np.log10(theta_min), np.log10(theta_max), 40)
        ntheta = len(theta_array_rad)

        Cl_yg_mat = (np.tile(Cl_yg_full.reshape(1, nl_full), (ntheta, 1)))
        l_theta = (np.tile(l_array_full.reshape(1, nl_full), (ntheta, 1))) * (
            np.tile(theta_array_rad.reshape(ntheta, 1), (1, nl_full)))
        j0_ltheta = sp.special.jv(0, l_theta)
        l_mat = (np.tile(l_array_full.reshape(1, nl_full), (ntheta, 1)))
        Cl_yg_theta = (sp.integrate.simps(l_mat * Cl_yg_mat * j0_ltheta, l_array_full)) / (2 * np.pi)

        R_array = theta_array_rad * self.PS_prepDV.cosmo_colossus.angularDiameterDistance(
            np.mean(self.PS_prepDV.z_array))

        Rmis_array = np.logspace(-4, 1, 28)
        psi_array = np.linspace(0, 2 * np.pi, 28)
        cospsi_array = np.cos(psi_array)
        nRmis = len(Rmis_array)
        npsi = len(psi_array)

        Rmis_nRmis_npsi = (np.tile(Rmis_array.reshape(1, nRmis, 1), (ntheta, 1, npsi)))
        cospsi_nRmis_npsi = (np.tile(cospsi_array.reshape(1, 1, npsi), (ntheta, nRmis, 1)))

        theta_min_rad_full = theta_min
        theta_max_rad_full = theta_max
        theta_array_rad_full = np.logspace(np.log10(theta_min_rad_full), np.log10(theta_max_rad_full), 3800)
        ntheta_full = len(theta_array_rad_full)

        Rmat_nRmis_npsi = (np.tile(R_array.reshape(ntheta, 1, 1), (1, nRmis, npsi)))

        R_arg_new = np.sqrt(
            Rmat_nRmis_npsi ** 2 + Rmis_nRmis_npsi ** 2 + 2 * Rmat_nRmis_npsi * Rmis_nRmis_npsi * cospsi_nRmis_npsi)

        # if np.any(Cl_yg_theta < 0):
        #     if self.verbose:
        #         print('negative values in Cl_yg_theta. Careful about extrapolation!!!!')
        # Cly_theta_interp = interpolate.interp1d(R_array, Cl_yg_theta, fill_value='extrapolate')
        Cly_theta_interp = interpolate.interp1d(R_array, Cl_yg_theta, fill_value=0.0, bounds_error=False)
        Cly_theta_argnew = Cly_theta_interp(R_arg_new)

        Cly_intpsi = (1. / (2 * np.pi)) * sp.integrate.simps(Cly_theta_argnew, psi_array)

        sigmaR_val = self.cmis * np.mean(self.PS_prepDV.r_vir_mat)

        sigmaR_mat = (np.tile(sigmaR_val.reshape(1, 1), (ntheta, nRmis)))
        Rmis_mat = (np.tile(Rmis_array.reshape(1, nRmis), (ntheta, 1)))
        PRmis_mat = (Rmis_mat / sigmaR_mat ** 2) * np.exp(-1. * ((Rmis_mat ** 2) / (2. * sigmaR_mat ** 2)))

        Cly_intRmis = sp.integrate.simps(Cly_intpsi * PRmis_mat, Rmis_array)

        if np.all(Cly_intRmis > 0):
            Cly_intRmis_interp = interpolate.interp1d(np.log(theta_array_rad), np.log(Cly_intRmis))
            Cly_misc_theta = np.exp(Cly_intRmis_interp(np.log(theta_array_rad_full)))

        else:
            # print 'negative values in Cly_intRmis. Careful about extrapolation!!!!'
            Cly_intRmis_interp = interpolate.interp1d(np.log(theta_array_rad), Cly_intRmis)
            Cly_misc_theta = (Cly_intRmis_interp(np.log(theta_array_rad_full)))

        # if np.all(Cl_yg_theta > 0):
        #     Cly_theta_zM_interp = interpolate.interp1d(np.log(theta_array_rad), np.log(Cl_yg_theta),
        #                                                fill_value='extrapolate')
        #     Cly_origcheck_theta = np.exp(Cly_theta_zM_interp(np.log(theta_array_rad_full)))
        #
        # else:
        #     print 'negative values in Cl_yg_theta'
        #     Cly_theta_zM_interp = interpolate.interp1d(np.log(theta_array_rad), Cl_yg_theta,
        #                                                fill_value='extrapolate')
        #     Cly_origcheck_theta = (Cly_theta_zM_interp(np.log(theta_array_rad_full)))

        Cly_misc_theta_full = np.tile(Cly_misc_theta.reshape(1, ntheta_full), (nl, 1))
        l_thetafull = (np.tile(l_array.reshape(nl, 1), (1, ntheta_full))) * (
            np.tile(theta_array_rad_full.reshape(1, ntheta_full), (nl, 1)))
        j0_lthetafull = sp.special.jv(0, l_thetafull)
        theta_mat = (np.tile(theta_array_rad_full.reshape(1, ntheta_full), (nl, 1)))
        Cly_misc_l = (2 * np.pi) * (
            sp.integrate.simps(theta_mat * Cly_misc_theta_full * j0_lthetafull, theta_array_rad_full))

        # Cly_origcheck_theta_full = np.tile(Cly_origcheck_theta.reshape(1, ntheta_full), (nl, 1))
        # Cly_origcheck_l = (2 * np.pi) * (
        #     sp.integrate.simps(theta_mat * Cly_origcheck_theta_full * j0_lthetafull, theta_array_rad_full))

        Cly_misc_l_final = self.PS_prepDV.fmis * Cly_misc_l + (1 - self.PS_prepDV.fmis) * Cl_yg

        # pdb.set_trace()
        return Cly_misc_l_final

    def get_cov_G(self, bin1_stat1, bin2_stat1, bin1_stat2, bin2_stat2, stats_analyze_1, stats_analyze_2,
                  Cl_result_dict, fsky_dict):

        cov_dict_G = {}

        # for j in range(len(self.stats_analyze_pairs)):
        #     stats_analyze_1, stats_analyze_2 = self.stats_analyze_pairs[j]
        A, B = list(stats_analyze_1)
        C, D = list(stats_analyze_2)
        stats_pairs = [A + C, B + D, A + D, B + C]
        bin_pairs = [[bin1_stat1, bin1_stat2], [bin2_stat1, bin2_stat2], [bin1_stat1, bin2_stat2],
                     [bin2_stat1, bin1_stat2]]
        Cl_stats_dict = {}

        for j in range(len(stats_pairs)):
            stat = stats_pairs[j]
            bin_pair = bin_pairs[j]
            bin_key = 'bin_' + str(bin_pair[0]) + '_' + str(bin_pair[1])
            Atemp, Btemp = list(stat)
            if Atemp == Btemp:
                try:
                    Cl_stats_dict[stat] = Cl_result_dict[stat][bin_key]['tot_plus_noise_ellsurvey']
                except:
                    bin_key = 'bin_' + str(bin_pair[1]) + '_' + str(bin_pair[0])
                    Cl_stats_dict[stat] = Cl_result_dict[stat][bin_key]['tot_plus_noise_ellsurvey']
            else:
                try:
                    Cl_stats_dict[stat] = Cl_result_dict[stat][bin_key]['tot_plus_noise_ellsurvey']
                except:
                    bin_key = 'bin_' + str(bin_pair[1]) + '_' + str(bin_pair[0])
                    Cl_stats_dict[stat] = Cl_result_dict[Btemp + Atemp][bin_key]['tot_plus_noise_ellsurvey']

        # import pdb;pdb.set_trace()
        fsky_j = np.sqrt(fsky_dict[A + B] * fsky_dict[C + D])

        val_diag = (1. / (fsky_j * (2 * Cl_result_dict['l_array_survey'] + 1.) * Cl_result_dict['dl_array_survey'])) * (
                Cl_stats_dict[A + C] * Cl_stats_dict[B + D] + Cl_stats_dict[A + D] * Cl_stats_dict[B + C])

        # cov_dict_G[A + B + '_' + C + D] = np.diag(val_diag)
        # cov_dict_G[C + D + '_' + A + B] = np.diag(val_diag)

        return np.diag(val_diag)

    def get_cov_NG(self, l_array_survey, stats_analyze_1, stats_analyze_2, use_only_halos, fsky_dict, uAl_zM_dict,
                   uBl_zM_dict, uCl_zM_dict, uDl_zM_dict):
        A, B = list(stats_analyze_1)
        C, D = list(stats_analyze_2)

        if use_only_halos and (A + B + '_' + C + D != 'yy_yy'):
            nl = len(l_array_survey)
            val_NG = np.zeros((nl, nl))
        else:
            T_l_ABCD = self.get_T_ABCD_NG(l_array_survey, A, B, C, D, uAl_zM_dict, uBl_zM_dict, uCl_zM_dict,
                                          uDl_zM_dict)
            fsky_j = np.sqrt(fsky_dict[A + B] * fsky_dict[C + D])
            # pdb.set_trace()
            # fsky_j = np.min(np.array([self.fsky[A + B] , self.fsky[C + D]]))
            val_NG = (1. / (4. * np.pi * fsky_j)) * T_l_ABCD

        return val_NG

    def do_Hankel_transform(self, nu, ell_array, Cell_array, theta_array_arcmin=None):
        l_array_full = np.logspace(np.log10(0.1), np.log10(20000), 200000)
        Cell_interp = interpolate.interp1d(np.log(ell_array), np.log(Cell_array), fill_value='extrapolate',
                                           bounds_error=False)
        Cell_full = np.exp(Cell_interp(np.log(l_array_full)))
        theta_out, xi_out = Hankel(l_array_full, nu=nu, q=1.0)(Cell_full, extrap=True)
        xi_out *= (1 / (2 * np.pi))
        theta_out_arcmin = theta_out * (180. / np.pi) * 60.
        if theta_array_arcmin is not None:
            xi_interp = interpolate.interp1d(np.log(theta_out_arcmin), np.log(xi_out), fill_value='extrapolate',
                                             bounds_error=False)
            xi_final = np.exp(xi_interp(np.log(theta_array_arcmin)))
        else:
            xi_final = xi_out
            theta_array_arcmin = theta_out_arcmin
        return xi_final, theta_array_arcmin


class DataVec:
    def __init__(self, PrepDV_params, block):
        self.CalcDV = CalcDataVec(PrepDV_params)
        PrepDV = PrepDV_params['PrepDV_fid']
        self.verbose = PrepDV_params['verbose']
        run_cov_pipe = PrepDV_params['run_cov_pipe']
        bins_source = PrepDV_params['bins_source']
        bins_lens = PrepDV_params['bins_lens']
        analysis_coords = PrepDV_params['analysis_coords']
        theta_array_arcmin = PrepDV_params['theta_array']
        gg_doauto = PrepDV_params['gg_doauto']
        fsky_dict = PrepDV_params['fsky_dict']
        sec_save_name = PrepDV_params['sec_save_name']
        self.Cl_result_dict = {'l_array': PrepDV.l_array, 'l_array_survey': PrepDV.l_array_survey,
                          'ind_select_survey': PrepDV.ind_select_survey,
                          'dl_array_survey': PrepDV.dl_array_survey}
        if analysis_coords == 'real':
            self.xi_result_dict = {}

        if self.verbose:
            print('starting Cls calculation')
        if ('kk' in PrepDV.stats_analyze) or (run_cov_pipe and ('kk' in PrepDV.lss_probes_allcomb)):
            Cl_kk_dict = {}
            if analysis_coords == 'real':
                xi_kk_dict = {}
            for j1 in bins_source:
                for j2 in bins_source:
                    bin_combs = []
                    if j2 >= j1:
                        Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('k', 'k', PrepDV.l_array,
                                                             PrepDV_params['ukl_zM_dict' + str(j1)],
                                                             PrepDV_params['ukl_zM_dict' + str(j2)])
                        Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('k', 'k', PrepDV.l_array,
                                                             PrepDV_params['bkl_z_dict' + str(j1)],
                                                             PrepDV_params['bkl_z_dict' + str(j2)])
                        Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('k', 'k', Cl1h_j1j2, Cl2h_j1j2)
                        if j1 == j2:
                            Cl_noise_ellsurvey = PrepDV_params['Cl_noise_kk_l_array' + str(j1)]
                        else:
                            Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                        bin_combs.append([j1, j2])
                        Cl_kk_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2,
                                                                        'tot': Cltot_j1j2,
                                                                        'tot_ellsurvey': Cltot_j1j2[
                                                                            PrepDV.ind_select_survey],
                                                                        'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                                                        PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                        if analysis_coords == 'real':
                            xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                      Cltot_j1j2,
                                                                                      theta_array_arcmin=theta_array_arcmin)
                            xi_kk_dict['bin_' + str(j1) + '_' + str(j2)] = xitot_j1j2
                            if 'theta' not in xi_kk_dict.keys():
                                xi_kk_dict['theta'] = theta_array
                            if 'kk' in PrepDV.stats_analyze:
                                block[sec_save_name, 'theory_corrf_' + 'kk' + '_' + 'bin_' + str(j1) + '_' + str(j2)] = xitot_j1j2
                                block[sec_save_name, 'xcoord_' + 'kk' + '_' + 'bin_' + str(j1) + '_' + str(j2)] = theta_array
                        else:
                            if 'kk' in PrepDV.stats_analyze:
                                block[sec_save_name, 'theory_corrf_' + 'kk' + '_' + 'bin_' + str(j1) + '_' + str(j2)] = Cltot_j1j2
                                block[sec_save_name, 'xcoord_' + 'kk' + '_' + 'bin_' + str(j1) + '_' + str(j2)] = PrepDV.l_array


            Cl_kk_dict['bin_combs'] = bin_combs
            self.Cl_result_dict['kk'] = Cl_kk_dict
            if analysis_coords == 'real':
                xi_kk_dict['bin_combs'] = bin_combs
                self.xi_result_dict['kk'] = xi_kk_dict

            if self.verbose:
                print('done shear-shear calculation')

        if ('ky' in PrepDV.stats_analyze) or (run_cov_pipe and ('ky' in PrepDV.lss_probes_allcomb)):
            Cl_ky_dict = {}
            if analysis_coords == 'real':
                # xi_ky_dict = {}
                xi_gty_dict = {}
            bin_combs = []
            for j1 in bins_source:
                Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('k', 'y', PrepDV.l_array,
                                                     PrepDV_params['ukl_zM_dict' + str(j1)],
                                                     PrepDV_params['uyl_zM_dict0'])
                Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('k', 'y', PrepDV.l_array, PrepDV_params['bkl_z_dict' + str(j1)],
                                                     PrepDV_params['byl_z_dict0'])
                Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('k', 'y', Cl1h_j1j2, Cl2h_j1j2)
                Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                bin_combs.append([j1, 0])
                Cl_ky_dict['bin_' + str(j1) + '_0'] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2, 'tot': Cltot_j1j2,
                                                       'tot_ellsurvey': Cltot_j1j2[PrepDV.ind_select_survey],
                                                       'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                                       PrepDV.ind_select_survey] + Cl_noise_ellsurvey}

                if analysis_coords == 'real':
                    # xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                    #                                                           Cltot_j1j2,
                    #                                                           theta_array_arcmin=theta_array_arcmin)
                    gt_tot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                               Cltot_j1j2,
                                                                               theta_array_arcmin=theta_array_arcmin)
                    # xi_ky_dict['bin_' + str(j1)] = xitot_j1j2
                    xi_gty_dict['bin_' + str(j1) + '_0'] = gt_tot_j1j2
                    if 'theta' not in xi_kk_dict.keys():
                        # xi_ky_dict['theta'] = theta_array
                        xi_gty_dict['theta'] = theta_array
                    if 'ky' in PrepDV.stats_analyze:
                        block[
                            sec_save_name, 'theory_corrf_' + 'gty' + '_' + 'bin_' + str(j1) + '_' + str(0)] = gt_tot_j1j2
                        block[sec_save_name, 'xcoord_' + 'gty' + '_' + 'bin_' + str(j1) + '_' + str(0)] = theta_array
                else:
                    if 'ky' in PrepDV.stats_analyze:
                        block[
                            sec_save_name, 'theory_corrf_' + 'gty' + '_' + 'bin_' + str(j1) + '_' + str(0)] = Cltot_j1j2
                        block[sec_save_name, 'xcoord_' + 'gty' + '_' + 'bin_' + str(j1) + '_' + str(0)] = PrepDV.l_array

            Cl_ky_dict['bin_combs'] = bin_combs
            self.Cl_result_dict['ky'] = Cl_ky_dict
            # Cl_result_dict['yk'] = Cl_ky_dict
            if analysis_coords == 'real':
                # xi_result_dict['ky'] = xi_ky_dict
                xi_gty_dict['bin_combs'] = bin_combs
                self.xi_result_dict['gty'] = xi_gty_dict

            if self.verbose:
                print('done shear-y calculation')

        if ('yy' in PrepDV.stats_analyze) or (run_cov_pipe and ('yy' in PrepDV.lss_probes_allcomb)):
            Cl_yy_dict = {}
            Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('y', 'y', PrepDV.l_array, PrepDV_params['uyl_zM_dict0'],
                                                 PrepDV_params['uyl_zM_dict0'])
            Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('y', 'y', PrepDV.l_array, PrepDV_params['byl_z_dict0'],
                                                 PrepDV_params['byl_z_dict0'])
            Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('y', 'y', Cl1h_j1j2, Cl2h_j1j2)
            Cl_noise_ellsurvey = PrepDV_params['Cl_noise_yy_l_array']

            Cl_yy_dict['bin_0_0'] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2, 'tot': Cltot_j1j2,
                                     'tot_ellsurvey': Cltot_j1j2[PrepDV.ind_select_survey],
                                     'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                     PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
            Cl_yy_dict['bin_combs'] = [[0, 0]]
            self.Cl_result_dict['yy'] = Cl_yy_dict
            if analysis_coords == 'real':
                xi_yy_dict = {}
                xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                          Cltot_j1j2,
                                                                          theta_array_arcmin=theta_array_arcmin)
                xi_yy_dict['bin_0_0'] = xitot_j1j2
                xi_yy_dict['theta'] = theta_array
                xi_yy_dict['bin_combs'] = [[0, 0]]
                self.xi_result_dict['yy'] = xi_yy_dict

                if 'yy' in PrepDV.stats_analyze:
                    block[
                        sec_save_name, 'theory_corrf_' + 'yy' + '_' + 'bin_' + str(0) + '_' + str(0)] = xitot_j1j2
                    block[sec_save_name, 'xcoord_' + 'yy' + '_' + 'bin_' + str(0) + '_' + str(0)] = theta_array
            else:
                if 'yy' in PrepDV.stats_analyze:
                    block[
                        sec_save_name, 'theory_corrf_' + 'yy' + '_' + 'bin_' + str(0) + '_' + str(0)] = Cltot_j1j2
                    block[sec_save_name, 'xcoord_' + 'yy' + '_' + 'bin_' + str(0) + '_' + str(0)] = PrepDV.l_array


            if self.verbose:
                print('done y-y calculation')

        if ('gg' in PrepDV.stats_analyze) or (run_cov_pipe and ('gg' in PrepDV.lss_probes_allcomb)):
            Cl_gg_dict = {}
            if analysis_coords == 'real':
                xi_gg_dict = {}
            bin_combs = []
            for j1 in bins_lens:
                for j2 in bins_lens:
                    if j1 >= j2:
                        if gg_doauto:
                            if j1 == j2:
                                runj1j2 = True
                            else:
                                runj1j2 = False
                        else:
                            runj1j2 = True
                        if runj1j2:
                            Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('g', 'g', PrepDV.l_array,
                                                                 PrepDV_params['ugl_zM_dict' + str(j1)],
                                                                 PrepDV_params['ugl_zM_dict' + str(j2)])
                            Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('g', 'g', PrepDV.l_array,
                                                                 PrepDV_params['bgl_z_dict' + str(j1)],
                                                                 PrepDV_params['bgl_z_dict' + str(j2)])
                            Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('g', 'g', Cl1h_j1j2, Cl2h_j1j2)
                            if j1 == j2:
                                Cl_noise_ellsurvey = PrepDV_params['Cl_noise_gg_l_array' + str(j1)]
                            else:
                                Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                            bin_combs.append([j1, j2])
                            Cl_gg_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2,
                                                                            'tot': Cltot_j1j2,
                                                                            'tot_ellsurvey': Cltot_j1j2[
                                                                                PrepDV.ind_select_survey],
                                                                            'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                                                            PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                            if analysis_coords == 'real':
                                xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                          Cltot_j1j2,
                                                                                          theta_array_arcmin=theta_array_arcmin)
                                xi_gg_dict['bin_' + str(j1) + '_' + str(j2)] = xitot_j1j2
                                if 'theta' not in xi_gg_dict.keys():
                                    xi_gg_dict['theta'] = theta_array

                                if 'gg' in PrepDV.stats_analyze:
                                    block[sec_save_name, 'theory_corrf_' + 'gg' + '_' + 'bin_' + str(j1) + '_' + str(
                                        j2)] = xitot_j1j2

                                    block[sec_save_name, 'xcoord_' + 'gg' + '_' + 'bin_' + str(j1) + '_' + str(
                                        j2)] = theta_array
                            else:
                                if 'gg' in PrepDV.stats_analyze:
                                    block[sec_save_name, 'theory_corrf_' + 'gg' + '_' + 'bin_' + str(j1) + '_' + str(
                                        j2)] = Cltot_j1j2

                                    block[sec_save_name, 'xcoord_' + 'gg' + '_' + 'bin_' + str(j1) + '_' + str(
                                        j2)] = PrepDV.l_array

            Cl_gg_dict['bin_combs'] = bin_combs
            self.Cl_result_dict['gg'] = Cl_gg_dict
            if analysis_coords == 'real':
                xi_gg_dict['bin_combs'] = bin_combs
                self.xi_result_dict['gg'] = xi_gg_dict

        if ('gy' in PrepDV.stats_analyze) or (run_cov_pipe and ('gy' in PrepDV.lss_probes_allcomb)):
            Cl_gy_dict = {}
            if analysis_coords == 'real':
                xi_gy_dict = {}
            bin_combs = []
            for j1 in bins_lens:
                Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('g', 'y', PrepDV.l_array,
                                                     PrepDV_params['ugl_zM_dict' + str(j1)],
                                                     PrepDV_params['uyl_zM_dict0'])
                Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('g', 'y', PrepDV.l_array, PrepDV_params['bgl_z_dict' + str(j1)],
                                                     PrepDV_params['byl_z_dict0'])
                Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('g', 'y', Cl1h_j1j2, Cl2h_j1j2)
                Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                bin_combs.append([j1, 0])
                Cl_gy_dict['bin_' + str(j1) + '_0'] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2, 'tot': Cltot_j1j2,
                                                       'tot_ellsurvey': Cltot_j1j2[PrepDV.ind_select_survey],
                                                       'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                                       PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                if analysis_coords == 'real':
                    xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                              Cltot_j1j2,
                                                                              theta_array_arcmin=theta_array_arcmin)
                    xi_gy_dict['bin_' + str(j1) + '_0'] = xitot_j1j2

                    if 'theta' not in xi_kk_dict.keys():
                        xi_gy_dict['theta'] = theta_array

                    if 'gy' in PrepDV.stats_analyze:
                        block[sec_save_name, 'theory_corrf_' + 'gy' + '_' + 'bin_' + str(j1) + '_' + str(
                            0)] = xitot_j1j2

                        block[sec_save_name, 'xcoord_' + 'gy' + '_' + 'bin_' + str(j1) + '_' + str(
                            0)] = theta_array
                else:
                    if 'gy' in PrepDV.stats_analyze:
                        block[sec_save_name, 'theory_corrf_' + 'gy' + '_' + 'bin_' + str(j1) + '_' + str(
                            0)] = Cltot_j1j2

                        block[sec_save_name, 'xcoord_' + 'gy' + '_' + 'bin_' + str(j1) + '_' + str(
                            0)] = PrepDV.l_array

            Cl_gy_dict['bin_combs'] = bin_combs
            self.Cl_result_dict['gy'] = Cl_gy_dict
            # Cl_result_dict['yg'] = Cl_gy_dict
            if analysis_coords == 'real':
                xi_gy_dict['bin_combs'] = bin_combs
                self.xi_result_dict['gy'] = xi_gy_dict

        if ('gk' in PrepDV.stats_analyze) or (run_cov_pipe and ('gk' in PrepDV.lss_probes_allcomb)):
            Cl_gk_dict = {}
            if analysis_coords == 'real':
                xi_gk_dict = {}
                xi_gtg_dict = {}
            bin_combs = []
            for j1 in bins_lens:
                for j2 in bins_source:
                    Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('g', 'k', PrepDV.l_array,
                                                         PrepDV_params['ugl_zM_dict' + str(j1)],
                                                         PrepDV_params['ukl_zM_dict' + str(j2)])
                    Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('g', 'k', PrepDV.l_array,
                                                         PrepDV_params['bgl_z_dict' + str(j1)],
                                                         PrepDV_params['ukl_zM_dict' + str(j2)])
                    Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('g', 'k', Cl1h_j1j2, Cl2h_j1j2)
                    Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                    bin_combs.append([j1, j2])
                    Cl_gk_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2, 'tot': Cltot_j1j2,
                                                                    'tot_ellsurvey': Cltot_j1j2[
                                                                        PrepDV.ind_select_survey],
                                                                    'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                                                    PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                    if analysis_coords == 'real':
                        xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                  Cltot_j1j2,
                                                                                  theta_array_arcmin=theta_array_arcmin)
                        gt_tot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                   Cltot_j1j2,
                                                                                   theta_array_arcmin=theta_array_arcmin)
                        xi_gk_dict['bin_' + str(j1) + '_' + str(j2)] = xitot_j1j2
                        xi_gtg_dict['bin_' + str(j1) + '_' + str(j2)] = gt_tot_j1j2
                        if 'theta' not in xi_kk_dict.keys():
                            xi_gk_dict['theta'] = theta_array
                            xi_gtg_dict['theta'] = theta_array

                        if 'gk' in PrepDV.stats_analyze:
                            block[sec_save_name, 'theory_corrf_' + 'gtg' + '_' + 'bin_' + str(j1) + '_' + str(
                                j2)] = gt_tot_j1j2
                            block[sec_save_name, 'xcoord_' + 'gtg' + '_' + 'bin_' + str(j1) + '_' + str(
                                j2)] = theta_array
                            block[sec_save_name, 'theory_corrf_' + 'gk' + '_' + 'bin_' + str(j1) + '_' + str(
                                j2)] = xitot_j1j2
                            block[sec_save_name, 'xcoord_' + 'gk' + '_' + 'bin_' + str(j1) + '_' + str(
                                j2)] = theta_array
                    else:
                        if 'gk' in PrepDV.stats_analyze:
                            block[sec_save_name, 'theory_corrf_' + 'gk' + '_' + 'bin_' + str(j1) + '_' + str(
                                j2)] = Cltot_j1j2
                            block[sec_save_name, 'xcoord_' + 'gk' + '_' + 'bin_' + str(j1) + '_' + str(
                                j2)] = PrepDV.l_array

            Cl_gk_dict['bin_combs'] = bin_combs
            self.Cl_result_dict['gk'] = Cl_gk_dict
            # Cl_result_dict['kg'] = Cl_gk_dict
            if analysis_coords == 'real':
                xi_gk_dict['bin_combs'] = bin_combs
                xi_gtg_dict['bin_combs'] = bin_combs
                self.xi_result_dict['gk'] = xi_gk_dict
                self.xi_result_dict['gtg'] = xi_gtg_dict

        if run_cov_pipe:
            if self.verbose:
                print('starting covariance calculation')

            if analysis_coords == 'real':
                if self.verbose:
                    print('setting up realspace covariance')

                isodd = 0
                ell_temp = PrepDV.l_array_survey

                if np.mod(len(ell_temp), 2) > 0:
                    isodd = 1
                    ell = ell_temp[:-1]
                else:
                    ell = ell_temp
                nl = len(ell)
                dlnk = np.log(ell[1] / ell[0])
                ell_mat = np.tile(ell.reshape(nl, 1), (1, nl))
                ell1_ell2 = ell_mat * ell_mat.T
                self.fftcovtot_dict = {}

            self.covG_dict = {}
            self.covNG_dict = {}
            self.covtot_dict = {}
            if analysis_coords == 'real':
                # self.fftcovG_dict = {}
                # self.fftcovNG_dict = {}
                self.fftcovtot_dict = {}

            for j in range(len(PrepDV.stats_analyze_pairs)):
                stats_analyze_1, stats_analyze_2 = PrepDV.stats_analyze_pairs[j]
                if self.verbose:
                    print('starting covariance of ' + str(stats_analyze_1) + ' and ' + str(stats_analyze_2))
                if stats_analyze_1 in self.Cl_result_dict.keys():
                    stats_analyze_1_ordered = stats_analyze_1
                else:
                    stats_analyze_1_ordered = list(stats_analyze_1)[1] + list(stats_analyze_1)[0]
                bin_combs_stat1 = self.Cl_result_dict[stats_analyze_1_ordered]['bin_combs']
                bins1_stat1 = []
                bins2_stat1 = []
                for jb in range(len(bin_combs_stat1)):
                    bins1_stat1.append(bin_combs_stat1[jb][0])
                    bins2_stat1.append(bin_combs_stat1[jb][1])

                if stats_analyze_2 in self.Cl_result_dict.keys():
                    stats_analyze_2_ordered = stats_analyze_2
                else:
                    stats_analyze_2_ordered = list(stats_analyze_2)[1] + list(stats_analyze_2)[0]
                bin_combs_stat2 = self.Cl_result_dict[stats_analyze_2_ordered]['bin_combs']
                bins1_stat2 = []
                bins2_stat2 = []
                for jb in range(len(bin_combs_stat2)):
                    bins1_stat2.append(bin_combs_stat2[jb][0])
                    bins2_stat2.append(bin_combs_stat2[jb][1])

                covG_stat12 = {}
                covNG_stat12 = {}
                covtot_stat12 = {}
                isgtykk, isgtygty = False, False
                if analysis_coords == 'real':
                    fftcovtot_stat12 = {}
                    if (stats_analyze_1_ordered == 'ky') and (stats_analyze_2_ordered == 'ky'):
                        gtfftcovtot_stat12 = {}
                        isgtygty = True
                    if ((stats_analyze_1_ordered == 'kk') and (stats_analyze_2_ordered == 'ky')) or ((stats_analyze_1_ordered == 'ky') and (stats_analyze_2_ordered == 'kk')):
                        kkgtfftcovtot_stat12 = {}
                        isgtykk = True
                bins_comb = []
                for jb1 in range(len(bins1_stat1)):
                    for jb2 in range(len(bins1_stat2)):
                        covG = self.CalcDV.get_cov_G(bins1_stat1[jb1], bins2_stat1[jb1], bins1_stat2[jb2],
                                                     bins2_stat2[jb2], stats_analyze_1_ordered, stats_analyze_2_ordered,
                                                     self.Cl_result_dict, fsky_dict)

                        A, B = list(stats_analyze_1_ordered)
                        C, D = list(stats_analyze_2_ordered)

                        uAl_zM_dict = PrepDV_params['u' + A + 'l_zM_dict' + str(bins1_stat1[jb1])]
                        uBl_zM_dict = PrepDV_params['u' + B + 'l_zM_dict' + str(bins2_stat1[jb1])]
                        uCl_zM_dict = PrepDV_params['u' + C + 'l_zM_dict' + str(bins1_stat2[jb2])]
                        uDl_zM_dict = PrepDV_params['u' + D + 'l_zM_dict' + str(bins2_stat2[jb2])]

                        covNG = self.CalcDV.get_cov_NG(PrepDV.l_array_survey, stats_analyze_1_ordered,
                                                       stats_analyze_2_ordered, PrepDV.PS.use_only_halos, fsky_dict,
                                                       uAl_zM_dict, uBl_zM_dict, uCl_zM_dict, uDl_zM_dict)

                        covtot = covG + covNG
                        bin_key = 'bin_' + str(bins1_stat1[jb1]) + '_' + str(bins2_stat1[jb1]) + '_' + str(bins1_stat2[jb2]) + '_' + str(bins2_stat2[jb2])
                        covG_stat12[bin_key] = covG
                        covNG_stat12[bin_key] = covNG
                        covtot_stat12[bin_key] = covtot
                        bins_comb.append([bins1_stat1[jb1],bins2_stat1[jb1],bins1_stat2[jb2],bins2_stat2[jb2]])
                        if analysis_coords == 'real':
                            if isodd:
                                covtot_rs = covtot[:-1, :][:, :-1]
                            else:
                                covtot_rs = covtot
                            newtwobessel = two_Bessel(ell, ell, covtot_rs * (ell1_ell2 ** 2) * (1. / (4 * np.pi ** 2)),
                                                      nu1=1.01, nu2=1.01, N_extrap_low=0, N_extrap_high=0,
                                                      c_window_width=0.25,
                                                      N_pad=1000)
                            t1, t2, cov_fft = newtwobessel.two_Bessel_binave(0, 0, dlnk, dlnk)
                            theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                            cov_tot_fft = cov_fft[:,:-1][:-1,:]
                            fftcovtot_stat12[bin_key] = cov_tot_fft

                            if isgtygty:
                                t1, t2, covgt_fft = newtwobessel.two_Bessel_binave(2, 2, dlnk, dlnk)
                                gtfftcovtot_stat12[bin_key] = covgt_fft[:,:-1][:-1,:]
                                theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                                if 'theta' not in gtfftcovtot_stat12.keys():
                                    gtfftcovtot_stat12['theta'] = theta_vals_arcmin_fft

                            if isgtykk:
                                t1, t2, covgt_fft = newtwobessel.two_Bessel_binave(2, 0, dlnk, dlnk)
                                kkgtfftcovtot_stat12[bin_key] = covgt_fft[:,:-1][:-1,:]
                                theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                                if 'theta' not in kkgtfftcovtot_stat12.keys():
                                    kkgtfftcovtot_stat12['theta'] = theta_vals_arcmin_fft

                            if 'theta' not in fftcovtot_stat12.keys():
                                fftcovtot_stat12['theta'] = theta_vals_arcmin_fft

                covG_stat12['bins_comb'] = bins_comb
                covNG_stat12['bins_comb'] = bins_comb
                covtot_stat12['bins_comb'] = bins_comb
                if analysis_coords == 'real':
                    fftcovtot_stat12['bins_comb'] = bins_comb
                    if isgtygty:
                        gtfftcovtot_stat12['bins_comb'] = bins_comb
                        stat_analyze_key = 'gty_gty'
                        self.fftcovtot_dict[stat_analyze_key] = gtfftcovtot_stat12

                    if isgtykk:
                        kkgtfftcovtot_stat12['bins_comb'] = bins_comb
                        stat_analyze_key = 'gty_kk'
                        self.fftcovtot_dict[stat_analyze_key] = kkgtfftcovtot_stat12

                    # self.fftcovG_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covG_stat12
                    # self.fftcovNG_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covNG_stat12
                    self.fftcovtot_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = fftcovtot_stat12

                self.covG_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covG_stat12
                self.covNG_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covNG_stat12
                self.covtot_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covtot_stat12

















