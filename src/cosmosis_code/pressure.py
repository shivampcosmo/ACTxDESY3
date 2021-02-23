import sys, os
import traceback
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
import mycosmo as cosmodef
from twobessel import *
import LSS_funcs as hmf
import plot_funcs as pf
import multiprocessing
import time
import pdb
import pickle as pk
import dill
from mcfit import Hankel

pi = np.pi


class Pressure:
    """
    Sets up the pressure profile functions.
    """
    def __init__(self, cosmo_params, pressure_params, other_params):
        cosmology.addCosmology('mock_cosmo', cosmo_params)
        self.cosmo_colossus = cosmology.setCosmology('mock_cosmo')
#        self.cosmo_colossus = cosmology.setCosmology('planck18')
        h = cosmo_params['H0'] / 100.
        cosmo_func = cosmodef.mynew_cosmo(h, cosmo_params['Om0'], cosmo_params['Ob0'], cosmo_params['ns'],
                                          cosmo_params['sigma8'])
        self.cosmo = cosmo_func
        self.verbose = other_params['verbose']

        self.z_array = other_params['z_array']

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

        params_names_gnfwp = copy.deepcopy([*self.pressure_params_dict.keys()])
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
        params_names_gnfwp = copy.deepcopy([*self.pressure_params_dict.keys()])

        nz, nm = M_mat_shape_nz_nm.shape
        nx = len(x)
        xmat = np.tile(x.reshape(1, 1, nx), (nz, nm, 1))
        M_mat_Delta_no_h_nz_nm_nx = (np.tile(M_mat_shape_nz_nm.reshape(nz, nm, 1), (1, 1, nx))) / (self.cosmo.h)

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
#        import ipdb; ipdb.set_trace()
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
            rescale_x = R_mat[0][0]/R200c_mat[0][0]
            to_integrate = 4. * np.pi * (x_array ** 2.) * self.get_y3d(M200c_mat, x_array*rescale_x, z_array, R200c_mat,
                                                                       M200c_mat=M200c_mat,
                                                                       mdef_M_mat='200c')
        else:
            print('provide correct Pe model')
            sys.exit(1)

        coeff = (((dang / (self.cosmo.h * 500.)) ** 2) / (Ez ** (2. / 3.))) * (180. * 60. / np.pi) ** 2
        # integrating dx so need extra R500**3.
        Y500sph = coeff * (rmdefP ** 3) * sp.integrate.simps(to_integrate, x_array) * (1. / dang ** 2)
        return Y500sph


