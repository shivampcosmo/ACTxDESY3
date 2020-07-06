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
from HOD import *
from pressure import *
from general_hm import *
pi = np.pi



class Powerspec:

    def __init__(self, cosmo_params, hod_params, pressure_params, other_params):
        cosmology.addCosmology('mock_cosmo', cosmo_params)
        self.cosmo_colossus = cosmology.setCosmology('mock_cosmo')
#        self.cosmo_colossus = cosmology.setCosmology('planck18')
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
        self.hod = HOD(hod_params,other_params)

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
        if self.mdef_analysis == 'fof':
#            self.M_mat_mdefP = M_mat_mdef / (1.4*hydro_B)
            self.M_mat_mdefP = M_mat_mdef / (hydro_B)
            self.rmdefP_mat = hmf.get_R_from_M_mat(self.M_mat_mdefP,
                                                   200 * self.rho_crit_array)
        elif self.mdef_analysis == other_params['pressure_model_mdef']:
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
        if self.mdef_analysis == 'fof':
            M_mat_200cP = M_mat_mdef*hydro_B    
        elif self.mdef_analysis == '200c':
            M_mat_200cP = M_mat_mdef
        else:
            M_mat_200cP, R_mat_200cP_kpc_h = np.zeros(M_mat_mdef.shape), np.zeros(M_mat_mdef.shape)
            for j in range(self.nz):
                M_mat_200cP[j, :], R_mat_200cP_kpc_h[j, :], _ = mass_defs.changeMassDefinition(M_mat_mdef[j, :],
                                                                                               self.halo_conc_mdef[j,
                                                                                               :], self.z_array[j],
                                                                                               self.mdef_analysis,
                                                                                               '200c')


        self.M_mat_200cP = M_mat_200cP / hydro_B
        self.r200cP_mat = hmf.get_R_from_M_mat(self.M_mat_200cP, 200 * self.rho_crit_array)

        # trying to test only single mdef
        M_mat_vir = M_mat_mdef
        self.r_vir_mat = hmf.get_R_from_M_mat(M_mat_mdef, 200 * self.rho_crit_array)
        self.halo_conc_vir = self.halo_conc_mdef

        # if self.mdef_analysis == 'vir':
        #     M_mat_vir = M_mat_mdef
        #     self.r_vir_mat = hmf.get_R_from_M_mat(M_mat_mdef, self.rho_vir_array)
        #     self.halo_conc_vir = self.halo_conc_mdef
        # else:
        #     if self.verbose:
        #         print('changing mdef to vir for nfw')
        #     M_mat_vir, R_mat_vir_kpc_h, halo_conc_vir = np.zeros(M_mat_mdef.shape), np.zeros(
        #         M_mat_mdef.shape), np.zeros(M_mat_mdef.shape)
        #     for j in range(self.nz):
        #         M_mat_vir[j, :], R_mat_vir_kpc_h[j, :], halo_conc_vir[j, :] = mass_defs.changeMassDefinition(
        #             M_mat_mdef[j, :], self.halo_conc_mdef[j, :], self.z_array[j], self.mdef_analysis, 'vir')
        #     self.r_vir_mat = R_mat_vir_kpc_h / 1000.
        #     self.halo_conc_vir = halo_conc_vir

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
        # import ipdb; ipdb.set_trace()

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
        else:
            self.int_prob = np.ones(self.Ntotal_mat.shape)

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
        if np.any(ng_value_source < 0.0):
            ng_interp_source = interpolate.interp1d(ng_zarray_source, ng_value_source,
                                                    fill_value=0.0, bounds_error=False)
            self.ng_array_source = ng_interp_source(self.z_array)
        else:
            ng_interp_source = interpolate.interp1d(ng_zarray_source, np.log(ng_value_source + 1e-40),
                                                    fill_value='extrapolate')
            self.ng_array_source = np.exp(ng_interp_source(self.z_array))
        ind_ltlz = np.where(self.z_array < 1e-2)[0]
        self.ng_array_source[ind_ltlz] = 0.0

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
        if 'pkzlin_interp' not in other_params.keys():
            if self.verbose:
                print('getting pkzlin interp')
            self.pkzlin_interp = self.ghmf.get_Pklin_zk_interp()
        else:
            self.pkzlin_interp = other_params['pkzlin_interp']

        if 'pkznl_interp' in other_params.keys():
            self.pkznl_interp = other_params['pkznl_interp']
        else:
            import ipdb; ipdb.set_trace() # BREAKPOINT

        self.suppress_1halo = other_params['suppress_1halo']
        self.kstar = other_params['kstar']

        self.add_beam_to_theory = other_params['add_beam_to_theory']
        self.beam_fwhm_arcmin = other_params['beam_fwhm_arcmin']
        if 'um_block_allinterp' in other_params.keys():
            self.um_block_allinterp = other_params['um_block_allinterp']
            self.bkm_block_allinterp = other_params['bkm_block_allinterp']

        if other_params['get_bp']:
            self.wplin_interp = other_params['wplin_interp']

        self.al_kk = other_params['kk_hm_trans'] 


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
        return value

    # get spherical harmonic transform of the galaxy distribution, eq 18 of Makiya et al
    def get_ug_l_zM(self, l):
        k_array = (l + 1. / 2.) / self.chi_array
        if self.use_only_halos:
            val = np.ones(self.Nc_mat.shape)
        else:
            ukzm_mat = hmf.get_ukmz_g_mat(self.r_max_mat, k_array, self.halo_conc_vir, self.rsg_rs)
            val = np.sqrt(
                (2 * self.Nc_mat * self.Ns_mat * ukzm_mat + (self.Nc_mat**2) * (self.Ns_mat ** 2) * (ukzm_mat ** 2)))
#            val = (self.Nc_mat * self.Ns_mat * ukzm_mat + self.Nc_mat) 

        coeff_mat = np.tile(
            (self.ng_array / ((self.chi_array ** 2) * self.dchi_dz_array * self.nbar)).reshape(self.nz, 1),
            (1, self.nm))
        if self.suppress_1halo:
            k_mat = np.tile(k_array.reshape(self.nz,1),(1,self.nm))
            suppress_fac =  (1. - np.exp(-1.*(k_mat/self.kstar)**4))
        else:
            suppress_fac = 1.
#        import ipdb; ipdb.set_trace() # BREAKPOINT

        return coeff_mat * val*suppress_fac


    def get_ug_cross_l_zM(self, l):
        k_array = (l + 1. / 2.) / self.chi_array
        if self.use_only_halos:
            val = np.ones(self.Nc_mat.shape)
        else:
            ukzm_mat = hmf.get_ukmz_g_mat(self.r_max_mat, k_array, self.halo_conc_vir, self.rsg_rs)
#            val = np.sqrt(
#                (2 * self.Nc_mat * self.Ns_mat * ukzm_mat + (self.Nc_mat**2) * (self.Ns_mat ** 2) * (ukzm_mat ** 2)))
            val = (self.Nc_mat * self.Ns_mat * ukzm_mat + self.Nc_mat) 

        coeff_mat = np.tile(
            (self.ng_array / ((self.chi_array ** 2) * self.dchi_dz_array * self.nbar)).reshape(self.nz, 1),
            (1, self.nm))
        if self.suppress_1halo:
            k_mat = np.tile(k_array.reshape(self.nz,1),(1,self.nm))
            suppress_fac =  (1. - np.exp(-1.*(k_mat/self.kstar)**4))
        else:
            suppress_fac = 1.
#        import ipdb; ipdb.set_trace() # BREAKPOINT

        return coeff_mat * val*suppress_fac

    # get spherical harmonic transform of the effective galaxy bias, eq 23 of Makiya et al
    def get_bg_l_z(self, l):
        k_array = (l + 1. / 2.) / self.chi_array

        if self.use_only_halos:

            toint = self.dndm_array * self.bm_array * self.M_mat_cond_inbin * self.int_prob
        else:
            ukzm_mat = hmf.get_ukmz_g_mat(self.r_max_mat, k_array, self.halo_conc_vir, self.rsg_rs)
            toint = self.dndm_array * (self.Nc_mat + self.Nc_mat*self.Ns_mat * ukzm_mat) * self.bm_array * self.M_mat_cond_inbin

        val = sp.integrate.simps(toint, self.M_array)

        coeff = self.ng_array / ((self.chi_array ** 2) * self.dchi_dz_array * self.nbar)
        # import ipdb; ipdb.set_trace()
        return coeff * val

    # # get spherical harmonic transform of the hot gas distribution, eq 12 of Makiya et al
    def get_uy_l_zM(self, l, x_mat_lmdefP_mat,x_mat2_y3d_mat,coeff_mat_y ):
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
        um_mat_normed = uml_zM_dict[round(l, 1)]
        coeff_mat = np.tile((self.Wk_array / self.chi_array ** 2).reshape(self.nz, 1), (1, self.nm))
        return coeff_mat * um_mat_normed

    # get spherical harmonic transform of the matter distribution
    def get_um_l_zM(self, l):
        k_array = (l + 1. / 2.) / self.chi_array
        if hasattr(self, 'um_block_allinterp'):
            ukzm_mat = np.zeros((self.nz, self.nm))
            for j in range(len(k_array)):
                kv = k_array[j]
                marray_rs = np.log(np.reshape(self.M_array, (1, self.nm, 1)))
                marray_insz = np.insert(marray_rs, 0, (self.z_array[j]), axis=-1)
                marray_insk = np.insert(marray_insz, 2, np.log(kv), axis=-1)[0]
                ukzm_mat[j, :] = np.exp(self.um_block_allinterp(marray_insk))
        else:
            ukzm_mat = (hmf.get_ukmz_g_mat(self.r_max_mat, k_array, self.halo_conc_vir, self.rsg_rs)) * self.M_mat / self.rho_m_bar

        if self.suppress_1halo:
            k_mat = np.tile(k_array.reshape(self.nz,1),(1,self.nm))
            suppress_fac =  (1. - np.exp(-1.*(k_mat/self.kstar)**2))
        else:
            suppress_fac = 1.

        return ukzm_mat * suppress_fac

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

            # val *= np.exp(self.bkm_block_allinterp(np.stack((self.z_array, np.log(k_array)), axis=-1)))

            val = np.exp(self.bkm_block_allinterp((self.z_array), np.log(k_array),grid=False))

        return val

    def get_Pkmm1h_zM(self, k_array, nu_mat, gm_mat, rhobar_M, um_block=None):
        nk = len(k_array)
        if um_block is None:
            arg_mat = np.meshgrid(self.z_array, np.log(self.M_array),np.log(k_array), indexing='ij')
            arg_mat_list = np.reshape(arg_mat, (3, -1), order='C').T
            ukzm_mat = np.exp(self.um_block_allinterp(arg_mat_list))
            ukzm_mat = ukzm_mat.reshape(self.nz, self.nm, nk)
        else:
            ukzm_mat = um_block
        ukzm_mat = ukzm_mat.transpose(0, 2, 1)
        dndm_mat = np.tile(self.dndm_array.reshape(self.nz, 1, self.nm), (1, nk, 1))
        Pk1h = sp.integrate.simps((ukzm_mat ** 2) * dndm_mat, self.M_array)
#        import ipdb; ipdb.set_trace() # BREAKPOINT

        Pk1h_bl = np.zeros(Pk1h.shape)
#        for j in range(len(self.z_array)):
#            gm_mat_j = np.tile(gm_mat[j,:].reshape(1,self.nm),(nk,1))
#            rhobar_M_j = np.tile(rhobar_M[j,:].reshape(1,self.nm),(nk,1))
#            Pk1h_bl[j,:] = sp.integrate.simps((ukzm_mat[j,:,:] ** 2) * gm_mat_j * rhobar_M_j, nu_mat[j,:])
#
        return Pk1h, Pk1h_bl

    def get_Pkmm2h_zM(self, k_array):
        Plin_kz = np.exp(self.pkzlin_interp((self.z_array), np.log(k_array), grid=True))
        if hasattr(self, 'bkm_block_allinterp'):
            nk = len(k_array)

            # arg_mat = np.meshgrid(np.log(self.z_array), np.log(k_array))
            # arg_mat_list = np.reshape(arg_mat, (2, -1), order='C').T
            # bkz_mat = np.exp(self.bkm_block_allinterp(arg_mat_list))

            bkz_mat = np.exp(self.bkm_block_allinterp((self.z_array), np.log(k_array),grid=True))

            bkz_mat = bkz_mat.reshape(self.nz, nk)
            # Plin_kz = np.exp(self.pkzlin_interp(np.log(self.z_array), np.log(k_array),grid=True))

            Pk2h = (bkz_mat**2) * Plin_kz
        else:
            Pk2h = Plin_kz
        return Pk2h

