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
from Powerspec import *
pi = np.pi



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

        self.fsky_gg = other_params['fsky_gg']
        self.fsky_yy = other_params['fsky_yy']
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
            self.uyl_zM_dict = other_params['uyl_zM_dict']
        else:
            ti = time.time()
            if self.verbose:
                print('getting y3d matrix')
#            global x_mat2_y3d_mat, x_mat_lmdefP_mat, coeff_mat_y
            y3d_mat = self.PS.Pressure.get_y3d(self.PS.M_mat_mdefP, self.PS.x_array, self.PS.z_array,
                                               self.PS.rmdefP_mat,
                                               M200c_mat=self.PS.M_mat_200cP, Mmat_cond=self.PS.M_mat_cond_inbin,
                                               zmat_cond=self.PS.z_mat_cond_inbin)

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
                print('getting uyl matrix')
                ti = time.time()

            self.uyl_zM_dict = {}
            for j in range(len(l_array)):
                self.uyl_zM_dict[round(l_array[j], 1)] = self.PS.get_uy_l_zM(l_array[j], x_mat_lmdefP_mat,x_mat2_y3d_mat,coeff_mat_y )


            if self.verbose:
                print('that took ', time.time() - ti, 'seconds')

        if self.verbose:
            print('getting uml matrix')
            ti = time.time()

        if 'uml_zM_dict' in other_params.keys():
            self.uml_zM_dict = other_params['uml_zM_dict']
        else:
            self.uml_zM_dict = {}
            if 'k' in self.lss_probes_analyze:
                for j in range(len(l_array)):
                    self.uml_zM_dict[round(l_array[j], 1)] = self.PS.get_um_l_zM(l_array[j])

        if self.verbose:
            print('that took ', time.time() - ti, 'seconds')

        if self.verbose:
            print('getting ugl, ukl matrix')
            ti = time.time()

        self.ugl_cross_zM_dict,self.ugl_zM_dict, self.ukl_zM_dict = {}, {}, {}
        for j in range(len(l_array)):
            if 'g' in self.lss_probes_analyze:
                self.ugl_zM_dict[round(l_array[j], 1)] = self.PS.get_ug_l_zM(l_array[j])
                self.ugl_cross_zM_dict[round(l_array[j], 1)] = self.PS.get_ug_cross_l_zM(l_array[j])
            if 'k' in self.lss_probes_analyze:
                self.ukl_zM_dict[round(l_array[j], 1)] = self.PS.get_uk_l_zM(l_array[j], self.uml_zM_dict)


        if self.verbose:
            print('that took ', time.time() - ti, 'seconds')


        if self.verbose:
            print('getting byl matrix')
            ti = time.time()

        if 'byl_z_dict' in other_params.keys():
            self.byl_z_dict = other_params['byl_z_dict']
        else:
            self.byl_z_dict = {}
            for j in range(len(l_array)):
                self.byl_z_dict[round(l_array[j], 1)] = self.PS.get_by_l_z(l_array[j], self.uyl_zM_dict)

        if self.verbose:
            print('that took ', time.time() - ti, 'seconds')


        if self.verbose:
            print('getting bml matrix')
            ti = time.time()

        if 'bml_z_dict' in other_params.keys():
            self.bml_z_dict = other_params['bml_z_dict']
        else:
            self.bml_z_dict = {}
            for j in range(len(l_array)):
                if 'k' in self.lss_probes_analyze:
                    self.bml_z_dict[round(l_array[j], 1)] = self.PS.get_bm_l_z(l_array[j])

        if self.verbose:
            print('that took ', time.time() - ti, 'seconds')

        if self.verbose:
            print('getting bgl, bkl matrix')
            ti = time.time()

        self.bgl_z_dict, self.bkl_z_dict = {}, {}
        for j in range(len(l_array)):
            if 'g' in self.lss_probes_analyze:
                self.bgl_z_dict[round(l_array[j], 1)] = self.PS.get_bg_l_z(l_array[j])
            if 'k' in self.lss_probes_analyze:
                self.bkl_z_dict[round(l_array[j], 1)] = self.PS.get_bk_l_z(l_array[j], self.bml_z_dict)

        if self.verbose:
            print('that took ', time.time() - ti, 'seconds')

        if self.verbose:
            print('done getting all the uk and bk')

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

        if 'uyl_zM_dict' not in other_params.keys():
            del x_mat2_y3d_mat, x_mat_lmdefP_mat, coeff_mat_y

#        savefname = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/compare_pkmm_cs_yx_wdndm.pk'
#        # getting the matter-matter power:
#        # um_block = other_params['um_block']
#        nu_mat, gm_mat, rhobar_M = other_params['nu_block'], other_params['gm_block'], other_params['rhobar_M_block']
#        k_array = other_params['k_array_block']
#        # import pdb; pdb.set_trace()
#        # Pkmm1h, Pkmm1h_block = self.PS.get_Pkmm1h_zM(k_array,nu_mat, gm_mat, rhobar_M, um_block=um_block)
#        print('getting 1h Pkmm')
#        Pkmm1h, Pkmm1h_block = self.PS.get_Pkmm1h_zM(k_array, nu_mat, gm_mat, rhobar_M)
#        # Pkmm1h = self.PS.get_Pkmm1h_zM(k_array, nu_mat, gm_mat, rhobar_M, um_block=um_block)
#        print('getting 2h Pkmm')
#        Pkmm2h = self.PS.get_Pkmm2h_zM(k_array)
#        Pkmmtot = Pkmm1h + Pkmm2h
#
#        outdict = {'k':k_array, 'z':self.PS.z_array,'Pk1h':Pkmm1h,'Pk2h':Pkmm2h,'Pktot':Pkmmtot, 'Pk1h_block':Pkmm1h_block,
#                'Pk1h_cs':other_params['pkmm1h_cs'],'Pk2h_cs':other_params['pkmm2h_cs'],'Pktot_cs':other_params['pkmmtot_cs'],'dndm':other_params['dndm_array'],'nu':other_params['nu_block'],'M':other_params['M_array'],'gnu':other_params['gm_block']}
#
#        with open(savefname, 'wb') as f:
#            dill.dump(outdict, f)
#        import pdb; pdb.set_trace()

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
#            import ipdb; ipdb.set_trace() # BREAKPOINT

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
            k_array = (l + (1. / 2.) ) / self.PS_prepDV.chi_array
            bgl_z1 = bAl_z_dict[round(l, 1)]
            bgl_z2 = bBl_z_dict[round(l, 1)]
            # toint_z = (bgl_z1 * bgl_z2) * (self.PS_prepDV.chi_array ** 2) * self.PS_prepDV.dchi_dz_array * np.exp(
            #     self.PS_prepDV.pkzlin_interp.ev(np.log(self.PS_prepDV.z_array), np.log(k_array))) * toint_z_multfac
            toint_z = (bgl_z1 * bgl_z2) * (self.PS_prepDV.chi_array ** 2) * self.PS_prepDV.dchi_dz_array * np.exp(
                self.PS_prepDV.pkzlin_interp.ev(self.PS_prepDV.z_array, np.log(k_array))) * toint_z_multfac
            val = sp.integrate.simps(toint_z, self.PS_prepDV.z_array)
            Cl_2h[j] = val
        return Cl_2h

    def get_Cl_AB_2h_nl(self, A, B, l_array, bAl_z_dict, bBl_z_dict):
        g_sum = (A == 'g') + (B == 'g')
        if g_sum > 0:
            toint_z_multfac = self.PS_prepDV.z_array_cond_inbin
        else:
            toint_z_multfac = 1.
        Cl_2h = np.zeros_like(l_array)
        for j in range(len(l_array)):
            l = l_array[j]
            k_array = (l + (1. / 2.) ) / self.PS_prepDV.chi_array
            bgl_z1 = bAl_z_dict[round(l, 1)]
            bgl_z2 = bBl_z_dict[round(l, 1)]
            toint_z = (bgl_z1 * bgl_z2) * (self.PS_prepDV.chi_array ** 2) * self.PS_prepDV.dchi_dz_array * np.exp(
                self.PS_prepDV.pkznl_interp.ev(self.PS_prepDV.z_array, np.log(k_array))) * toint_z_multfac
            val = sp.integrate.simps(toint_z, self.PS_prepDV.z_array)
            Cl_2h[j] = val
        return Cl_2h

    def get_Cl_AB_tot_modelNL(self, A, B, l_array, uAl_zM_dict, uBl_zM_dict,bAl_z_dict, bBl_z_dict):
        g_sum = (A == 'g') + (B == 'g')
        toint_M_multfac = 1.
        toint_z_multfac = 1.
        Cl_tot = np.zeros_like(l_array)
        for j in range(len(l_array)):
            l = l_array[j]
            uAl_zM = uAl_zM_dict[round(l, 1)]
            if g_sum == 2:
                if self.PS_prepDV.use_only_halos:
                    val_z1h = np.zeros_like(self.PS_prepDV.z_array)   
                else:
                    toint_M_multfac = self.PS_prepDV.M_mat_cond_inbin
                    toint_z_multfac = self.PS_prepDV.z_array_cond_inbin
            elif g_sum == 1:
                toint_M_multfac = self.PS_prepDV.M_mat_cond_inbin * self.PS_prepDV.int_prob
                toint_z_multfac = self.PS_prepDV.z_array_cond_inbin
            else:
                toint_M_multfac = 1.
                toint_z_multfac = 1.
            uBl_zM = uBl_zM_dict[round(l, 1)]
            toint_M = (uAl_zM * uBl_zM) * self.PS_prepDV.dndm_array * toint_M_multfac

            if (g_sum < 2) or (not self.PS_prepDV.use_only_halos):
                val_z1h = sp.integrate.simps(toint_M, self.PS_prepDV.M_array)
            k_array = (l + (1. / 2.) ) / self.PS_prepDV.chi_array
            bgl_z1 = bAl_z_dict[round(l, 1)]
            bgl_z2 = bBl_z_dict[round(l, 1)]
            val_z2h = (bgl_z1 * bgl_z2) *  np.exp(self.PS_prepDV.pkznl_interp.ev(self.PS_prepDV.z_array, np.log(k_array)))
            val_z = np.maximum(val_z1h,val_z2h)

            toint_z = val_z * (self.PS_prepDV.chi_array ** 2) * self.PS_prepDV.dchi_dz_array * toint_z_multfac
            val = sp.integrate.simps(toint_z, self.PS_prepDV.z_array)
            Cl_tot[j] = val
        return Cl_tot


    def get_Cl_AB_tot_modelmead(self, A, B, l_array, uAl_zM_dict, uBl_zM_dict,bAl_z_dict, bBl_z_dict):
        g_sum = (A == 'g') + (B == 'g')
        toint_M_multfac = 1.
        toint_z_multfac = 1.
        Cl_tot = np.zeros_like(l_array)
        for j in range(len(l_array)):
            l = l_array[j]
            uAl_zM = uAl_zM_dict[round(l, 1)]
            if g_sum == 2:
                if self.PS_prepDV.use_only_halos:
                    val_z1h = np.zeros_like(self.PS_prepDV.z_array)   
                else:
                    toint_M_multfac = self.PS_prepDV.M_mat_cond_inbin
                    toint_z_multfac = self.PS_prepDV.z_array_cond_inbin
            elif g_sum == 1:
                toint_M_multfac = self.PS_prepDV.M_mat_cond_inbin * self.PS_prepDV.int_prob
                toint_z_multfac = self.PS_prepDV.z_array_cond_inbin
            else:
                toint_M_multfac = 1.
                toint_z_multfac = 1.
            uBl_zM = uBl_zM_dict[round(l, 1)]
            toint_M = (uAl_zM * uBl_zM) * self.PS_prepDV.dndm_array * toint_M_multfac

            if (g_sum < 2) or (not self.PS_prepDV.use_only_halos):
                val_z1h = sp.integrate.simps(toint_M, self.PS_prepDV.M_array)
            k_array = (l + (1. / 2.) ) / self.PS_prepDV.chi_array
            bgl_z1 = bAl_z_dict[round(l, 1)]
            bgl_z2 = bBl_z_dict[round(l, 1)]
            val_z2h = (bgl_z1 * bgl_z2) *  np.exp(self.PS_prepDV.pkznl_interp.ev(self.PS_prepDV.z_array, np.log(k_array)))
            val_z = (val_z1h**al + val_z2h**al)**(1./al)

            toint_z = val_z * (self.PS_prepDV.chi_array ** 2) * self.PS_prepDV.dchi_dz_array * toint_z_multfac
            val = sp.integrate.simps(toint_z, self.PS_prepDV.z_array)
            Cl_tot[j] = val
        return Cl_tot



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

        fsky_j = np.sqrt(fsky_dict[A + B] * fsky_dict[C + D])

        val_diag = (1. / (fsky_j * (2 * Cl_result_dict['l_array_survey'] + 1.) * Cl_result_dict['dl_array_survey'])) * (
                Cl_stats_dict[A + C] * Cl_stats_dict[B + D] + Cl_stats_dict[A + D] * Cl_stats_dict[B + C])

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
#        import ipdb; ipdb.set_trace() # BREAKPOINT

        if theta_array_arcmin is not None:
            xi_interp = interpolate.interp1d(np.log(theta_out_arcmin), np.log(xi_out), fill_value='extrapolate',
                                             bounds_error=False)
            xi_final = np.exp(xi_interp(np.log(theta_array_arcmin)))
        else:
            xi_final = xi_out
            theta_array_arcmin = theta_out_arcmin
        return xi_final, theta_array_arcmin
