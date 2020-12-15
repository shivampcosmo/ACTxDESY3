import sys, os
import traceback
import pdb
import numpy as np
import scipy as sp
import healpy as hp
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

from PrepDataVec import *

pi = np.pi


class DataVec:
    def __init__(self, PrepDV_params, block):
        self.CalcDV = CalcDataVec(PrepDV_params)
        beam_fwhm_arcmin = self.CalcDV.PS_prepDV.beam_fwhm_arcmin
        addbth = self.CalcDV.add_beam_to_theory
        addpw = self.CalcDV.add_pixwin_to_theory
        nside_pw = self.CalcDV.PS_prepDV.nside_pixwin
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
        put_IA = PrepDV_params['put_IA']
        only2h_IA = PrepDV_params['only_2h_IA']
        model_2h_IA = PrepDV_params['model_2h_IA']
        save_detailed_DV = PrepDV_params['save_detailed_DV']
        do_tomo_Bhse = PrepDV_params['do_tomo_Bhse']
        self.Cl_result_dict = {'l_array': PrepDV.l_array, 'l_array_survey': PrepDV.l_array_survey,
                               'ind_select_survey': PrepDV.ind_select_survey,
                               'dl_array_survey': PrepDV.dl_array_survey}
        if analysis_coords == 'real':
            self.xi_result_dict = {}

        if self.verbose:
            print('starting Cls calculation')
        if ('kk' in PrepDV.stats_analyze) or (run_cov_pipe and ('kk' in PrepDV.lss_probes_allcomb)):
            Cl_kk_dict = {}
            bin_combs = []
            if analysis_coords == 'real':
                xi_kk_dict = {}
            for j1 in bins_source:
                for j2 in bins_source:
                    if j2 >= j1:

                        if PrepDV_params['kk_1h2h_model'] == 'sum_normal':
                            Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('k', 'k', PrepDV.l_array,
                                                                 PrepDV_params['ukl_zM_dict' + str(j1)],
                                                                 PrepDV_params['ukl_zM_dict' + str(j2)])
                            Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('k', 'k', PrepDV.l_array,
                                                                 PrepDV_params['bkl_z_dict' + str(j1)],
                                                                 PrepDV_params['bkl_z_dict' + str(j2)])
                            Cltotphy_j1j2 = self.CalcDV.get_Cl_AB_tot('k', 'k', Cl1h_j1j2, Cl2h_j1j2)

                        if PrepDV_params['kk_1h2h_model'] == 'sum_fsmooth':
                            if PrepDV_params['kk_alpha_1h2h_model' + str(j1)] == 0.0:
                                alvj = self.CalcDV.PS_prepDV.al_kk
                            else:
                                alvj = PrepDV_params['kk_alpha_1h2h_model' + str(j1)]
                            Cltotphy_j1j2 = self.CalcDV.get_Cl_AB_tot_modelmead('k', 'k', PrepDV.l_array,
                                                                                PrepDV_params['ukl_zM_dict' + str(j1)],
                                                                                PrepDV_params['ukl_zM_dict' + str(j2)],
                                                                                PrepDV_params['bkl_z_dict' + str(j1)],
                                                                                PrepDV_params['bkl_z_dict' + str(j2)],
                                                                                al = alvj)

                        if PrepDV_params['kk_1h2h_model'] == 'larger_fourier':
                            Cltotphy_j1j2 = self.CalcDV.get_Cl_AB_tot_modelNL('k', 'k', PrepDV.l_array,
                                                                              PrepDV_params['ukl_zM_dict' + str(j1)],
                                                                              PrepDV_params['ukl_zM_dict' + str(j2)],
                                                                              PrepDV_params['bkl_z_dict' + str(j1)],
                                                                              PrepDV_params['bkl_z_dict' + str(j2)])

                        if put_IA:
                            ClII2h_j1j2 = self.CalcDV.get_Cl_AB_2h('I', 'I', PrepDV.l_array,
                                                                   PrepDV_params['bIl_z_dict' + str(j1)],
                                                                   PrepDV_params['bIl_z_dict' + str(j2)],
                                                                   model_2h=model_2h_IA)
                            ClGI2h_j1j2 = self.CalcDV.get_Cl_AB_2h('G', 'I', PrepDV.l_array,
                                                                   PrepDV_params['bIl_z_dict' + str(j1)],
                                                                   PrepDV_params['bkl_z_dict' + str(j2)],
                                                                   model_2h=model_2h_IA)
                            ClGI2h_j2j1 = self.CalcDV.get_Cl_AB_2h('G', 'I', PrepDV.l_array,
                                                                   PrepDV_params['bkl_z_dict' + str(j1)],
                                                                   PrepDV_params['bIl_z_dict' + str(j2)],
                                                                   model_2h=model_2h_IA)
                            Clintrinsic2h_j1j2 = ClII2h_j1j2 + ClGI2h_j1j2 + ClGI2h_j2j1
                            if only2h_IA:
                                Clintrinsic_j1j2 = Clintrinsic2h_j1j2
                            else:
                                ClII1h_j1j2 = self.CalcDV.get_Cl_AB_1h('I', 'I', PrepDV.l_array,
                                                                       PrepDV_params['uIl_zM_dict' + str(j1)],
                                                                       PrepDV_params['uIl_zM_dict' + str(j2)])
                                ClGI1h_j1j2 = self.CalcDV.get_Cl_AB_1h('G', 'I', PrepDV.l_array,
                                                                       PrepDV_params['uIl_zM_dict' + str(j1)],
                                                                       PrepDV_params['ukl_zM_dict' + str(j2)])
                                ClGI1h_j2j1 = self.CalcDV.get_Cl_AB_1h('G', 'I', PrepDV.l_array,
                                                                       PrepDV_params['ukl_zM_dict' + str(j1)],
                                                                       PrepDV_params['uIl_zM_dict' + str(j2)])

                                Clintrinsic1h_j1j2 = ClII1h_j1j2 + ClGI1h_j1j2 + ClGI1h_j2j1
                                Clintrinsic_j1j2 = Clintrinsic1h_j1j2 + Clintrinsic2h_j1j2

                            # ClII_j1j2 = self.CalcDV.get_Cl_AB_2h_nl('I', 'I', PrepDV.l_array,
                            # PrepDV_params['bIl_z_dict' + str(j1)],
                            # PrepDV_params['bIl_z_dict' + str(j2)])
                            # ClGI_j1j2 = self.CalcDV.get_Cl_AB_2h_nl('G', 'I', PrepDV.l_array,
                            # PrepDV_params['bIl_z_dict' + str(j1)],
                            # PrepDV_params['bkl_z_dict' + str(j2)])
                            # ClGI_j2j1 = self.CalcDV.get_Cl_AB_2h_nl('G', 'I', PrepDV.l_array,
                            # PrepDV_params['bkl_z_dict' + str(j1)],
                            # PrepDV_params['bIl_z_dict' + str(j2)])

                            # Clintrinsic_j1j2 =  ClII_j1j2 + ClGI_j1j2 + ClGI_j2j1
                            Cltot_j1j2 = Cltotphy_j1j2 + Clintrinsic_j1j2
                        else:
                            Cltot_j1j2 = Cltotphy_j1j2

                        if j1 == j2:
                            Cl_noise_ellsurvey = PrepDV_params['Cl_noise_kk_l_array' + str(j1)]
                        else:
                            Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                        bin_combs.append([j1, j2])
                        #                        Cl_kk_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2,
                        #                                                                        'tot': Cltot_j1j2,
                        #                                                                        'tot_mead': Cltotmead_j1j2,
                        #                                                                        'tot_ellsurvey': Cltot_j1j2[
                        #                                                                            PrepDV.ind_select_survey],
                        #                                                                        'tot_plus_noise_ellsurvey': Cltot_j1j2[
                        #                                                                                                        PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                        #
                        if save_detailed_DV:
                            if put_IA:
                                if only2h_IA:
                                    Cl_kk_dict['bin_' + str(j1) + '_' + str(j2)] = {'tot_phy': Cltotphy_j1j2,
                                                                                    'tot_II': ClII2h_j1j2,
                                                                                    'tot_GI': ClGI2h_j1j2 + ClGI2h_j2j1,
                                                                                    'tot': Cltot_j1j2,
                                                                                    'tot_ellsurvey': Cltot_j1j2[
                                                                                        PrepDV.ind_select_survey],
                                                                                    'tot_plus_noise_ellsurvey':
                                                                                        Cltot_j1j2[
                                                                                            PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                                else:
                                    Cl_kk_dict['bin_' + str(j1) + '_' + str(j2)] = {'tot_phy': Clphy_j1j2,
                                                                                    'tot_II': ClII1h_j1j2 + ClII2h_j1j2,
                                                                                    'tot_GI': ClGI1h_j1j2 + ClGI1h_j2j1 + ClGI2h_j1j2 + ClGI2h_j2j1,
                                                                                    'tot': Cltot_j1j2,
                                                                                    'tot_ellsurvey': Cltot_j1j2[
                                                                                        PrepDV.ind_select_survey],
                                                                                    'tot_plus_noise_ellsurvey':
                                                                                        Cltot_j1j2[
                                                                                            PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                            else:
                                Cl_kk_dict['bin_' + str(j1) + '_' + str(j2)] = {'tot': Cltotphy_j1j2,
                                                                                'tot_ellsurvey': Cltot_j1j2[
                                                                                    PrepDV.ind_select_survey],
                                                                                'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                                                                PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                        else:
                            Cl_kk_dict['bin_' + str(j1) + '_' + str(j2)] = {'tot': Cltotphy_j1j2,
                                                                            'tot_ellsurvey': Cltot_j1j2[
                                                                                PrepDV.ind_select_survey],
                                                                            'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                                                            PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                        if analysis_coords == 'real':
                            #                            xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                            #                                                                                      Cltot_j1j2,
                            #                                                                                      theta_array_arcmin=theta_array_arcmin)

                            xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                      Cltot_j1j2,
                                                                                      theta_array_arcmin=theta_array_arcmin)

                            ximtot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(4, PrepDV.l_array,
                                                                                       Cltot_j1j2,
                                                                                       theta_array_arcmin=theta_array_arcmin)
                            #                            xi_kk_dict['bin_' + str(j1) + '_' + str(j2)] = xitot_j1j2
                            #                            xi_kk_dict['bin_' + str(j1) + '_' + str(j2)] =  {'1h':xi1h_j1j2,'2h':xi2h_j1j2,'tot':xitot_j1j2,
                            #                                            'tot2':xitotmead_j1j2}

                            if save_detailed_DV:
                                if put_IA:
                                    xiGG_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                             Cltotphy_j1j2,
                                                                                             theta_array_arcmin=theta_array_arcmin)
                                    if not only2h_IA:
                                        xi1hint_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                                    Clintrinsic1h_j1j2,
                                                                                                    theta_array_arcmin=theta_array_arcmin)
                                    xi2hint_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                                Clintrinsic2h_j1j2,
                                                                                                theta_array_arcmin=theta_array_arcmin)
                                    xiint_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                              Clintrinsic_j1j2,
                                                                                              theta_array_arcmin=theta_array_arcmin)

                                    ximGG_j1j2, theta_array = self.CalcDV.do_Hankel_transform(4, PrepDV.l_array,
                                                                                              Cltotphy_j1j2,
                                                                                              theta_array_arcmin=theta_array_arcmin)
                                    if not only2h_IA:
                                        xim1hint_j1j2, theta_array = self.CalcDV.do_Hankel_transform(4, PrepDV.l_array,
                                                                                                     Clintrinsic1h_j1j2,
                                                                                                     theta_array_arcmin=theta_array_arcmin)
                                    xim2hint_j1j2, theta_array = self.CalcDV.do_Hankel_transform(4, PrepDV.l_array,
                                                                                                 Clintrinsic2h_j1j2,
                                                                                                 theta_array_arcmin=theta_array_arcmin)
                                    ximint_j1j2, theta_array = self.CalcDV.do_Hankel_transform(4, PrepDV.l_array,
                                                                                               Clintrinsic_j1j2,
                                                                                               theta_array_arcmin=theta_array_arcmin)
                                    # import ipdb; ipdb.set_trace() # BREAKPOINT

                                    if not only2h_IA:
                                        xi_kk_dict['bin_' + str(j1) + '_' + str(j2)] = {'phy': xiGG_j1j2,
                                                                                        'int': xiint_j1j2,
                                                                                        '1hint': xi1hint_j1j2,
                                                                                        '2hint': xi2hint_j1j2,
                                                                                        'tot': xitot_j1j2,
                                                                                        'totm': ximtot_j1j2,
                                                                                        'phym': ximGG_j1j2,
                                                                                        'intm': ximint_j1j2,
                                                                                        '1hintm': xim1hint_j1j2,
                                                                                        '2hintm': xim2hint_j1j2}
                                    else:
                                        xi_kk_dict['bin_' + str(j1) + '_' + str(j2)] = {'phy': xiGG_j1j2,
                                                                                        'int': xiint_j1j2,
                                                                                        '2hint': xi2hint_j1j2,
                                                                                        'tot': xitot_j1j2,
                                                                                        'totm': ximtot_j1j2,
                                                                                        'phym': ximGG_j1j2,
                                                                                        'intm': ximint_j1j2,
                                                                                        '2hintm': xim2hint_j1j2}
                                else:
                                    xi_kk_dict['bin_' + str(j1) + '_' + str(j2)] = {'tot': xitot_j1j2,
                                                                                    'totm': ximtot_j1j2}
                            else:
                                xi_kk_dict['bin_' + str(j1) + '_' + str(j2)] = {'tot': xitot_j1j2, 'totm': ximtot_j1j2}

                            if 'theta' not in xi_kk_dict.keys():
                                xi_kk_dict['theta'] = theta_array
                            if 'kk' in PrepDV.stats_analyze:
                                block[sec_save_name, 'theory_corrf_' + 'kk' + '_' + 'bin_' + str(j1) + '_' + str(
                                    j2)] = xitot_j1j2
                                block[sec_save_name, 'xcoord_' + 'kk' + '_' + 'bin_' + str(j1) + '_' + str(
                                    j2)] = theta_array
                                block[sec_save_name, 'xcoord_' + 'kkm' + '_' + 'bin_' + str(j1) + '_' + str(
                                    j2)] = theta_array
                                block[sec_save_name, 'theory_corrf_' + 'kkm' + '_' + 'bin_' + str(j1) + '_' + str(
                                    j2)] = ximtot_j1j2
                        else:
                            if 'kk' in PrepDV.stats_analyze:
                                block[sec_save_name, 'theory_corrf_' + 'kk' + '_' + 'bin_' + str(j1) + '_' + str(
                                    j2)] = Cltot_j1j2
                                block[sec_save_name, 'xcoord_' + 'kk' + '_' + 'bin_' + str(j1) + '_' + str(
                                    j2)] = PrepDV.l_array

            Cl_kk_dict['bin_combs'] = bin_combs
            self.Cl_result_dict['kk'] = Cl_kk_dict
            if analysis_coords == 'real':
                xi_kk_dict['bin_combs'] = bin_combs
                xi_kk_dict['theta'] = theta_array
                self.xi_result_dict['kk'] = xi_kk_dict

            if self.verbose:
                print('done shear-shear calculation')

        if ('ky' in PrepDV.stats_analyze) or (run_cov_pipe and ('ky' in PrepDV.lss_probes_allcomb)):
            Cl_ky_dict = {}
            if analysis_coords == 'real':
                # xi_ky_dict = {}
                xi_gty_dict = {}
                self.xi_gty_log_sens_zM = {}
            bin_combs = []
            for j1 in bins_source:
                try:
                    if do_tomo_Bhse:
                        yind = j1
                    else:
                        yind = 0
                    if (PrepDV_params['ky_1h2h_model'] in ['sum_normal']) or save_detailed_DV:
                        Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('k', 'y', PrepDV.l_array,
                                                             PrepDV_params['ukl_zM_dict' + str(j1)],
                                                             PrepDV_params['uyl_zM_dict' + str(yind)])
                        Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('k', 'y', PrepDV.l_array,
                                                             PrepDV_params['bkl_z_dict' + str(j1)],
                                                             PrepDV_params['byl_z_dict' + str(yind)])
                        Cltotphy_j1j2 = self.CalcDV.get_Cl_AB_tot('k', 'y', Cl1h_j1j2, Cl2h_j1j2)
                    if (PrepDV_params['ky_1h2h_model'] in ['larger_real']) or save_detailed_DV:                        
                        Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('k', 'y', PrepDV.l_array,
                                                             PrepDV_params['ukl_zM_dict' + str(j1)],
                                                             PrepDV_params['uyl_zM_dict' + str(yind)])
                        Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('k', 'y', PrepDV.l_array,
                                                             PrepDV_params['bkl_z_dict' + str(j1)],
                                                             PrepDV_params['byl_z_dict' + str(yind)], model_2h='nl')
                        Cltotphy_j1j2 = self.CalcDV.get_Cl_AB_tot('k', 'y', Cl1h_j1j2, Cl2h_j1j2)
                    if PrepDV_params['ky_1h2h_model'] == 'sum_fsmooth':
                        # import pdb; pdb.set_trace()
                        if PrepDV_params['ky_alpha_1h2h_model' + str(j1)] == 0.0:
                            alvj = self.CalcDV.PS_prepDV.al_kk
                        else:
                            alvj = PrepDV_params['ky_alpha_1h2h_model' + str(j1)]
                        Cltotphy_j1j2 = self.CalcDV.get_Cl_AB_tot_modelmead('k', 'y', PrepDV.l_array,
                                                                            PrepDV_params['ukl_zM_dict' + str(j1)],
                                                                            PrepDV_params['uyl_zM_dict' + str(yind)],
                                                                            PrepDV_params['bkl_z_dict' + str(j1)],
                                                                            PrepDV_params['byl_z_dict' + str(yind)],
                                                                                al = alvj)

                    if PrepDV_params['ky_1h2h_model'] == 'larger_fourier':
                        Cltotphy_j1j2 = self.CalcDV.get_Cl_AB_tot_modelNL('k', 'y', PrepDV.l_array,
                                                                          PrepDV_params['ukl_zM_dict' + str(j1)],
                                                                          PrepDV_params['uyl_zM_dict' + str(yind)],
                                                                          PrepDV_params['bkl_z_dict' + str(j1)],
                                                                          PrepDV_params['byl_z_dict' + str(yind)])

                    if put_IA:

                        ClGI2h_j1j2 = self.CalcDV.get_Cl_AB_2h('I', 'y', PrepDV.l_array,
                                                               PrepDV_params['bIl_z_dict' + str(j1)],
                                                               PrepDV_params['byl_z_dict' + str(yind)], model_2h=model_2h_IA)

                        if only2h_IA:
                            Clintrinsic_j1j2 = ClGI2h_j1j2
                            ClGI_j1j2 = ClGI2h_j1j2
                        else:
                            ClGI1h_j1j2 = self.CalcDV.get_Cl_AB_1h('I', 'y', PrepDV.l_array,
                                                                   PrepDV_params['uIl_zM_dict' + str(j1)],
                                                                   PrepDV_params['uyl_zM_dict' + str(yind)])
                            ClGI_j1j2 = self.CalcDV.get_Cl_AB_tot('I', 'y', ClGI1h_j1j2, ClGI2h_j1j2)

                            Clintrinsic_j1j2 = ClGI_j1j2
                        Cltot_j1j2 = Cltotphy_j1j2 + Clintrinsic_j1j2
                    else:
                        Cltot_j1j2 = Cltotphy_j1j2
                    Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                    bin_combs.append([j1, 0])

                    for jb in range(len(beam_fwhm_arcmin)):

                        sig_beam = beam_fwhm_arcmin[jb] * (1. / 60.) * (np.pi / 180.) * (1. / np.sqrt(8. * np.log(2)))
                        Bl = (np.exp(
                            -1. * PrepDV.l_array_survey * (PrepDV.l_array_survey + 1) * (sig_beam ** 2) / 2.)) ** (
                                 addbth) + 1e-200
                        pw_ip = interpolate.interp1d(np.log(np.arange(3 * nside_pw[jb])),
                                                     np.log(hp.pixwin(nside_pw[jb])), fill_value='extrapolate')
                        Bl *= (np.exp(pw_ip(np.log(PrepDV.l_array_survey)))) ** (addpw) + 1e-200

                        if save_detailed_DV:
                            if put_IA:
                                Cl_ky_dict['yt_' + str(jb + 1) + 'bin_' + str(j1) + '_' + str(0)] = {
                                    'tot_phy': Cltotphy_j1j2 * Bl, 'tot_GI': ClGI_j1j2 * Bl,
                                    'tot': Cltot_j1j2 * Bl,
                                    'tot_ellsurvey': (Cltot_j1j2 * Bl)[
                                        PrepDV.ind_select_survey],
                                    'tot_plus_noise_ellsurvey': (Cltot_j1j2 * Bl)[
                                                                    PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                            else:
                                Cl_ky_dict['yt_' + str(jb + 1) + 'bin_' + str(j1) + '_0'] = {'tot': Cltot_j1j2 * Bl,
                                                                                             '1h': Cl1h_j1j2 * Bl,
                                                                                             '2h': Cl2h_j1j2 * Bl,
                                                                                             'tot_ellsurvey':
                                                                                                 (Cltot_j1j2 * Bl)[
                                                                                                     PrepDV.ind_select_survey],
                                                                                             'tot_plus_noise_ellsurvey':
                                                                                                 (Cltot_j1j2 * Bl)[
                                                                                                     PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                        else:
                            Cl_ky_dict['yt_' + str(jb + 1) + 'bin_' + str(j1) + '_0'] = {'tot': Cltot_j1j2 * Bl,
                                                                                         'tot_ellsurvey':
                                                                                             (Cltot_j1j2 * Bl)[
                                                                                                 PrepDV.ind_select_survey],
                                                                                         'tot_plus_noise_ellsurvey':
                                                                                             (Cltot_j1j2 * Bl)[
                                                                                                 PrepDV.ind_select_survey] + Cl_noise_ellsurvey}

                        if analysis_coords == 'real':
                            if (PrepDV_params['ky_1h2h_model'] not in ['larger_real']):
                                gt_tot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                        Cltot_j1j2 * Bl,
                                                                                        theta_array_arcmin=theta_array_arcmin)
                            else:
                                gt1h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                         Cl1h_j1j2 * Bl,
                                                                                         theta_array_arcmin=theta_array_arcmin)

                                gt2h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                         (Cl2h_j1j2 + ClGI2h_j1j2) * Bl,
                                                                                         theta_array_arcmin=theta_array_arcmin)
                                # import ipdb; ipdb.set_trace()
                                if not np.any(gt1h_j1j2 > 0): 
                                    ind_all = np.arange(len(gt1h_j1j2))
                                    ind_gt0 = np.where(gt1h_j1j2 > 0)[0]
                                    ind_ngt0 = np.setdiff1d(np.union1d(ind_all, ind_gt0), np.intersect1d(ind_all, ind_gt0))
                                    gt1h_j1j2[ind_ngt0]=0.0
                                if not np.any(gt2h_j1j2 > 0):
                                    ind_all = np.arange(len(gt2h_j1j2))
                                    ind_gt0 = np.where(gt2h_j1j2 > 0)[0]
                                    ind_ngt0 = np.setdiff1d(np.union1d(ind_all, ind_gt0), np.intersect1d(ind_all, ind_gt0))
                                    gt2h_j1j2[ind_ngt0]=0.0
                                # import ipdb; ipdb.set_trace()
                                gt_tot_j1j2 = np.maximum(gt1h_j1j2,gt2h_j1j2)

                            if save_detailed_DV:
                                gt1h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                         Cl1h_j1j2 * Bl,
                                                                                         theta_array_arcmin=theta_array_arcmin)

                                gt2h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                         Cl2h_j1j2 * Bl,
                                                                                         theta_array_arcmin=theta_array_arcmin)

                            if save_detailed_DV:
                                if put_IA:
                                    xiphy_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                              Cltotphy_j1j2 * Bl,
                                                                                              theta_array_arcmin=theta_array_arcmin)
                                    xiint_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                              Clintrinsic_j1j2 * Bl,
                                                                                              theta_array_arcmin=theta_array_arcmin)

                                    xi2hint_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                                ClGI2h_j1j2 * Bl,
                                                                                                theta_array_arcmin=theta_array_arcmin)
                                    if not only2h_IA:
                                        xi1hint_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                                    ClGI1h_j1j2 * Bl,
                                                                                                    theta_array_arcmin=theta_array_arcmin)
                                        xi_gty_dict['yt_' + str(jb + 1) + 'bin_' + str(j1) + '_' + str(0)] = {
                                            'phy': xiphy_j1j2, 'int': xiint_j1j2, 'tot': gt_tot_j1j2,
                                            '1hint': xi1hint_j1j2, '2hint': xi2hint_j1j2, '1h': gt1h_j1j2, '2h': gt2h_j1j2}
                                    else:
                                        xi_gty_dict['yt_' + str(jb + 1) + 'bin_' + str(j1) + '_' + str(0)] = {
                                            'phy': xiphy_j1j2, 'int': xiint_j1j2, 'tot': gt_tot_j1j2,
                                            '2hint': xi2hint_j1j2,'1h': gt1h_j1j2, '2h': gt2h_j1j2}
                                else:
                                    xi_gty_dict['yt_' + str(jb + 1) + 'bin_' + str(j1) + '_' + str(0)] = {
                                        'tot': gt_tot_j1j2, '1h': gt1h_j1j2, '2h': gt2h_j1j2}
                            else:
                                xi_gty_dict['yt_' + str(jb + 1) + 'bin_' + str(j1) + '_' + str(0)] = {
                                    'tot': gt_tot_j1j2}

                            if 'theta' not in xi_gty_dict.keys():
                                # xi_ky_dict['theta'] = theta_array
                                xi_gty_dict['theta'] = theta_array
                            if 'ky' in PrepDV.stats_analyze:

                                block[
                                    sec_save_name, 'theory_corrf_' + 'gty' + str(jb + 1) + '_' + 'bin_' + str(
                                        j1) + '_' + str(0)] = gt_tot_j1j2

                                block[
                                    sec_save_name, 'xcoord_' + 'gty' + str(jb + 1) + '_' + 'bin_' + str(j1) + '_' + str(
                                        0)] = theta_array
                            
                            if PrepDV_params['get_logsens_zM']: 

                                Bl_array = np.ones_like(Bl)                    
                                ratioM, theta_out_arcmin, M_array, z_array = self.CalcDV.get_dlnxigty1h_AB_dlnM(PrepDV.l_array, PrepDV_params['ukl_zM_dict' + str(j1)],
                                                                          PrepDV_params['uyl_zM_dict0'], Bl_array)
                                # import pdb; pdb.set_trace()  
                                ratioz, theta_out_arcmin, M_array, z_array = self.CalcDV.get_dlnxigty1h_AB_dlnz(PrepDV.l_array, PrepDV_params['ukl_zM_dict' + str(j1)],
                                                                          PrepDV_params['uyl_zM_dict0'], Bl_array)
                                # import pdb; pdb.set_trace()                                                               
                                # ratio, theta_out_arcmin, M_mat, z_mat = self.CalcDV.get_d2lnxigty1h_AB_dlnMdlnz(PrepDV.l_array, PrepDV_params['ukl_zM_dict' + str(j1)],
                                #                                           PrepDV_params['uyl_zM_dict0'], Bl_array)
                                self.xi_gty_log_sens_zM['sensz_' + str(jb + 1) + 'bin_' + str(j1) + '_' + str(0)] = ratioz
                                self.xi_gty_log_sens_zM['sensM_' + str(jb + 1) + 'bin_' + str(j1) + '_' + str(0)] = ratioM
                                if jb == 0 and j1 == 1:
                                   self.xi_gty_log_sens_zM['theta'] = theta_out_arcmin
                                   self.xi_gty_log_sens_zM['M_array'] = M_array
                                   self.xi_gty_log_sens_zM['z_array'] = z_array
                                # import pdb; pdb.set_trace();

                        else:
                            if 'ky' in PrepDV.stats_analyze:
                                block[
                                    sec_save_name, 'theory_corrf_' + 'gty' + str(jb + 1) + '_' + 'bin_' + str(
                                        j1) + '_' + str(0)] = Cltot_j1j2
                                block[
                                    sec_save_name, 'xcoord_' + 'gty' + str(jb + 1) + '_' + 'bin_' + str(j1) + '_' + str(
                                        0)] = PrepDV.l_array

                except:
                    print(traceback.format_exc())

            Cl_ky_dict['bin_combs'] = bin_combs
            self.Cl_result_dict['ky'] = Cl_ky_dict
            # Cl_result_dict['yk'] = Cl_ky_dict
            if analysis_coords == 'real':
                # xi_result_dict['ky'] = xi_ky_dict
                xi_gty_dict['theta'] = theta_array
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

            for jb in range(len(beam_fwhm_arcmin)):
                sig_beam = beam_fwhm_arcmin[jb] * (1. / 60.) * (np.pi / 180.) * (1. / np.sqrt(8. * np.log(2)))
                Bl = (np.exp(-1. * PrepDV.l_array_survey * (PrepDV.l_array_survey + 1) * (sig_beam ** 2) / 2.)) ** (
                            2 * addbth) + 1e-200
                pw_ip = interpolate.interp1d(np.log(np.arange(3 * nside_pw[jb])), np.log(hp.pixwin(nside_pw[jb])),
                                             fill_value='extrapolate')
                Bl *= (np.exp(pw_ip(np.log(PrepDV.l_array_survey)))) ** (2 * addpw) + 1e-200
                if PrepDV_params['PrepDV_fid'].has_yy_noise:
                    Cl_yy_tot_plus_noise = (Cltot_j1j2 * Bl)[PrepDV.ind_select_survey] + Cl_noise_ellsurvey
                else:
                    Cl_yy_tot_plus_noise = PrepDV_params['Cl_tot_yy_l_array']

                Cl_yy_dict['bin_0_0'] = {'1h': Cl1h_j1j2 * Bl, '2h': Cl2h_j1j2 * Bl, 'tot': Cltot_j1j2 * Bl,
                                         'tot_ellsurvey': (Cltot_j1j2 * Bl)[PrepDV.ind_select_survey],
                                         'tot_plus_noise_ellsurvey': Cl_yy_tot_plus_noise}
                Cl_yy_dict['bin_combs'] = [[0, 0]]
                self.Cl_result_dict['yy'] = Cl_yy_dict
                if analysis_coords == 'real':
                    xi_yy_dict = {}
                    xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                              Cltot_j1j2 * Bl,
                                                                              theta_array_arcmin=theta_array_arcmin)
                    xi_yy_dict['bin_0_0'] = xitot_j1j2
                    xi_yy_dict['theta'] = theta_array
                    xi_yy_dict['bin_combs'] = [[0, 0]]
                    self.xi_result_dict['yy'] = xi_yy_dict

                    if 'yy' in PrepDV.stats_analyze:
                        block[
                            sec_save_name, 'theory_corrf_' + 'yy' + str(jb + 1) + '_' + 'bin_' + str(0) + '_' + str(
                                0)] = xitot_j1j2
                        block[sec_save_name, 'xcoord_' + 'yy' + str(jb + 1) + '_' + 'bin_' + str(0) + '_' + str(
                            0)] = theta_array
                else:
                    if 'yy' in PrepDV.stats_analyze:
                        block[
                            sec_save_name, 'theory_corrf_' + 'yy' + str(jb + 1) + '_' + 'bin_' + str(0) + '_' + str(
                                0)] = Cltot_j1j2
                        block[sec_save_name, 'xcoord_' + 'yy' + str(jb + 1) + '_' + 'bin_' + str(0) + '_' + str(
                            0)] = PrepDV.l_array

            if self.verbose:
                print('done y-y calculation')

        if ('gg' in PrepDV.stats_analyze) or (run_cov_pipe and ('gg' in PrepDV.lss_probes_allcomb)):
            try:
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
                                                                     PrepDV_params['bgl_z_dict' + str(j2)],
                                                                     model_2h='nl')
                                # Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('g', 'g', PrepDV.l_array,
                                # PrepDV_params['bgl_z_dict' + str(j1)],
                                # PrepDV_params['bgl_z_dict' + str(j2)])
                                # Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('g', 'g', Cl1h_j1j2, Cl2h_j1j2)
                                Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot_modelNL('g', 'g', PrepDV.l_array,
                                                                               PrepDV_params['ugl_zM_dict' + str(j1)],
                                                                               PrepDV_params['ugl_zM_dict' + str(j2)],
                                                                               PrepDV_params['bgl_z_dict' + str(j1)],
                                                                               PrepDV_params['bgl_z_dict' + str(j2)])

                                if j1 == j2:
                                    Cl_noise_ellsurvey = PrepDV_params['Cl_noise_gg_l_array' + str(j1)]
                                else:
                                    Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                                bin_combs.append([j1, j2])
                                # Cl_gg_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2,'2h_nl': Cl2h_j1j2_nl,
                                # 'tot': Cltot_j1j2,'tot2': Cltot_j1j2_m2,
                                # 'tot_ellsurvey': Cltot_j1j2[
                                # PrepDV.ind_select_survey],
                                # 'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                # PrepDV.ind_select_survey] + Cl_noise_ellsurvey}

                                Cl_gg_dict['bin_' + str(j1) + '_' + str(j2)] = {
                                    'tot': Cltot_j1j2,
                                    'tot_ellsurvey': Cltot_j1j2[
                                        PrepDV.ind_select_survey],
                                    'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                    PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                                if analysis_coords == 'real':
                                    # xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                    # Cltot_j1j2,
                                    # theta_array_arcmin=theta_array_arcmin)
                                    if self.CalcDV.PS_prepDV.use_only_halos:
                                        xi1h_j1j2 = np.zeros_like(theta_array)
                                    else:
                                        xi1h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                                 Cl1h_j1j2,
                                                                                                 theta_array_arcmin=theta_array_arcmin)
                                    xi2h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                             Cl2h_j1j2,
                                                                                             theta_array_arcmin=theta_array_arcmin)
                                    # xi2h_j1j2_nl, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                    # Cl2h_j1j2_nl,
                                    # theta_array_arcmin=theta_array_arcmin)
                                    # xitot2_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                    # Cltot_j1j2_m2,
                                    # theta_array_arcmin=theta_array_arcmin)
                                    #                                xi_gg_dict['bin_' + str(j1) + '_' + str(j2)] = xitot_j1j2
                                    # xi_gg_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h':xi1h_j1j2,'2h':xi2h_j1j2,'2h_nl':xi2h_j1j2_nl,'tot':xitot_j1j2,
                                    # 'tot2':xitot2_j1j2}

                                    ind_gt = np.where(xi1h_j1j2 > xi2h_j1j2)[0]
                                    if len(ind_gt) > 0:
                                        xitot_j1j2 = np.hstack((xi1h_j1j2[:ind_gt[-1] + 1], xi2h_j1j2[ind_gt[-1] + 1:]))
                                    else:
                                        xitot_j1j2 = xi2h_j1j2
                                    # xi_gg_dict['bin_' + str(j1) + '_' + str(j2)] = {'tot':xitot_j1j2}

                                    xi_gg_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h': xi1h_j1j2, '2h': xi2h_j1j2,
                                                                                    'tot': xitot_j1j2}
                                    if 'theta' not in xi_gg_dict.keys():
                                        xi_gg_dict['theta'] = theta_array

                                    if 'gg' in PrepDV.stats_analyze:
                                        block[
                                            sec_save_name, 'theory_corrf_' + 'gg' + '_' + 'bin_' + str(j1) + '_' + str(
                                                j2)] = xitot_j1j2

                                        block[sec_save_name, 'xcoord_' + 'gg' + '_' + 'bin_' + str(j1) + '_' + str(
                                            j2)] = theta_array
                                else:
                                    if 'gg' in PrepDV.stats_analyze:
                                        block[
                                            sec_save_name, 'theory_corrf_' + 'gg' + '_' + 'bin_' + str(j1) + '_' + str(
                                                j2)] = Cltot_j1j2

                                        block[sec_save_name, 'xcoord_' + 'gg' + '_' + 'bin_' + str(j1) + '_' + str(
                                            j2)] = PrepDV.l_array

                Cl_gg_dict['bin_combs'] = bin_combs
                self.Cl_result_dict['gg'] = Cl_gg_dict
                if analysis_coords == 'real':
                    xi_gg_dict['bin_combs'] = bin_combs
                    xi_gg_dict['theta'] = theta_array
                    self.xi_result_dict['gg'] = xi_gg_dict

                if self.verbose:
                    print('done galaxy-galaxy calculation')
            except:
                print(traceback.format_exc())

        if ('gy' in PrepDV.stats_analyze) or (run_cov_pipe and ('gy' in PrepDV.lss_probes_allcomb)):
            try:
                Cl_gy_dict = {}
                if analysis_coords == 'real':
                    xi_gy_dict = {}
                bin_combs = []
                for j1 in bins_lens:
                    if save_detailed_DV:
                        Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('g', 'y', PrepDV.l_array,
                                                             PrepDV_params['ugl_cross_zM_dict' + str(j1)],
                                                             PrepDV_params['uyl_zM_dict0'])
                        Cl2h_j1j2_nl = self.CalcDV.get_Cl_AB_2h('g', 'y', PrepDV.l_array,
                                                                PrepDV_params['bgl_z_dict' + str(j1)],
                                                                PrepDV_params['byl_z_dict0'], model_2h='nl')
                        Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('g', 'y', PrepDV.l_array,
                                                             PrepDV_params['bgl_z_dict' + str(j1)],
                                                             PrepDV_params['byl_z_dict0'])
                        Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('g', 'y', Cl1h_j1j2, Cl2h_j1j2)
                    else:
                        Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot_modelmead('g', 'y', PrepDV.l_array,
                                                                         PrepDV_params['ugl_cross_zM_dict' + str(j1)],
                                                                         PrepDV_params['uyl_zM_dict' + str(0)],
                                                                         PrepDV_params['bgl_z_dict' + str(j1)],
                                                                         PrepDV_params['byl_z_dict' + str(0)],
                                                                                al = 1.)
                    if self.CalcDV.PS_prepDV.use_only_halos:
                        if self.CalcDV.PS_prepDV.fmis > 0:
                            # import ipdb; ipdb.set_trace()
                            Cltot_j1j2_orig = Cltot_j1j2
                            Cltot_j1j2 = self.CalcDV.get_Cl_yg_miscentered(PrepDV.l_array, Cltot_j1j2)
                    Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                    bin_combs.append([j1, 0])
                    # Cl_gy_dict['bin_' + str(j1) + '_0'] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2,'2h_nl': Cl2h_j1j2_nl, 'tot': Cltot_j1j2,'tot2': Cltot_j1j2_m2,
                    # 'tot_ellsurvey': Cltot_j1j2[PrepDV.ind_select_survey],
                    # 'tot_plus_noise_ellsurvey': Cltot_j1j2[
                    # PrepDV.ind_select_survey] + Cl_noise_ellsurvey}

                    # import ipdb; ipdb.set_trace() # BREAKPOINT

                    for jb in range(len(beam_fwhm_arcmin)):
                        sig_beam = beam_fwhm_arcmin[jb] * (1. / 60.) * (np.pi / 180.) * (1. / np.sqrt(8. * np.log(2)))
                        Bl = (np.exp(
                            -1. * PrepDV.l_array_survey * (PrepDV.l_array_survey + 1) * (sig_beam ** 2) / 2.)) ** (
                                 addbth) + 1e-200
                        pw_ip = interpolate.interp1d(np.log(np.arange(3 * nside_pw[jb])+1),
                                                     np.log(hp.pixwin(nside_pw[jb])), fill_value='extrapolate')
                        
                        Bl *= (np.exp(pw_ip(np.log(PrepDV.l_array_survey)))) ** (addpw) + 1e-200
                        if save_detailed_DV:
                            Cl_gy_dict['bin_' + str(j1) + '_0'] = {'1h': Cl1h_j1j2 * Bl, '2h': Cl2h_j1j2 * Bl,
                                                                   '2h_nl': Cl2h_j1j2_nl * Bl, 'tot': Cltot_j1j2 * Bl,
                                                                   'tot_ellsurvey': (Cltot_j1j2 * Bl)[
                                                                       PrepDV.ind_select_survey],
                                                                   'tot_plus_noise_ellsurvey': (Cltot_j1j2 * Bl)[
                                                                                                   PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                        else:
                            Cl_gy_dict['bin_' + str(j1) + '_0'] = {'tot': Cltot_j1j2 * Bl,
                                                                   'tot_ellsurvey': (Cltot_j1j2 * Bl)[
                                                                       PrepDV.ind_select_survey],
                                                                   'tot_plus_noise_ellsurvey': (Cltot_j1j2 * Bl)[
                                                                                                   PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                        if analysis_coords == 'real':
                            # import ipdb; ipdb.set_trace()
                            xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                      Cltot_j1j2 * Bl,
                                                                                      theta_array_arcmin=theta_array_arcmin)
                            # import ipdb; ipdb.set_trace()
                            if save_detailed_DV:
                                # import pdb; pdb.set_trace()
                                xi1h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                         Cl1h_j1j2,
                                                                                         theta_array_arcmin=theta_array_arcmin)
                                xi2h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                         Cl2h_j1j2,
                                                                                         theta_array_arcmin=theta_array_arcmin)
                                xi2h_j1j2_nl, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                            Cl2h_j1j2_nl,
                                                                                            theta_array_arcmin=theta_array_arcmin)
                                xi_gy_dict['bin_' + str(j1) + '_' + str(0)] = {'1h': xi1h_j1j2, '2h': xi2h_j1j2,
                                                                               '2h_nl': xi2h_j1j2_nl, 'tot': xitot_j1j2}
                            else:
                                xi_gy_dict['bin_' + str(j1) + '_' + str(0)] = {'tot': xitot_j1j2}
                            if 'theta' not in xi_gy_dict.keys():
                                xi_gy_dict['theta'] = theta_array

                            if 'gy' in PrepDV.stats_analyze:
                                block[sec_save_name, 'theory_corrf_' + 'gy' + str(jb + 1) + '_' + 'bin_' + str(
                                    j1) + '_' + str(
                                    0)] = xitot_j1j2

                                block[
                                    sec_save_name, 'xcoord_' + 'gy' + str(jb + 1) + '_' + 'bin_' + str(j1) + '_' + str(
                                        0)] = theta_array
                        else:
                            if 'gy' in PrepDV.stats_analyze:
                                block[sec_save_name, 'theory_corrf_' + 'gy' + str(jb + 1) + '_' + 'bin_' + str(
                                    j1) + '_' + str(
                                    0)] = Cltot_j1j2

                                block[
                                    sec_save_name, 'xcoord_' + 'gy' + str(jb + 1) + '_' + 'bin_' + str(j1) + '_' + str(
                                        0)] = PrepDV.l_array

                    Cl_gy_dict['bin_combs'] = bin_combs
                    self.Cl_result_dict['gy'] = Cl_gy_dict
                    if analysis_coords == 'real':
                        xi_gy_dict['bin_combs'] = bin_combs
                        xi_gy_dict['theta'] = theta_array
                        self.xi_result_dict['gy'] = xi_gy_dict

                if self.verbose:
                    print('done galaxy-y calculation')

            except:
                print(traceback.format_exc())

        if ('gk' in PrepDV.stats_analyze) or (run_cov_pipe and ('gk' in PrepDV.lss_probes_allcomb)):
            Cl_gk_dict = {}
            if analysis_coords == 'real':
                xi_gk_dict = {}
                xi_gtg_dict = {}
            bin_combs = []
            for j1 in bins_lens:
                for j2 in bins_source:
                    Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('g', 'k', PrepDV.l_array,
                                                         PrepDV_params['ugl_cross_zM_dict' + str(j1)],
                                                         PrepDV_params['ukl_zM_dict' + str(j2)])
                    Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('g', 'k', PrepDV.l_array,
                                                         PrepDV_params['bgl_z_dict' + str(j1)],
                                                         PrepDV_params['bkl_z_dict' + str(j2)],
                                                         model_2h='nl')
                    Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('g', 'k', Cl1h_j1j2, Cl2h_j1j2)
                    Cltot_j1j2_m2 = self.CalcDV.get_Cl_AB_tot_modelNL('g', 'k', PrepDV.l_array,
                                                                      PrepDV_params['ugl_cross_zM_dict' + str(j1)],
                                                                      PrepDV_params['ukl_zM_dict' + str(j2)],
                                                                      PrepDV_params['bgl_z_dict' + str(j1)],
                                                                      PrepDV_params['bkl_z_dict' + str(j2)])

                    Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                    bin_combs.append([j1, j2])
                    Cl_gk_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2, 'tot': Cltot_j1j2,
                                                                    'tot2': Cltot_j1j2_m2,
                                                                    'tot_ellsurvey': Cltot_j1j2[
                                                                        PrepDV.ind_select_survey],
                                                                    'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                                                    PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                    if analysis_coords == 'real':
                        xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                  Cltot_j1j2,
                                                                                  theta_array_arcmin=theta_array_arcmin)
                        gt_1h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                  Cl1h_j1j2,
                                                                                  theta_array_arcmin=theta_array_arcmin)
                        gt_2h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                  Cl2h_j1j2,
                                                                                  theta_array_arcmin=theta_array_arcmin)
                        gt_tot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                   Cltot_j1j2,
                                                                                   theta_array_arcmin=theta_array_arcmin)
                        gt_tot_j1j2_m2, theta_array = self.CalcDV.do_Hankel_transform(2, PrepDV.l_array,
                                                                                      Cltot_j1j2_m2,
                                                                                      theta_array_arcmin=theta_array_arcmin)
                        xi_gk_dict['bin_' + str(j1) + '_' + str(j2)] = xitot_j1j2
                        xi_gtg_dict['bin_' + str(j1) + '_' + str(j2)] = {'tot': gt_tot_j1j2, 'tot2': gt_tot_j1j2_m2,
                                                                         '1h': gt_1h_j1j2, '2h': gt_2h_j1j2}
                        if 'theta' not in xi_gk_dict.keys():
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
                xi_gk_dict['theta'] = theta_array
                xi_gtg_dict['theta'] = theta_array
                self.xi_result_dict['gk'] = xi_gk_dict
                self.xi_result_dict['gtg'] = xi_gtg_dict

            if self.verbose:
                print('done galaxy-shear calculation')

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
                isgtykk, isgtygty, iskkkk, isgygy = False, False, False, False
                if analysis_coords == 'real':
                    fftcovtot_stat12 = {}
                    fftmcovtot_stat12 = {}
                    fftpmcovtot_stat12 = {}
                    if (stats_analyze_1_ordered == 'ky') and (stats_analyze_2_ordered == 'ky'):
                        gtfftcovtot_stat12 = {}
                        isgtygty = True
                    if (stats_analyze_1_ordered == 'gy') and (stats_analyze_2_ordered == 'gy'):
                        isgygy = True
                    if (stats_analyze_1_ordered == 'kk') and (stats_analyze_2_ordered == 'kk'):
                        iskkkk = True
                    if ((stats_analyze_1_ordered == 'kk') and (stats_analyze_2_ordered == 'ky')) or (
                            (stats_analyze_1_ordered == 'ky') and (stats_analyze_2_ordered == 'kk')):
                        kkgtfftcovtot_stat12 = {}
                        kkmgtfftcovtot_stat12 = {}
                        isgtykk = True
                bins_comb = []
                for jb1 in range(len(bins1_stat1)):
                    for jb2 in range(len(bins1_stat2)):
                        if self.verbose:
                            print(stats_analyze_1_ordered, stats_analyze_2_ordered, bins1_stat1[jb1], bins2_stat1[jb1],
                                  bins1_stat2[jb2], bins2_stat2[jb2])

                        # covG, cov_cl, cov_clnl, cov_nl = self.CalcDV.get_cov_G(bins1_stat1[jb1], bins2_stat1[jb1], bins1_stat2[jb2],
                        # bins2_stat2[jb2], stats_analyze_1_ordered, stats_analyze_2_ordered,
                        # self.Cl_result_dict, fsky_dict)

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
                                                       uAl_zM_dict, uBl_zM_dict, uCl_zM_dict, uDl_zM_dict,
                                                       beam_fwhm_arcmin[0])

                        covtot = covG + covNG
                        # covtot = covG 
                        bin_key = 'bin_' + str(bins1_stat1[jb1]) + '_' + str(bins2_stat1[jb1]) + '_' + str(
                            bins1_stat2[jb2]) + '_' + str(bins2_stat2[jb2])
                        covG_stat12[bin_key] = covG
                        covNG_stat12[bin_key] = covNG
                        covtot_stat12[bin_key] = covtot
                        bins_comb.append([bins1_stat1[jb1], bins2_stat1[jb1], bins1_stat2[jb2], bins2_stat2[jb2]])
                        if analysis_coords == 'real':
                            if isodd:
                                covtot_rs = covtot[:-1, :][:, :-1]
                                # covcl_rs = cov_cl[:-1, :][:, :-1]
                                # covclnl_rs = cov_clnl[:-1, :][:, :-1]
                                # covnl_rs = cov_nl[:-1, :][:, :-1]
                            else:
                                covtot_rs = covtot
                                # covcl_rs = cov_cl
                                # covclnl_rs = cov_clnl
                                # covnl_rs = cov_nl
                            newtwobessel = two_Bessel(ell, ell, covtot_rs * (ell1_ell2 ** 2) * (1. / (4 * np.pi ** 2)),
                                                      nu1=1.01, nu2=1.01, N_extrap_low=0, N_extrap_high=0,
                                                      c_window_width=0.25,
                                                      N_pad=1000)
                            t1, t2, cov_fft = newtwobessel.two_Bessel_binave(0, 0, dlnk, dlnk)
                            theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                            cov_tot_fft = cov_fft[:, :-1][:-1, :]
                            fftcovtot_stat12[bin_key] = cov_tot_fft

                            # newtwobessel = two_Bessel(ell, ell, covcl_rs * (ell1_ell2 ** 2) * (1. / (4 * np.pi ** 2)),
                            # nu1=1.01, nu2=1.01, N_extrap_low=0, N_extrap_high=0,
                            # c_window_width=0.25,
                            # N_pad=1000)
                            # t1, t2, cov_fft = newtwobessel.two_Bessel_binave(0, 0, dlnk, dlnk)
                            # theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                            # cov_tot_fft = cov_fft[:,:-1][:-1,:]
                            # fftcovtot_stat12[bin_key + '_cl'] = cov_tot_fft

                            # newtwobessel = two_Bessel(ell, ell, covclnl_rs * (ell1_ell2 ** 2) * (1. / (4 * np.pi ** 2)),
                            # nu1=1.01, nu2=1.01, N_extrap_low=0, N_extrap_high=0,
                            # c_window_width=0.25,
                            # N_pad=1000)
                            # t1, t2, cov_fft = newtwobessel.two_Bessel_binave(0, 0, dlnk, dlnk)
                            # theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                            # cov_tot_fft = cov_fft[:,:-1][:-1,:]
                            # fftcovtot_stat12[bin_key + '_clnl'] = cov_tot_fft

                            if iskkkk:
                                t1, t2, cov_fft = newtwobessel.two_Bessel_binave(4, 4, dlnk, dlnk)
                                theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                                cov_tot_fftm = cov_fft[:, :-1][:-1, :]
                                fftmcovtot_stat12[bin_key] = cov_tot_fftm

                                t1, t2, cov_fft = newtwobessel.two_Bessel_binave(4, 0, dlnk, dlnk)
                                theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                                cov_tot_fftm = cov_fft[:, :-1][:-1, :]
                                fftpmcovtot_stat12[bin_key] = cov_tot_fftm

                            if isgtygty:
                                t1, t2, covgt_fft = newtwobessel.two_Bessel_binave(2, 2, dlnk, dlnk)
                                gtfftcovtot_stat12[bin_key] = covgt_fft[:, :-1][:-1, :]
                                theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                                if 'theta' not in gtfftcovtot_stat12.keys():
                                    gtfftcovtot_stat12['theta'] = theta_vals_arcmin_fft

                            if isgtykk:
                                t1, t2, covgt_fft = newtwobessel.two_Bessel_binave(2, 0, dlnk, dlnk)
                                kkgtfftcovtot_stat12[bin_key] = covgt_fft[:, :-1][:-1, :]
                                theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                                if 'theta' not in kkgtfftcovtot_stat12.keys():
                                    kkgtfftcovtot_stat12['theta'] = theta_vals_arcmin_fft

                                t1, t2, covgt_fft = newtwobessel.two_Bessel_binave(2, 4, dlnk, dlnk)
                                kkmgtfftcovtot_stat12[bin_key] = covgt_fft[:, :-1][:-1, :]
                                theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                                if 'theta' not in kkmgtfftcovtot_stat12.keys():
                                    kkmgtfftcovtot_stat12['theta'] = theta_vals_arcmin_fft

                            if 'theta' not in fftcovtot_stat12.keys():
                                fftcovtot_stat12['theta'] = theta_vals_arcmin_fft
                            if 'theta' not in fftmcovtot_stat12.keys():
                                fftmcovtot_stat12['theta'] = theta_vals_arcmin_fft
                            if 'theta' not in fftpmcovtot_stat12.keys():
                                fftpmcovtot_stat12['theta'] = theta_vals_arcmin_fft

                covG_stat12['bins_comb'] = bins_comb
                covNG_stat12['bins_comb'] = bins_comb
                covtot_stat12['bins_comb'] = bins_comb
                if analysis_coords == 'real':
                    fftcovtot_stat12['bins_comb'] = bins_comb
                    fftmcovtot_stat12['bins_comb'] = bins_comb
                    fftpmcovtot_stat12['bins_comb'] = bins_comb
                    if isgtygty:
                        gtfftcovtot_stat12['bins_comb'] = bins_comb
                        stat_analyze_key = 'gty_gty'
                        self.fftcovtot_dict[stat_analyze_key] = gtfftcovtot_stat12

                    if isgtykk:
                        kkgtfftcovtot_stat12['bins_comb'] = bins_comb
                        if ((stats_analyze_1_ordered == 'kk') and (stats_analyze_2_ordered == 'ky')):
                            stat_analyze_key1 = 'kk_gty'
                            stat_analyze_key2 = 'kkm_gty'
                        else:
                            stat_analyze_key1 = 'gty_kk'
                            stat_analyze_key2 = 'gty_kkm'

                        self.fftcovtot_dict[stat_analyze_key1] = kkgtfftcovtot_stat12
                        kkmgtfftcovtot_stat12['bins_comb'] = bins_comb
                        self.fftcovtot_dict[stat_analyze_key2] = kkmgtfftcovtot_stat12

                    # self.fftcovG_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covG_stat12
                    # self.fftcovNG_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covNG_stat12
                    self.fftcovtot_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = fftcovtot_stat12
                    if iskkkk:
                        self.fftcovtot_dict['kkm_kkm'] = fftmcovtot_stat12
                        self.fftcovtot_dict['kk_kkm'] = fftpmcovtot_stat12

                self.covG_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covG_stat12
                self.covNG_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covNG_stat12
                self.covtot_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covtot_stat12
