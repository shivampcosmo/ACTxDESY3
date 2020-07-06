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

from PrepDataVec import *
pi = np.pi







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
                xi_kk_dict['theta'] = theta_array
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
                                Cl2h_j1j2_nl = self.CalcDV.get_Cl_AB_2h_nl('g', 'g', PrepDV.l_array,
                                                                    PrepDV_params['bgl_z_dict' + str(j1)],
                                                                    PrepDV_params['bgl_z_dict' + str(j2)])
                                Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('g', 'g', PrepDV.l_array,
                                                                    PrepDV_params['bgl_z_dict' + str(j1)],
                                                                    PrepDV_params['bgl_z_dict' + str(j2)])
                                Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('g', 'g', Cl1h_j1j2, Cl2h_j1j2)
                                Cltot_j1j2_m2 = self.CalcDV.get_Cl_AB_tot_model2('g', 'g', PrepDV.l_array,
                                                                    PrepDV_params['ugl_zM_dict' + str(j1)],
                                                                    PrepDV_params['ugl_zM_dict' + str(j2)],
                                                                    PrepDV_params['bgl_z_dict' + str(j1)],
                                                                    PrepDV_params['bgl_z_dict' + str(j2)])

                                if j1 == j2:
                                    Cl_noise_ellsurvey = PrepDV_params['Cl_noise_gg_l_array' + str(j1)]
                                else:
                                    Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                                bin_combs.append([j1, j2])
                                Cl_gg_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2,'2h_nl': Cl2h_j1j2_nl,
                                                                                'tot': Cltot_j1j2,'tot2': Cltot_j1j2_m2,
                                                                                'tot_ellsurvey': Cltot_j1j2[
                                                                                    PrepDV.ind_select_survey],
                                                                                'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                                                                PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                                if analysis_coords == 'real':
                                    xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                            Cltot_j1j2,
                                                                                            theta_array_arcmin=theta_array_arcmin)
                                    if self.CalcDV.PS_prepDV.use_only_halos:
                                        xi1h_j1j2 = np.zeros_like(theta_array)
                                    else:
                                        xi1h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                                Cl1h_j1j2,
                                                                                                theta_array_arcmin=theta_array_arcmin)
                                    xi2h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                            Cl2h_j1j2,
                                                                                            theta_array_arcmin=theta_array_arcmin)
                                    xi2h_j1j2_nl, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                            Cl2h_j1j2_nl,
                                                                                            theta_array_arcmin=theta_array_arcmin)
                                    xitot2_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                            Cltot_j1j2_m2,
                                                                                            theta_array_arcmin=theta_array_arcmin)
#                                xi_gg_dict['bin_' + str(j1) + '_' + str(j2)] = xitot_j1j2
                                    xi_gg_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h':xi1h_j1j2,'2h':xi2h_j1j2,'2h_nl':xi2h_j1j2_nl,'tot':xitot_j1j2,
                                            'tot2':xitot2_j1j2}
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
                    Cl1h_j1j2 = self.CalcDV.get_Cl_AB_1h('g', 'y', PrepDV.l_array,
                                                        PrepDV_params['ugl_cross_zM_dict' + str(j1)],
                                                        PrepDV_params['uyl_zM_dict0'])
                    Cl2h_j1j2_nl = self.CalcDV.get_Cl_AB_2h_nl('g', 'y', PrepDV.l_array, PrepDV_params['bgl_z_dict' + str(j1)],
                                                        PrepDV_params['byl_z_dict0'])
                    Cl2h_j1j2 = self.CalcDV.get_Cl_AB_2h('g', 'y', PrepDV.l_array, PrepDV_params['bgl_z_dict' + str(j1)],
                                                        PrepDV_params['byl_z_dict0'])
                    Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('g', 'y', Cl1h_j1j2, Cl2h_j1j2)
                    Cltot_j1j2_m2 = self.CalcDV.get_Cl_AB_tot_model2('g', 'y', PrepDV.l_array,
                                                        PrepDV_params['ugl_cross_zM_dict' + str(j1)],
                                                        PrepDV_params['ugl_cross_zM_dict' + str(j2)],
                                                        PrepDV_params['bgl_z_dict' + str(j1)],
                                                        PrepDV_params['bgl_z_dict' + str(j2)])

                    Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                    bin_combs.append([j1, 0])
                    Cl_gy_dict['bin_' + str(j1) + '_0'] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2,'2h_nl': Cl2h_j1j2_nl, 'tot': Cltot_j1j2,'tot2': Cltot_j1j2_m2,
                                                        'tot_ellsurvey': Cltot_j1j2[PrepDV.ind_select_survey],
                                                        'tot_plus_noise_ellsurvey': Cltot_j1j2[
                                                                                        PrepDV.ind_select_survey] + Cl_noise_ellsurvey}
                    if analysis_coords == 'real':
                        xitot_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                Cltot_j1j2,
                                                                                theta_array_arcmin=theta_array_arcmin)
#                    xi_gy_dict['bin_' + str(j1) + '_0'] = xitot_j1j2

                        xi1h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                    Cl1h_j1j2,
                                                                                    theta_array_arcmin=theta_array_arcmin)
                        xi2h_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                    Cl2h_j1j2,
                                                                                    theta_array_arcmin=theta_array_arcmin)
                        xi2h_j1j2_nl, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                    Cl2h_j1j2_nl,
                                                                                    theta_array_arcmin=theta_array_arcmin)
                        xitot2_j1j2, theta_array = self.CalcDV.do_Hankel_transform(0, PrepDV.l_array,
                                                                                    Cltot_j1j2_m2,
                                                                                    theta_array_arcmin=theta_array_arcmin)
#                                xi_gg_dict['bin_' + str(j1) + '_' + str(j2)] = xitot_j1j2
                        xi_gy_dict['bin_' + str(j1) + '_' + str(0)] = {'1h':xi1h_j1j2,'2h':xi2h_j1j2,'2h_nl':xi2h_j1j2_nl,'tot':xitot_j1j2,'tot2':xitot2_j1j2}
                        if 'theta' not in xi_gy_dict.keys():
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
                                                         PrepDV_params['bkl_z_dict' + str(j2)])
                    Cltot_j1j2 = self.CalcDV.get_Cl_AB_tot('g', 'k', Cl1h_j1j2, Cl2h_j1j2)
                    Cltot_j1j2_m2 = self.CalcDV.get_Cl_AB_tot_model2('g', 'k', PrepDV.l_array,
                                                         PrepDV_params['ugl_cross_zM_dict' + str(j1)],
                                                         PrepDV_params['ukl_zM_dict' + str(j2)],
                                                         PrepDV_params['bgl_z_dict' + str(j1)],
                                                         PrepDV_params['bkl_z_dict' + str(j2)])

                    Cl_noise_ellsurvey = np.zeros_like(PrepDV.l_array_survey)
                    bin_combs.append([j1, j2])
                    Cl_gk_dict['bin_' + str(j1) + '_' + str(j2)] = {'1h': Cl1h_j1j2, '2h': Cl2h_j1j2, 'tot': Cltot_j1j2,'tot2': Cltot_j1j2_m2,
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
                        xi_gtg_dict['bin_' + str(j1) + '_' + str(j2)] = {'tot':gt_tot_j1j2,'tot2':gt_tot_j1j2_m2,'1h':gt_1h_j1j2,'2h':gt_2h_j1j2}
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
                    # import ipdb; ipdb.set_trace()


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
















