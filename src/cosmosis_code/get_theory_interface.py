import sys, os
from cosmosis.datablock import names, option_section, BlockError
sys.path.insert(0, os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/helper/')
import numpy as np
import copy
import pdb
import ast
import scipy as sp
from scipy import interpolate
from HOD import *
from pressure import *
from general_hm import *
from Powerspec import *
from PrepDataVec import *
from DataVec import *
from scipy.interpolate import RegularGridInterpolator
import multiprocessing
import dill
from configobj import ConfigObj
from configparser import ConfigParser
import pickle as pk
from mcfit import P2xi
from mcfit import xi2P
import pdb
import time
import traceback as tb

def get_value(section, value, config_run, config_def):
    if section in config_run.keys() and value in config_run[section].keys():
        val = config_run[section][value]
    else:
        val = config_def[section][value]
    return val


def QR_inverse(matrix):
    _Q, _R = np.linalg.qr(matrix)
    return np.dot(_Q, np.linalg.inv(_R.T))


def read_ini(ini_file, ini_def=None, twopt_file=None, get_bp=False, use_Plin_block=False, use_dndm_block=False, use_conc_block=False, use_bm_block=False,
        theta_min=2.5, theta_max=250.,ntheta=0):
    config_run = ConfigObj(ini_file, unrepr=True)
    if ini_def is None:
        config_def = ConfigObj(config_run['DEFAULT']['params_default_file'], unrepr=True)
    else:
        config_def = ConfigObj(ini_def, unrepr=True)

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

    # import pdb; pdb.set_trace()
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

    other_params_dict['do_vary_cosmo'] = False

    other_params_dict['M_array'] = np.logspace(other_params_dict['logM_array_min'], other_params_dict['logM_array_max'],
                                               other_params_dict['num_M'])

    if other_params_dict['zarray_spacing'] == 'log':
        other_params_dict['z_array'] = np.logspace(np.log10(other_params_dict['z_array_min']),
                                                   np.log10(other_params_dict['z_array_max']),
                                                   other_params_dict['num_z'])

    if other_params_dict['zarray_spacing'] == 'lin':
        other_params_dict['z_array'] = np.linspace((other_params_dict['z_array_min']),
                                                   (other_params_dict['z_array_max']),
                                                   other_params_dict['num_z'])

    other_params_dict['x_array'] = np.logspace(np.log10(other_params_dict['xmin']), np.log10(other_params_dict['xmax']),
                                               other_params_dict['num_x'])

    if ntheta > 0:
        theta_temp = np.logspace(np.log10(theta_min),np.log10(theta_max), ntheta+1)
        theta_temp_rad = theta_temp*(1./60.)*(np.pi/180.)
        ell_temp = (1./theta_temp_rad)[::-1]
        logell_temp = np.log(ell_temp)
        dell = logell_temp[1] - logell_temp[0]
        log_lmax = np.log(other_params_dict['lmax'])
        log_lmin = np.log(other_params_dict['lmin'])
        logell_rightext = np.arange(logell_temp[-1],log_lmax,dell)
        logell_leftext = np.arange(logell_temp[0],log_lmin,-dell)[::-1]
        logell_cen = np.hstack((logell_leftext[:-1],logell_temp,logell_rightext))
        logell_all = (logell_cen[1:] + logell_cen[:-1])/2.
        logell_all = np.insert(logell_all,[0,len(logell_all)],[logell_all[0]-dell,logell_all[-1]+dell])
        ell_cen = np.exp(logell_cen)
        dell_all = np.exp(logell_all)[1:] - np.exp(logell_all)[:-1]
        other_params_dict['dl_array'] = dell_all
        other_params_dict['l_array'] = ell_cen

    else:
        if twopt_file is None:
            if other_params_dict['larray_spacing'] == 'log':
                if other_params_dict['num_l'] == 0:
                    l_array_all = np.exp(np.arange(np.log(other_params_dict['lmin']), np.log(other_params_dict['lmax']),
                                                other_params_dict['dl_log_array']))
                else:
                    l_array_all = np.logspace(np.log10(other_params_dict['lmin']), np.log10(other_params_dict['lmax']),
                                            other_params_dict['num_l'])

            if other_params_dict['larray_spacing'] == 'lin':
                l_array_all = np.linspace((other_params_dict['lmin']), (other_params_dict['lmax']),
                                        other_params_dict['num_l'])
            other_params_dict['dl_array'] = l_array_all[1:] - l_array_all[:-1]
            other_params_dict['l_array'] = (l_array_all[1:] + l_array_all[:-1]) / 2.
        else:
            clf = pk.load(open(twopt_file, 'rb'))
            ell_data = clf['ell']
            lmin = (ell_data - (ell_data[1] - ell_data[0]) / 2.)[0]
            lmax = (ell_data + (ell_data[1] - ell_data[0]) / 2.)[-1]
            l_array_all = np.linspace(lmin, lmax, len(ell_data) + 1)
            other_params_dict['dl_array'] = l_array_all[1:] - l_array_all[:-1]
            other_params_dict['l_array'] = (l_array_all[1:] + l_array_all[:-1]) / 2.

    other_params_dict['ng_zarray'] = np.linspace(0.1, 2.0, 100)
    other_params_dict['ng_value'] = np.ones(len(other_params_dict['ng_zarray'])) / sp.integrate.simps(
        np.ones(len(other_params_dict['ng_zarray'])), other_params_dict['ng_zarray'])

    M_array, z_array = other_params_dict['M_array'], other_params_dict['z_array']
    nm, nz = len(M_array), len(z_array)
    M_mat_mdef = np.tile(M_array.reshape(1, nm), (nz, 1))
    mdef_analysis = other_params_dict['mdef_analysis']
    ghmf = general_hm(cosmo_params_dict, pressure_params_dict, other_params_dict)
    if not use_conc_block:
        halo_conc_mdef = ghmf.get_halo_conc_Mz(M_mat_mdef, mdef_analysis)
        other_params_dict['halo_conc_mdef'] = halo_conc_mdef


    if not use_dndm_block:
        dndm_array, bm_array = ghmf.get_dndm_bias(M_mat_mdef, mdef_analysis)
        other_params_dict['dndm_array'] = dndm_array

    if not use_bm_block:
        dndm_array, bm_array = ghmf.get_dndm_bias(M_mat_mdef, mdef_analysis)
        other_params_dict['bm_array'] = bm_array

    if not use_Plin_block:
        pkzlin_interp = ghmf.get_Pklin_zk_interp()
        if get_bp:
            wplin_interp = ghmf.get_wplin_interp(2, pkzlin_interp)
            other_params_dict['wplin_interp'] = wplin_interp
        other_params_dict['pkzlin_interp'] = pkzlin_interp

    other_params_dict['get_bp'] = get_bp
    ini_info = {'cosmo_params_dict': cosmo_params_dict, 'pressure_params_dict': pressure_params_dict,
                'hod_params_dict': hod_params_dict, 'other_params_dict': other_params_dict}

    return ini_info


def setup(options):
    params_files_dir = options.get_string(option_section, "params_files_dir")
    params_file = options.get_string(option_section, "params_file")
    params_def_file = options.get_string(option_section, "params_def_file")
    twopt_file = (options.get_string(option_section, 'twopt_file','None'))
    if twopt_file == 'None':
        twopt_file = None
    do_use_measured_2pt = options.get_bool(option_section, "do_use_measured_2pt", default=False)
    get_bp = options.get_bool(option_section, "get_bp", default=False)
    bp_model = options.get_string(option_section, "bp_model", default='linear')

    use_Plin_block = options.get_bool(option_section, "use_Plin_block", default=False)
    use_dndm_block = options.get_bool(option_section, "use_dndm_block", default=False)
    use_conc_block = options.get_bool(option_section, "use_conc_block", default=False)
    use_bm_block = options.get_bool(option_section, "use_bm_block", default=False)

    ntheta = options.get_int(option_section, "ntheta", default=0)
    dlogtheta = options.get_string(option_section, "dlogtheta", default='')
    theta_min = options.get_double(option_section, "theta_min", default=1.0)
    theta_max = options.get_double(option_section, "theta_max", default=100.0)

    if do_use_measured_2pt:
        ini_info = read_ini(params_files_dir + params_file, ini_def=params_files_dir + params_def_file,
                            twopt_file=twopt_file, get_bp=get_bp, use_Plin_block=use_Plin_block,
                            use_dndm_block=use_dndm_block, use_conc_block=use_conc_block, use_bm_block=use_bm_block)
    else:
        ini_info = read_ini(params_files_dir + params_file, ini_def=params_files_dir + params_def_file, get_bp=get_bp,
                            use_Plin_block=use_Plin_block, use_dndm_block=use_dndm_block, use_conc_block=use_conc_block,
                            use_bm_block=use_bm_block, theta_min=theta_min, theta_max=theta_max, ntheta=0)

    z_edges = ast.literal_eval(
        options.get_string(option_section, "z_edges", default='[ 0.20, 0.40, 0.55, 0.70, 0.85, 0.95, 1.05 ]'))
    bins_source = ast.literal_eval(options.get_string(option_section, "bins_source", default='[1]'))
    bins_lens = ast.literal_eval(options.get_string(option_section, "bins_lens", default='[1]'))
    gg_doauto = options.get_bool(option_section, "gg_doauto", default=True)

    sec_name = options.get_string(option_section, "sec_name", default='theory_yx')
    sec_save_name = options.get_string(option_section, "sec_save_name", default='theory_yx')
    analysis_coords = options.get_string(option_section, "analysis_coords", default='real')
    save_data_fname = options.get_string(option_section, "save_data_fname", default='')
    save_real_space_cov = options.get_bool(option_section, "save_real_space_cov", default=False)

    verbose = options.get_bool(option_section, "verbose", default=False)
    run_cov_pipe = options.get_bool(option_section, "run_cov_pipe", default=False)
    save_detailed_DV = options.get_bool(option_section, "save_detailed_DV", default=False)
    save_DV = options.get_bool(option_section, "save_DV", default=False)

    returndict = {'bins_source': bins_source, 'bins_lens': bins_lens, 'z_edges': z_edges, 'twopt_file': twopt_file,
                  'sec_name': sec_name, 'sec_save_name': sec_save_name, 'save_data_fname': save_data_fname,
                  'save_real_space_cov': save_real_space_cov, 'run_cov_pipe':run_cov_pipe,
                  'do_use_measured_2pt': do_use_measured_2pt, 'get_bp': get_bp, 'bp_model':bp_model, 'dlogtheta': dlogtheta,
                  'ntheta': ntheta, 'theta_min': theta_min, 'theta_max': theta_max, 'analysis_coords': analysis_coords,
                  'verbose': verbose, 'gg_doauto':gg_doauto, 'use_Plin_block':use_Plin_block,
                  'use_dndm_block':use_dndm_block, 'use_conc_block':use_conc_block, 'save_detailed_DV':save_detailed_DV, 'save_DV':save_DV}

    return ini_info, returndict


def execute(block, config):
    ini_info, returndict = config

    bins_source, bins_lens, z_edges = returndict['bins_source'], returndict['bins_lens'], returndict['z_edges']
    gg_doauto = returndict['gg_doauto']
    twopt_file = returndict['twopt_file']
    sec_name, sec_save_name, save_data_fname, run_cov_pipe = returndict['sec_name'], returndict['sec_save_name'], \
                                                                returndict['save_data_fname'], returndict[
                                                                    'run_cov_pipe']
    save_real_space_cov, do_use_measured_2pt, get_bp, bp_model = returndict['save_real_space_cov'], returndict[
        'do_use_measured_2pt'], returndict['get_bp'], returndict['bp_model']
    dlogtheta, ntheta, theta_min, theta_max = returndict['dlogtheta'], returndict['ntheta'], returndict['theta_min'], \
                                              returndict['theta_max']
    analysis_coords, verbose = returndict['analysis_coords'], returndict['verbose']

    use_Plin_block, use_dndm_block, use_conc_block = returndict['use_Plin_block'], returndict['use_dndm_block'], returndict['use_conc_block']
    save_detailed_DV = returndict['save_detailed_DV']
    save_DV = returndict['save_DV']
    if twopt_file is not None:
        try:
            clf = pk.load(open(twopt_file, 'rb'))
        except:
            clf = pk.load(open(twopt_file, 'rb'), encoding='latin1')

    other_params_dict = copy.deepcopy(ini_info['other_params_dict'])
    cosmo_params_dict = copy.deepcopy(ini_info['cosmo_params_dict'])
    pressure_params_dict = copy.deepcopy(ini_info['pressure_params_dict'])
    hod_params_dict = copy.deepcopy(ini_info['hod_params_dict'])
    other_params_dict['z_edges'] = z_edges
    if use_Plin_block:
        lin_power = names.matter_power_lin
        nl_power = names.matter_power_nl
        z_block, k_block, pk_block = block.get_grid(lin_power, "z", "k_h", "p_k")
        pkzlin_interp = interpolate.RectBivariateSpline(z_block, np.log(k_block), np.log(pk_block))
        if get_bp:
            wplin_interp = ghmf.get_wplin_interp(2, pkzlin_interp)
            other_params_dict['wplin_interp'] = wplin_interp
        other_params_dict['pkzlin_interp'] = pkzlin_interp
        z_block, k_block, pk_block = block.get_grid(nl_power, "z", "k_h", "p_k")
        pkznl_interp = interpolate.RectBivariateSpline(z_block, np.log(k_block), np.log(pk_block))
        other_params_dict['pkznl_interp'] = pkznl_interp

    nl_power = names.matter_power_nl
    PrepDV_dict_allbins = {}
    if verbose:
        print('setting up the theta values')
    if ntheta == 0:
        if dlogtheta == 'uselarray':
            block[sec_save_name, 'theory_min'] = theta_min
            block[sec_save_name, 'theory_max'] = theta_max
            dlogtheta = np.log(other_params_dict['l_array'][1] / other_params_dict['l_array'][0])
            block[sec_save_name, 'dlogtheta'] = dlogtheta
            theta_array_all = np.exp(
                np.arange(np.log(theta_min) - dlogtheta/2., np.log(theta_max) + dlogtheta/2, block[sec_save_name, 'dlogtheta']))
            ntheta = len(theta_array_all)
            theta_array = (theta_array_all[1:] + theta_array_all[:-1]) / 2.
        else:
            theta_array = None
            theta_array_all = None
    else:
        block[sec_save_name, 'theory_min'] = theta_min
        block[sec_save_name, 'theory_max'] = theta_max
        block[sec_save_name, 'ntheta'] = ntheta
        theta_array_all = np.logspace(np.log10(theta_min), np.log10(theta_max), ntheta+1)
        theta_array = np.exp((np.log(theta_array_all)[1:] + np.log(theta_array_all)[:-1])/2.)
    other_params_dict['kk_hm_trans'] = 1
    for binvs in bins_source:
        for binvl in bins_lens:
            if verbose:
                print('starting the calculations for bin-lens:' + str(binvl) + ' and bin-source:' + str(binvs))
            cosmo_params_dict_bin = copy.deepcopy(cosmo_params_dict)
            pressure_params_dict_bin = copy.deepcopy(pressure_params_dict)
            hod_params_dict_bin = copy.deepcopy(hod_params_dict)
            other_params_dict_bin = copy.deepcopy(other_params_dict)
            for key in block.keys():
                if key[0] == sec_name:
                    param_val = key[1]
                    if '-' in param_val:
                        ln = list(param_val)
                        bin_n_sep = [n for n, x in enumerate(ln) if x == '-']
                        var_name = ''.join(ln[:bin_n_sep[-2]])
                        bin_n = int(ln[-1])

                        if bin_n == 0:
                            for pressure_keys in pressure_params_dict_bin.keys():
                                if var_name == pressure_keys.lower():
                                    pressure_params_dict_bin[pressure_keys] = block[key]
                             
                            if hod_params_dict['hod_type'] == 'DES_maglim_exp_zev':
                                for hod_keys in hod_params_dict_bin.keys():
                                    if var_name == hod_keys.lower():
                                        hod_params_dict_bin[hod_keys] = block[key]

                            if other_params_dict['put_IA']:
                                for other_keys in other_params_dict_bin.keys():
                                    if var_name == other_keys.lower():
                                        other_params_dict_bin[hod_keys] = block[key]

                        if bin_n == binvl:
                            for hod_keys in hod_params_dict_bin.keys():
                                if var_name == hod_keys.lower():
                                    hod_params_dict_bin[hod_keys] = block[key]

            if verbose:
                print('done putting in values file data in the dict')
            other_params_dict_bin['ng_zarray'] = block['nz_lens', 'z']

            binv_lens = binvl
            other_params_dict_bin['ng_value'] = block['nz_lens', 'bin_' + str(binv_lens)]
            other_params_dict_bin['ng_zarray_source'] = block['nz_source', 'z']
            if ('nz_source', 'bin_' + str(binvs)) in block.keys():
                other_params_dict_bin['ng_value_source'] = block['nz_source', 'bin_' + str(binvs)]
            else:
                other_params_dict_bin['ng_value_source'] = np.zeros_like(block['nz_source', 'z'])
            other_params_dict_bin['cosmo_fid'] = cosmo_params_dict_bin
            other_params_dict_bin['hod_fid'] = hod_params_dict_bin
            other_params_dict_bin['pressure_fid'] = pressure_params_dict_bin
            other_params_dict_bin['binvl'] = binvl
            other_params_dict_bin['binvs'] = binvs

            if do_use_measured_2pt:
                other_params_dict_bin['Clyy_measured'] = clf[('ymap', 'ymap')][('y', 'y')]['data'][0]
                other_params_dict_bin['Clgg_measured'] = \
                    clf[('galaxy_density', 'galaxy_density')][(binvl - 1, binvl - 1)]['data'][0]
                other_params_dict_bin['ell_measured'] = clf['ell']
                other_params_dict_bin['Nlgg_measured'] = \
                    clf[('galaxy_density', 'galaxy_density')][(binvl - 1, binvl - 1)]['random'][0][0]
            nbar_bin = block[sec_name, 'nbar--' + str(binvl)]
            other_params_dict_bin['nbar'] = nbar_bin
            if (sec_name, 'noisek--' + str(binvs)) in block.keys():
                other_params_dict_bin['noise_kappa'] = block[sec_name, 'noisek--' + str(binvs)]
            else:
                other_params_dict_bin['noise_kappa'] = 0.0

            if other_params_dict_bin['do_vary_cosmo']:
                del other_params_dict_bin['pkzlin_interp'], other_params_dict_bin['dndm_array'], other_params_dict_bin[
                    'bm_array'], other_params_dict_bin['halo_conc_mdef']
            if other_params_dict['put_IA'] and (not other_params_dict['only_2h_IA']) and ('gammaIA_allinterp' not in other_params_dict.keys()):
                if verbose:
                    print('getting IA interpolated object')
                save_data_fnameIA = other_params_dict['save_IA_fname']
                nk_temp = 100000
                if not os.path.isfile(other_params_dict['save_IA_fname']):
                    gammaIA_block = np.zeros((len(other_params_dict['z_array']), len(other_params_dict['M_array']), nk_temp))
                    a1h_IA = other_params_dict['a1h_IA']

                    H0 = 100. * (u.km / (u.s * u.Mpc))
                    G_new = const.G.to(u.Mpc ** 3 / ((u.s ** 2) * u.M_sun))
                    rho_crit = ((3 * (H0 ** 2) / (8 * np.pi * G_new)).to(u.M_sun / (u.Mpc ** 3))).value
                    for jz in range(len(other_params_dict['z_array'])):
                        for jM in range(len(other_params_dict['M_array'])):
                            cv = other_params_dict['halo_conc_mdef'][jz, jM]
                            gv, kv = hmf.compute_gamma_k_m(a1h_IA,other_params_dict['M_array'][jM],cv,rho_crit,Dv=200, nk=nk_temp)
                            gammaIA_block[jz, jM, :] = gv
                    ind_nz = np.where(gammaIA_block <= 0)
                    gammaIA_block[ind_nz] = 1e-200
                    to_save_dict = {'z':other_params_dict['z_array'],'M':other_params_dict['M_array'],'k':kv,'gIA':gammaIA_block}
                    with open(save_data_fnameIA,'wb') as f:
                        dill.dump(to_save_dict,f)
                else:
                    to_save_dict = dill.load(open(save_data_fnameIA,'rb'))

                gammaIA_allinterp = RegularGridInterpolator(
                    (to_save_dict['z'], np.log(to_save_dict['M']), np.log(to_save_dict['k'])),
                    np.log(to_save_dict['gIA']), fill_value=-200, bounds_error=False)
                other_params_dict['gammaIA_allinterp'] = gammaIA_allinterp
                other_params_dict_bin['gammaIA_allinterp'] = gammaIA_allinterp
                if verbose:
                    print('done getting IA interpolated object')

            other_params_dict['kk_hm_trans'] = 1.
            if ('uml_zM_dict' not in other_params_dict.keys()) and ('um_block_allinterp' not in other_params_dict.keys())\
                    and ((nl_power, 'um_1') in block.keys()):
                z_array_block = block[nl_power, 'z']
                hm_1h2h_alpha_mead = 2.93*(1.77**block[nl_power,'neff_out'])
                hm_interp = interpolate.interp1d(z_array_block, hm_1h2h_alpha_mead,fill_value='extrapolate')
                hm_1h2h_alpha = hm_interp(other_params_dict['z_array'])


                other_params_dict['kk_hm_trans'] = hm_1h2h_alpha
                other_params_dict_bin['kk_hm_trans'] = hm_1h2h_alpha


                array_num = np.arange(1, len(z_array_block) + 1, 1)
                array_num_python = array_num - 1
                M_mat_block = block[nl_power, 'mass_h_um']
                z_array_selum = z_array_block[array_num_python]
                M_array_block = M_mat_block[0, :]
                k_array_block = block[nl_power, 'k_h']
                nk_bl, nz_bl = len(k_array_block), len(z_array_block)
                um_block = np.zeros((len(z_array_selum), len(M_array_block), nk_bl))
                for j in range(len(array_num)):
                    um_block_j = block[nl_power, 'um_' + str(int(array_num[j]))]
                    um_block[j, :, :] = um_block_j.T
                um_block_allinterp = RegularGridInterpolator(
                    (z_array_selum, np.log(M_array_block), np.log(k_array_block)),
                    np.log(um_block + 1e-200), fill_value=None, bounds_error=False)

                other_params_dict['um_block_allinterp'] = um_block_allinterp
                other_params_dict_bin['um_block_allinterp'] = um_block_allinterp
                other_params_dict_bin['um_block'] = um_block

                bmkz_block = block[nl_power, 'bt_out']

                # bkm_block_allinterp = RegularGridInterpolator((z_array_block, np.log(k_array_block)),
                #                                               np.log(bmkz_block), fill_value=None,
                #                                               bounds_error=False)
                bkm_block_allinterp = interpolate.RectBivariateSpline(z_array_block, np.log(k_array_block),
                                                                       np.log(bmkz_block))

                other_params_dict['bkm_block_allinterp'] = bkm_block_allinterp
                other_params_dict_bin['bkm_block_allinterp'] = bkm_block_allinterp

                other_params_dict['z_array_block'] = z_array_block
                other_params_dict_bin['z_array_block'] = z_array_block

                other_params_dict['k_array_block'] = k_array_block
                other_params_dict_bin['k_array_block'] = k_array_block

                other_params_dict['M_array_block'] = M_array_block
                other_params_dict_bin['M_array_block'] = M_array_block

                gm_block = np.zeros((len(z_array_selum), len(M_array_block)))
                dndm_Marray = np.zeros((len(z_array_selum), len(other_params_dict['M_array'])))
                nu_mat_block = block[nl_power, 'nu_out']
                rho_m = 2.775e11 * cosmo_params_dict['Om0']
                for j in range(len(array_num)):
                    gm_block_j = block[nl_power, 'g_' + str(int(array_num[j]))].T[:,0]
                    gm_block[j,:] =  gm_block_j
                    gm_interp = interpolate.interp1d(np.log(M_array_block),np.log(gm_block_j+1e-300),fill_value='extrapolate')
                    gm_val_Marray = np.exp(gm_interp(np.log(other_params_dict['M_array'])))

                    nu_block_j = np.exp(0.5*(np.log(nu_mat_block[j,1:]) + np.log(nu_mat_block[j,:-1])))
                    M_block_j = np.exp(0.5*(np.log(M_mat_block[j,1:]) + np.log(M_mat_block[j,:-1])))
                    dlognu_dlogM = (np.log(nu_mat_block[j,1:]) - np.log(nu_mat_block[j,:-1]))/(np.log(M_mat_block[j,1:]) - np.log(M_mat_block[j,:-1]))
                    dnu_dM = (nu_block_j/M_block_j)*dlognu_dlogM
                    dnu_dM_interp = interpolate.interp1d(np.log(M_block_j),np.log(dnu_dM),fill_value='extrapolate')
                    dnu_dM_Marray = np.exp(dnu_dM_interp(np.log(other_params_dict['M_array'])))
                    dndm_Marray[j,:] = gm_val_Marray*dnu_dM_Marray*(rho_m/other_params_dict['M_array'])

                dndm_Marray_allinterp = interpolate.RectBivariateSpline(z_array_selum, np.log(other_params_dict['M_array']),
                                                                       np.log(dndm_Marray + 1e-300))

                dndm_Marray_yx = np.exp(dndm_Marray_allinterp(other_params_dict['z_array'], np.log(other_params_dict['M_array']),grid=True))
                if use_dndm_block:
                    other_params_dict_bin['dndm_array'] = dndm_Marray_yx
                    other_params_dict['dndm_array'] = dndm_Marray_yx

                other_params_dict_bin['nu_block'] = nu_mat_block
                other_params_dict_bin['gm_block'] = gm_block
                other_params_dict_bin['rhobar_M_block'] = rho_m/M_mat_block

                other_params_dict_bin['pkmm1h_cs'] = block[nl_power, 'p_k_1h']
                other_params_dict_bin['pkmm2h_cs'] = block[nl_power, 'p_k_2h']
                other_params_dict_bin['pkmmtot_cs'] = block[nl_power, 'p_k']
                # dndmdict = {'nu_block':nu_mat_block,'gm_block':gm_block,'M_block':M_mat_block, 'z_block':z_array_block,'k_block':k_array_block,'sigma_block':block[nl_power, 'sigma_out']}
                # dndmdict['M_yx'] = other_params_dict['M_array']
                # dndmdict['z_yx'] = other_params_dict['z_array']
                # dndmdict['dndm_yx'] = other_params_dict['dndm_array']
                #
                # # dndmdict['']
                #
                # savefname = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/dndmdict_comp_interp.pk'
                # with open(savefname, 'wb') as f:
                #     dill.dump(dndmdict, f)

                # savedict = {'other':other_params_dict_bin,'cosmo':cosmo_params_dict_bin}
                # savefname = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/all_vars_dict.pk'
                # with open(savefname, 'wb') as f:
                #     dill.dump(savedict, f)
                # import pdb; pdb.set_trace()

                # get_Pe_HSE = True
                get_Pe_HSE = False
                if get_Pe_HSE:
                    if verbose:
                        print('getting HSE pressure profile')

                    H0 = 100. * (u.km / (u.s * u.Mpc))
                    G_new = const.G.to(u.Mpc ** 3 / ((u.s ** 2) * u.M_sun))
                    rho_crit = ((3 * (H0 ** 2) / (8 * np.pi * G_new)).to(u.M_sun / (u.Mpc ** 3))).value

                    # um_block_allinterp = RegularGridInterpolator(
                        # (z_array_selum, np.log(M_array_block), np.log(k_array_block)),
                        # np.log(um_block + 1e-200), fill_value=None, bounds_error=False)

                    coeff = (const.G * (const.M_sun ** 2) / ((1.0 * u.Mpc) ** 4)).to((u.eV / (u.cm ** 3))).value
                    coeff_si = (const.G * (const.M_sun ** 2) / ((1.0 * u.Mpc) ** 4)).value
                    nk = 200
                    k_array = np.logspace(-3,3,nk)
                    P_mat = np.zeros((len(z_array_selum), len(M_array_block), nk-1))
                    pressure_model_delta = 200
                    M_from_rho = np.zeros((len(z_array_selum), len(M_array_block)))
                    rhoM = np.zeros((len(z_array_selum), len(M_array_block),nk-1))
                    for jz in range(len(z_array_selum)):
                        print(jz)
                        for jM in range(len(M_array_block)):
                            ksel = np.where((k_array_block > 1e-4) & (k_array_block < 20))[0]
                            um_interp = interpolate.interp1d(np.log(k_array_block[ksel]),(rho_m)*np.log(um_block[jz,jM,ksel]),fill_value='extrapolate')
                            r_all, rho_M = P2xi(k_array)(np.exp(um_interp(np.log(k_array))))
                            r_cent, rho_M_cent = 0.5*(r_all[1:] + r_all[:-1]), 0.5*(rho_M[1:] + rho_M[:-1])
                            rho_cent = rho_M_cent * M_array_block[jM]
                            rhoM[jz,jM,:] = rho_cent
                            dr_all = r_all[1:] - r_all[:-1]
                            int_M = 4*np.pi * (r_cent**2) * rho_cent * dr_all
                            M_ltr = np.cumsum(int_M)
                            to_int = -1 * G_new.value * M_ltr * (1./r_cent**2) * dr_all
                            rhs_all = np.cumsum(to_int)
                            gv = 1.14
                            rho0 = rho_cent[0]
                            rvir = np.power(3 * M_array_block[jM] / (4 * np.pi * 200. * rho_crit), 1. / 3.)
                            rcv = np.where(r_cent > rvir)[0][0]
                            M_from_rho[jz,jM] = M_ltr[rcv]

                            P0 = (coeff / 2.) * pressure_model_delta * (
                                    0.044 / 0.3) * M_array_block[jM] * rho_crit / rvir
                            P0_rho0 = G_new.value * (M_array_block[jM] / rvir)
                            rhs_mod = (1 + (1./P0_rho0) * ((gv-1)/gv) * rhs_all)**(gv/(gv-1))
                            Pr_HSE = rhs_mod * P0
                            P_mat[jz, jM, :] = Pr_HSE
                            # if M_array_block[jM] > 1e14:
                                # import ipdb; ipdb.set_trace() # BREAKPOINT

                    np.savez('/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/save_Pe_HSE.npz',r=r_cent,M=M_array_block, z=z_array_selum,P=P_mat)
                    import ipdb; ipdb.set_trace() # BREAKPOINT


            if get_bp:
                PS = Powerspec(cosmo_params_dict_bin, hod_params_dict_bin, pressure_params_dict_bin, other_params_dict_bin)
                if bp_model == 'const':
                    bpz = block[sec_name, 'bp--' + str(binvs)]
                    xi_ky_2h_array = np.zeros_like(theta_array)
                    for jt in range(len(theta_array)):
                        xi_ky_2h_array[jt] = PS.get_xi_kappy_2h(theta_array[jt], bpz_keVcm3=bpz*1e-7,bp_model='const')


                if bp_model == 'linear':
                    bpz0 = block[sec_name, 'bpz0']
                    bpalpha = block[sec_name, 'bpalpha']

                    xi_ky_2h_array = np.zeros_like(theta_array)
                    for jt in range(len(theta_array)):
                        xi_ky_2h_array[jt] = PS.get_xi_kappy_2h(theta_array[jt], bpz0_keVcm3=bpz0*1e-7, bpalpha=bpalpha*1e-7,
                                        bp_model='linear')

                block[sec_save_name, 'theory_corrf_' + 'gty' + '_bin_' + str(binvs) + '_' + str(0)] = xi_ky_2h_array
                block[sec_save_name, 'xcoord_' + 'gty' + '_bin_' + str(binvs) + '_' + str(0)] = theta_array
            else:
                ti = time.time()
                try:
                    PrepDV_fid = PrepDataVec(cosmo_params_dict_bin, hod_params_dict_bin, pressure_params_dict_bin, other_params_dict_bin)
                except:
                    print(tb.format_exc())
                if verbose:
                    print('Setting up DV took : ' + str(time.time() - ti) + 's')

                if 'uyl_zM_dict' not in other_params_dict.keys():
                    other_params_dict['uyl_zM_dict'] = PrepDV_fid.uyl_zM_dict
                    other_params_dict['byl_z_dict'] = PrepDV_fid.byl_z_dict

                if 'uml_zM_dict' not in other_params_dict.keys():
                    other_params_dict['uml_zM_dict'] = PrepDV_fid.uml_zM_dict
                    other_params_dict['bml_z_dict'] = PrepDV_fid.bml_z_dict

                PrepDV_dict_allbins['ukl_zM_dict' + str(binvs)] = PrepDV_fid.ukl_zM_dict
                PrepDV_dict_allbins['uIl_zM_dict' + str(binvs)] = PrepDV_fid.uIl_zM_dict
                PrepDV_dict_allbins['ugl_zM_dict' + str(binvl)] = PrepDV_fid.ugl_zM_dict
                PrepDV_dict_allbins['ugl_cross_zM_dict' + str(binvl)] = PrepDV_fid.ugl_cross_zM_dict
                PrepDV_dict_allbins['bkl_z_dict' + str(binvs)] = PrepDV_fid.bkl_z_dict
                PrepDV_dict_allbins['bIl_z_dict' + str(binvs)] = PrepDV_fid.bIl_z_dict
                PrepDV_dict_allbins['bgl_z_dict' + str(binvl)] = PrepDV_fid.bgl_z_dict
                PrepDV_dict_allbins['Cl_noise_gg_l_array' + str(binvl)] = PrepDV_fid.Cl_noise_gg_l_array
                PrepDV_dict_allbins['Cl_noise_kk_l_array' + str(binvs)] = PrepDV_fid.Cl_noise_kk_l_array

                if 'uyl_zM_dict0' not in PrepDV_dict_allbins.keys():
                    PrepDV_dict_allbins['uyl_zM_dict0'] = PrepDV_fid.uyl_zM_dict
                    PrepDV_dict_allbins['byl_z_dict0'] = PrepDV_fid.byl_z_dict
                    PrepDV_dict_allbins['uml_zM_dict0'] = PrepDV_fid.uml_zM_dict
                    PrepDV_dict_allbins['bml_z_dict0'] = PrepDV_fid.bml_z_dict
                    PrepDV_dict_allbins['PrepDV_fid'] = PrepDV_fid
                    PrepDV_dict_allbins['Cl_noise_yy_l_array'] = PrepDV_fid.Cl_noise_yy_l_array
                    PrepDV_dict_allbins['Cl_tot_yy_l_array'] = PrepDV_fid.Cl_tot_yy_l_array
                    PrepDV_dict_allbins['bins_source'] = bins_source
                    PrepDV_dict_allbins['bins_lens'] = bins_lens
                    PrepDV_dict_allbins['run_cov_pipe'] = run_cov_pipe
                    PrepDV_dict_allbins['theta_min'] = theta_min
                    PrepDV_dict_allbins['theta_max'] = theta_max
                    PrepDV_dict_allbins['ntheta'] = ntheta
                    PrepDV_dict_allbins['theta_array'] = theta_array
                    PrepDV_dict_allbins['theta_array_all'] = theta_array_all
                    PrepDV_dict_allbins['analysis_coords'] = analysis_coords
                    PrepDV_dict_allbins['gg_doauto'] = gg_doauto
                    PrepDV_dict_allbins['fsky_dict'] = PrepDV_fid.fsky
                    # PrepDV_dict_allbins['verbose'] = other_params_dict['verbose']
                    PrepDV_dict_allbins['verbose'] = verbose
                    PrepDV_dict_allbins['put_IA'] = other_params_dict['put_IA']
                    PrepDV_dict_allbins['only_2h_IA'] = other_params_dict['only_2h_IA']
                    PrepDV_dict_allbins['sec_save_name'] = sec_save_name
                    PrepDV_dict_allbins['save_detailed_DV'] = save_detailed_DV
                    PrepDV_dict_allbins['add_beam_to_theory'] = other_params_dict['add_beam_to_theory']
                    PrepDV_dict_allbins['beam_fwhm_arcmin'] = other_params_dict['beam_fwhm_arcmin']

    if not get_bp:
        try:
            DV = DataVec(PrepDV_dict_allbins, block)
        except:
            print(tb.format_exc())
        if save_DV:
            with open(save_data_fname,'wb') as f:
                dill.dump(DV,f)
            import ipdb; ipdb.set_trace() # BREAKPOINT

        # z_block, k_block, pktot_block = block.get_grid(nl_power, "z", "k_h", "p_k")
        # z_block, k_block, pk1h_block = block.get_grid(nl_power, "z", "k_h", "p_k_1h")
        # z_block, k_block, pk2h_block = block.get_grid(nl_power, "z", "k_h", "p_k_2h")
        # outdict = {'k':k_block, 'z':z_block,'Pk1h':pk1h_block,'Pk2h':pk2h_block,'Pktot':pktot_block}
        # savefname = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/Pkmmdict_cosmosis_imead1.pk'
        # with open(savefname, 'wb') as f:
        #     dill.dump(outdict, f)
        # import pdb; pdb.set_trace()
    return 0

def cleanup(config):
    pass
