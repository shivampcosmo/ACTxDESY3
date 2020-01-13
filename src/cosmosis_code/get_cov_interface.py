import sys, os
from cosmosis.datablock import names, option_section, BlockError
sys.path.insert(0, os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/helper/')
# sys.path.insert(0, '../../helper/')
# sys.path.insert(0, '../')
import numpy as np
import copy
import pdb
import ast
import scipy as sp
from cross_corr_funcs import DataVec, general_hm
from scipy import interpolate
import multiprocessing
import dill
from configobj import ConfigObj
from configparser import ConfigParser
import pickle as pk


def get_value(section, value, config_run, config_def):
    if section in config_run.keys() and value in config_run[section].keys():
        val = config_run[section][value]
    else:
        val = config_def[section][value]
    return val



def QR_inverse(matrix):
    _Q,_R = np.linalg.qr(matrix)
    return np.dot(_Q,np.linalg.inv(_R.T))

def read_ini(ini_file, ini_def=None, twopt_file=None):
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
    other_params_dict['z_array'] = np.logspace(np.log10(other_params_dict['z_array_min']),
                                               np.log10(other_params_dict['z_array_max']),
                                               other_params_dict['num_z'])
    other_params_dict['x_array'] = np.logspace(np.log10(other_params_dict['xmin']), np.log10(other_params_dict['xmax']),
                                               other_params_dict['num_x'])



    if twopt_file is None:
        if other_params_dict['larray_spacing'] == 'log':
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
    dndm_array, bm_array = ghmf.get_dndm_bias(M_mat_mdef, mdef_analysis)
    halo_conc_mdef = ghmf.get_halo_conc_Mz(M_mat_mdef, mdef_analysis)
    pkzlin_interp = ghmf.get_Pklin_zk_interp()
    other_params_dict['pkzlin_interp'] = pkzlin_interp
    other_params_dict['dndm_array'] = dndm_array
    other_params_dict['bm_array'] = bm_array
    other_params_dict['halo_conc_mdef'] = halo_conc_mdef

    ini_info = {'cosmo_params_dict': cosmo_params_dict, 'pressure_params_dict': pressure_params_dict,
                'hod_params_dict': hod_params_dict, 'other_params_dict': other_params_dict}

    return ini_info


def setup(options):
    params_files_dir = options.get_string(option_section, "params_files_dir")
    params_file = options.get_string(option_section, "params_file")
    params_def_file = options.get_string(option_section, "params_def_file")
    twopt_file = options.get_string(option_section, 'twopt_file')
    # ini_info = read_ini(params_files_dir + params_file, ini_def=params_files_dir + params_def_file,
    #                     twopt_file=twopt_file)
    ini_info = read_ini(params_files_dir + params_file, ini_def=params_files_dir + params_def_file)

    z_edges = ast.literal_eval(options.get_string(option_section, "z_edges", default='[ 0.20, 0.40, 0.55, 0.70, 0.85, 0.95, 1.05 ]'))
    bins_numbers = ast.literal_eval(options.get_string(option_section, "bins_numbers", default='[ 1,2,3,4,5,6 ]'))

    sec_name = options.get_string(option_section, "sec_name", default='get_cov')
    sec_save_name = options.get_string(option_section, "sec_save_name", default='save_get_cov')
    save_cov_fname = options.get_string(option_section, "save_cov_fname")
    save_block_fname = options.get_string(option_section, "save_block_fname")

    return ini_info, bins_numbers, z_edges, twopt_file, sec_name, sec_save_name, save_cov_fname, save_block_fname


def execute(block, config):
    ini_info, bins_numbers, z_edges, twopt_file, sec_name, sec_save_name, save_cov_fname, save_block_fname = config
    clf = pk.load(open(twopt_file, 'rb'))

    other_params_dict = ini_info['other_params_dict']
    cosmo_params_dict = ini_info['cosmo_params_dict']
    pressure_params_dict = ini_info['pressure_params_dict']
    hod_params_dict = ini_info['hod_params_dict']
    nl = len(other_params_dict['l_array'])
    nstats = len(other_params_dict['stats_analyze'])
    nbins = len(bins_numbers)
    cov_fid_G = np.zeros((nstats * nl * nbins, nstats * nl * nbins))
    cov_fid_NG = np.zeros((nstats * nl * nbins, nstats * nl * nbins))
    Cl_vec = np.zeros(nstats * nl * nbins)
    Cl_vec_data = np.zeros(nstats * nl * nbins)
    zmin_array = np.array(z_edges)[:-1]
    zmax_array = np.array(z_edges)[1:]
    ell_all = []

    for binv in bins_numbers:
        cosmo_params_dict_bin = copy.deepcopy(cosmo_params_dict)
        pressure_params_dict_bin = copy.deepcopy(pressure_params_dict)
        hod_params_dict_bin = copy.deepcopy(hod_params_dict)
        other_params_dict_bin = copy.deepcopy(other_params_dict)

        for key in block.keys():
            if key[0] == sec_name:
                param_val = key[1]
                ln = list(param_val)
                bin_n_sep = [n for n, x in enumerate(ln) if x == '-']
                var_name = ''.join(ln[:bin_n_sep[-2]])
                bin_n = int(ln[-1])

                if bin_n == 0:
                    for pressure_keys in pressure_params_dict_bin.keys():
                        if var_name == pressure_keys.lower():
                            pressure_params_dict_bin[pressure_keys] = block[key]

                if bin_n == binv:
                    for cosmo_keys in cosmo_params_dict_bin.keys():
                        if var_name == cosmo_keys.lower():
                            cosmo_params_dict_bin[cosmo_keys] = block[key]
                            other_params_dict_bin['do_vary_cosmo'] = True

                    for hod_keys in hod_params_dict_bin.keys():
                        if var_name == hod_keys.lower():
                            hod_params_dict_bin[hod_keys] = block[key]

        other_params_dict_bin['ng_zarray'] = block['nz_lens', 'z']
        other_params_dict_bin['ng_value'] = block['nz_lens', 'bin_' + str(binv)]
        other_params_dict_bin['ng_zarray_source'] = block['nz_source', 'z']
        other_params_dict_bin['ng_value_source'] = block['nz_source', 'bin_' + str(binv)]
        other_params_dict_bin['cosmo_fid'] = cosmo_params_dict_bin
        other_params_dict_bin['hod_fid'] = hod_params_dict_bin
        other_params_dict_bin['pressure_fid'] = pressure_params_dict_bin

        # other_params_dict_bin['Clyy_measured'] = clf[('ymap', 'ymap')][('y', 'y')]['true'][0]
        # other_params_dict_bin['Clgg_measured'] = \
        #     clf[('galaxy_density', 'galaxy_density')][(binv - 1, binv - 1)]['true'][0]
        # other_params_dict_bin['ell_measured'] = clf['ell']
        nbar_bin = block[sec_name,'nbar--' + str(binv)]
        other_params_dict_bin['nbar'] = nbar_bin
        other_params_dict_bin['noise_kappa'] = block[sec_name,'noisek--' + str(binv)]


        if other_params_dict_bin['do_vary_cosmo']:
            del other_params_dict_bin['pkzlin_interp'], other_params_dict_bin['dndm_array'], other_params_dict_bin[
                'bm_array'], other_params_dict_bin['halo_conc_mdef']

        # import pdb;pdb.set_trace()
        DV_fid = DataVec(cosmo_params_dict_bin, hod_params_dict_bin, pressure_params_dict_bin, other_params_dict_bin)

        block[sec_save_name, 'theory_Clgg_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['gg']['total']
        block[sec_save_name, 'theory_Clgy_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['gy']['total']
        block[sec_save_name, 'theory_Clkk_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['kk']['total']
        block[sec_save_name, 'theory_Clky_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['ky']['total']
        block[sec_save_name, 'theory_Clyy'] = DV_fid.Cl_dict['yy']['total']

        block[sec_save_name, 'theory_Clkk1h_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['kk']['1h']
        block[sec_save_name, 'theory_Clky1h_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['ky']['1h']
        block[sec_save_name, 'theory_Clgg1h_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['gg']['1h']
        block[sec_save_name, 'theory_Clgy1h_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['gy']['1h']
        block[sec_save_name, 'theory_Clyy1h'] = DV_fid.Cl_dict['yy']['1h']

        block[sec_save_name, 'theory_Clkk2h_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['kk']['2h']
        block[sec_save_name, 'theory_Clky2h_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['ky']['2h']
        block[sec_save_name, 'theory_Clgg2h_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['gg']['2h']
        block[sec_save_name, 'theory_Clgy2h_bin_' + str(binv) + '_' + str(binv)] = DV_fid.Cl_dict['gy']['2h']
        block[sec_save_name, 'theory_Clyy2h'] = DV_fid.Cl_dict['yy']['2h']

        block[sec_save_name, 'theory_ell'] = DV_fid.l_array

        stats_analyze_pairs_all = DV_fid.stats_analyze_pairs_all
        cov_fid_dict_G = DV_fid.get_cov_G()
        cov_fid_dict_NG = DV_fid.get_cov_NG()

        j1 = 0
        for stats in other_params_dict['stats_analyze']:

            if stats == 'kk':
                block[sec_save_name, 'covG_kk_kk_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_G[
                    'kk' + '_' + 'kk']
                block[sec_save_name, 'covNG_kk_kk_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_NG[
                    'kk' + '_' + 'kk']
                block[sec_save_name, 'cov_total_kk_kk_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_G[
                                                                                                 'kk' + '_' + 'kk'] + \
                                                                                             cov_fid_dict_NG[
                                                                                                 'kk' + '_' + 'kk']

            if stats == 'gg':
                block[sec_save_name, 'covG_gg_gg_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_G[
                    'gg' + '_' + 'gg']
                block[sec_save_name, 'covNG_gg_gg_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_NG[
                    'gg' + '_' + 'gg']
                block[sec_save_name, 'cov_total_gg_gg_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_G[
                                                                                                 'gg' + '_' + 'gg'] + \
                                                                                             cov_fid_dict_NG[
                                                                                                 'gg' + '_' + 'gg']

                # Cl_vec_data[j1 * (nl * nbins) + (binv - 1) * nl:j1 * (nl * nbins) + binv * nl] = \
                #     clf[('galaxy_density', 'galaxy_density')][(binv - 1, binv - 1)]['true'][0] - 1./nbar_bin
                # Cl_gg_data_bin =  clf[('galaxy_density', 'galaxy_density')][(binv - 1, binv - 1)]['true'][0] - 1./nbar_bin
                # cov_gg_bin = cov_fid_dict_G['gg' + '_' + 'gg'] + cov_fid_dict_NG['gg' + '_' + 'gg']
                # inv_cov_bin = QR_inverse(cov_gg_bin)
                # snr_bin = np.sqrt(np.dot(np.array([Cl_gg_data_bin]), np.dot(inv_cov_bin, np.array([Cl_gg_data_bin]).T)))
                # print 'SNR gg bin:' + str(binv) + ', ' + str(zmin_array[binv-1]) + '<z<' + str(zmax_array[binv-1]) + '=' + str(np.round(snr_bin[0][0],2)) + ' sigma'

            if stats == 'ky':
                block[sec_save_name, 'covG_ky_ky_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_G[
                    'ky' + '_' + 'ky']
                block[sec_save_name, 'covNG_ky_ky_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_NG[
                    'ky' + '_' + 'ky']
                block[sec_save_name, 'cov_total_ky_ky_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_G[
                                                                                                 'ky' + '_' + 'ky'] + \
                                                                                             cov_fid_dict_NG[
                                                                                                 'ky' + '_' + 'ky']

            if stats == 'gy':
                block[sec_save_name, 'covG_gy_gy_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_G[
                    'gy' + '_' + 'gy']
                block[sec_save_name, 'covNG_gy_gy_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_NG[
                    'gy' + '_' + 'gy']
                block[sec_save_name, 'cov_total_gy_gy_bin_' + str(binv) + '_' + str(binv)] = cov_fid_dict_G[
                                                                                                 'gy' + '_' + 'gy'] + \
                                                                                             cov_fid_dict_NG[
                                                                                                 'gy' + '_' + 'gy']
                # Cl_vec_data[j1 * (nl * nbins) + (binv - 1) * nl:j1 * (nl * nbins) + binv * nl] = \
                # clf[('galaxy_density', 'ymap')][(binv - 1, 'y')]['true'][0]
                # Cl_gy_data_bin =  clf[('galaxy_density', 'ymap')][(binv - 1, 'y')]['true'][0]
                # cov_gy_bin = cov_fid_dict_G['gy' + '_' + 'gy'] + cov_fid_dict_NG['gy' + '_' + 'gy']
                # inv_cov_bin = QR_inverse(cov_gy_bin)
                # snr_bin = np.sqrt(np.dot(np.array([Cl_gy_data_bin]), np.dot(inv_cov_bin, np.array([Cl_gy_data_bin]).T)))
                # print 'SNR gy bin:' + str(binv) + ', ' + str(zmin_array[binv-1]) + '<z<' + str(zmax_array[binv-1]) + '=' + str(np.round(snr_bin[0][0],2)) + ' sigma'

            if stats == 'yy':
                block[sec_save_name, 'covG_yy_yy'] = cov_fid_dict_G['yy' + '_' + 'yy']
                block[sec_save_name, 'covNG_yy_yy'] = cov_fid_dict_NG['yy' + '_' + 'yy']
                block[sec_save_name, 'cov_total_yy_yy'] = cov_fid_dict_G['yy' + '_' + 'yy'] + cov_fid_dict_NG[
                    'yy' + '_' + 'yy']

            Cl_vec[j1 * (nl * nbins) + (binv - 1) * nl:j1 * (nl * nbins) + binv * nl] = DV_fid.Cl_dict[stats]['total']

            ell_all.append(clf['ell'])

            j1 += 1

        # pdb.set_trace()
        k = 0
        for j1 in range(nstats):
            for j2 in range(nstats):
                stats_analyze_1, stats_analyze_2 = stats_analyze_pairs_all[k]

                cov_G_j1j2 = cov_fid_dict_G[stats_analyze_1 + '_' + stats_analyze_2]
                cov_NG_j1j2 = cov_fid_dict_NG[stats_analyze_1 + '_' + stats_analyze_2]

                cov_fid_G[j2 * (nl * nbins) + (binv - 1) * nl:j2 * (nl * nbins) + binv * nl,
                j1 * (nl * nbins) + (binv - 1) * nl:j1 * (nl * nbins) + binv * nl] = cov_G_j1j2
                cov_fid_NG[j2 * (nl * nbins) + (binv - 1) * nl:j2 * (nl * nbins) + binv * nl,
                j1 * (nl * nbins) + (binv - 1) * nl:j1 * (nl * nbins) + binv * nl] = cov_NG_j1j2

                k += 1

        cov_fid = cov_fid_G + cov_fid_NG
        block[sec_save_name, 'cov_total'] = cov_fid
        block[sec_save_name, 'cov_G'] = cov_fid_G
        block[sec_save_name, 'cov_NG'] = cov_fid_NG
        block[sec_save_name, 'data_Clyy'] = clf[('ymap', 'ymap')][('y', 'y')]['true'][0]
        block[sec_save_name, 'data_Clgy_bin_' + str(binv) + '_' + str(binv)] = \
            clf[('galaxy_density', 'ymap')][(binv - 1, 'y')]['true'][0]
        block[sec_save_name, 'data_Clgg_bin_' + str(binv) + '_' + str(binv)] = \
            clf[('galaxy_density', 'galaxy_density')][(binv - 1, binv - 1)]['true'][0] - 1./nbar_bin

    block[sec_save_name, 'ell'] = clf['ell']
    block[sec_save_name, 'theory_dv'] = Cl_vec
    savecov = {'cov_total':cov_fid, 'cov_G':cov_fid_G, 'cov_NG':cov_fid_NG, 'mean':Cl_vec_data,'ell_all':ell_all}
    pk.dump(savecov,open(save_cov_fname,'wb'))

    out_dict = {}
    all_keys = block.keys()
    for key in all_keys:
        out_dict[key[1]] = block[key]

    np.savez(save_block_fname, **out_dict)
    pdb.set_trace()



    return 0


def cleanup(config):
    pass
