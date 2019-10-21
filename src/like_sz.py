import sys, os
from cosmosis.datablock import names, option_section, BlockError
sys.path.insert(0, '../../helper/')
sys.path.insert(0, '../')
import numpy as np
import copy
import pdb
import call_pipe as fp
import ast
import scipy as sp
from scipy import interpolate
import multiprocessing
import dill
# import pickle as pk
from configobj import ConfigObj
from configparser import ConfigParser


def get_newvalues_array(block, sec_name, params_names):
    new_values = []

    for param in params_names:
        val = block[sec_name, param]
        new_values.append(val)

    return new_values


def setup(options):
    twopt_file = options.get_string(option_section, "twopt_file")
    params_file = options.get_string(option_section, "params_file")
    d_list = (dill.load(open(twopt_file, 'rb')))

    # d_list = (pk.load(open(twopt_file, 'rb')))
    p_list = (ConfigObj(params_file, unrepr=True))

    sec_name = options.get_string(option_section, "sec_name")
    # pdb.set_trace()
    fp_obj = fp.FisherPlotter(d_list)
    return fp_obj, sec_name


def execute(block, config):
    fp_obj, sec_name = config
    params_vary_all = fp_obj.fisher_params_vary

    new_values = get_newvalues_array(block, sec_name, params_vary_all)

    cosmo, hod, pressure, other = copy.deepcopy(fp_obj.cosmo_params_fid), copy.deepcopy(fp_obj.hod_params_fid), copy.deepcopy(
        fp_obj.pressure_params_fid), copy.deepcopy(fp_obj.other_params_fid)

    cosmo_new, hod_new, pressure_new = fp_obj.set_values(cosmo, hod, pressure, new_values)
    other_new = copy.deepcopy(other)

    l_array, Cl_fid_dict, Cl_dicts_samples, cov_fid, inv_cov_fid, chi2_samples = fp_obj.get_Clyg_mat_Makiya18(
        [cosmo_new], [pressure_new], [hod_new], [other_new])

    like_sz = -0.5*chi2_samples[0][0][0]

    likes = names.likelihoods

    # pdb.set_trace()

    block[likes, 'sz_LIKE'] = like_sz
    block["data_vector", 'sz_inverse_covariance'] = inv_cov_fid
    block["data_vector", 'sz_theory'] = Cl_dicts_samples[0]['yg']['total']

    return 0


def cleanup(config):
    pass

