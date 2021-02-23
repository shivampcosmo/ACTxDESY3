import sys, os
import os.path
from os import path
try:
    from cosmosis.datablock import names, option_section, BlockError
except:
    pass
import numpy as np
import copy
import pdb
import ast
import scipy as sp
from scipy import interpolate
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


def setup(options):
    config = dict()
    save_files_dir = options.get_string(option_section, "save_files_dir")
    save_file = options.get_string(option_section, "save_file")
    config['sec_save_name'] = options.get_string(option_section, "sec_save_name", 'save_theory')
    save_fname = save_files_dir + save_file
    config['save_fname'] = save_fname
    config['save_dir'] = save_files_dir
    return config


def execute(block, config):
    save_fname = config['save_fname']
    save_dir = config['save_dir']

    if path.isdir(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    
    if path.isfile(save_fname):
        pass
    else:
        saved = {}
        with open(save_fname,'wb') as f:
                dill.dump(saved,f)
        

    dfd = dill.load(open(save_fname,'rb'))
    # Ahmv = block['halo_model_parameters','a']
    Ahmv = block['halo_model_parameters','eta_0']
    # Ahmv = block['theory_yx','a_ia--0']
    # Ahmv = block['theory_yx','eta_ia--0']

    savedjA = {}
    theta_array = block.get_double_array_1d(config['sec_save_name'], "xcoord_" + 'gty1' + '_bin_' + str(1) + '_' + str(0))
    savedjA['theta'] = theta_array

    for ji in range(4):
        corrf_stat =  block.get_double_array_1d(config['sec_save_name'],'theory_corrf_' + 'gty1' + '_bin_' + str(ji+1) + '_' + str(0))
        bin_str = 'yt_' + str(1) + 'bin_' + str(ji+1) + '_' + str(0)
        savedjA[bin_str] = corrf_stat

        corrf_stat =  block.get_double_array_1d(config['sec_save_name'],'theory_corrf_' + 'gty2' + '_bin_' + str(ji+1) + '_' + str(0))
        bin_str = 'yt_' + str(2) + 'bin_' + str(ji+1) + '_' + str(0)
        savedjA[bin_str] = corrf_stat

    dfd[Ahmv] = savedjA
    with open(save_fname,'wb') as f:
        dill.dump(dfd,f)

    # import ipdb; ipdb.set_trace() # BREAKPOINT
    return 0

def cleanup(config):
    pass
