import sys, os
import copy
from cosmosis.datablock import names, option_section, BlockError
import ast
import numpy as np
import math
#
#  ******** setup options for cosmosis  *************************************************
def setup(options):
    config = dict()
    bins_source = ast.literal_eval(options.get_string(option_section, "bins_source", "[1, 2, 3, 4]"))
    bins_lens = ast.literal_eval(options.get_string(option_section, "bins_lens", "[1, 2, 3, 4]"))
    auto_only = options.get_string(option_section,"auto_only", default="")
    auto_only = auto_only.split()
    config['sec_save_name'] = options.get_string(option_section, "sec_save_name", 'save_theory')
    config['verbose'] = options.get_bool(option_section,'verbose',False)
    # load clustering-z quantities
    config['bins_source'] = bins_source
    config['bins_lens'] = bins_lens
    config[0] = '0'
    config['auto_only'] = auto_only
    return config

def execute(block, config):
    if config['verbose']:
        print ('*******  CONVERSION MODULE *********')
    conversion_dict = dict()
    conversion_dict['gty1']  = ['shear_compton_xi','bins_source',0]
    conversion_dict['yy1']   = ['compton_compton_xi',0,0]
    conversion_dict['gy1']   = ['galaxy_compton_xi','bins_lens',0]
    conversion_dict['gty2']  = ['shear_compton1_xi','bins_source',0]
    conversion_dict['yy2']   = ['compton1_compton1_xi',0,0]
    conversion_dict['gy2']   = ['galaxy_compton1_xi','bins_source',0]
    conversion_dict['gg']   = ['galaxy_xi','bins_lens','bins_lens']
    conversion_dict['gk']   = ['galaxy_shear_xi','bins_lens','bins_source']
    conversion_dict['kk']   = ['shear_xi_plus','bins_source','bins_source']
    conversion_dict['kkm']  = ['shear_xi_minus','bins_source','bins_source']
    for key in conversion_dict.keys():
        name_stat = conversion_dict[key][0]
        for i in (config[conversion_dict[key][1]]):
            for j in (config[conversion_dict[key][2]]):
                try:
                    xcoord_array = block.get_double_array_1d(config['sec_save_name'], "xcoord_" + key + '_bin_' + str(i) + '_' + str(j))
                    corrf_stat =  block.get_double_array_1d(config['sec_save_name'],'theory_corrf_' + key + '_bin_' + str(i) + '_' +str(j))
                    # the following lines are to convert the convention i_0 for shear x y in Shivam's code and i_i for cosmosis.
                    ix = copy.copy(i)
                    jx = copy.copy(j)
                    if ((np.int(i) ==0) and (np.int(j)==0)):
                        ix =1
                        jx =1
                    if ((np.int(i) ==0) and (np.int(j)!=0)):
                        ix = copy.copy(j)
                    if ((np.int(i) !=0) and (np.int(j)==0)):
                        jx = copy.copy(i)
                    # save to block.
                    name = 'bin_%d_%d' % (np.int(ix), np.int(jx))
                    block[name_stat, name] = corrf_stat
                    name = 'bin_%d_%d' % (np.int(jx), np.int(ix))
                    block[name_stat, name] = corrf_stat
                    # import ipdb; ipdb.set_trace() # BREAKPOINT
                except:
                    pass
                    if config['verbose']:
                        print ('warning'+ key + '_bin_' + str(i) + '_' + str(j)+ ' not predicted by theory code.')
        try:
            block[name_stat, "nbin_a"] = len(conversion_dict[key][0])
        except:
            block[name_stat, "nbin_a"] = 1
        try:
            block[name_stat, "nbin_b"] = len(conversion_dict[key][1])
        except:
            block[name_stat, "nbin_b"] = 1
        try:
            block[name_stat, "sample_a"] = conversion_dict[key][1].split('bins_')[1]
        except:
            block[name_stat, "sample_a"] = 'compton'
        try:
            block[name_stat, "sample_b"] = conversion_dict[key][2].split('bins_')[1]
        except:
            block[name_stat, "sample_b"] = 'compton'
        # this is not really important...
        if key in config['auto_only']:
            block[name_stat, "is_auto"] =  True
        else:
            block[name_stat, "is_auto"] = False
        # save theta and a few metadata
        #block[name_stat, "cl_section"] = cl_section
        try:
            block[name_stat, "theta"] = xcoord_array/(60./((2*math.pi)/360))
            del xcoord_array
            block.put_metadata(name_stat, "theta", "unit", "radians")
            block[name_stat, "sep_name"] = "theta"
            block[name_stat, "save_name"] = name_stat
            block[name_stat, "bin_avg"] = False
        except:
            pass


    if config['verbose']:
        print ('done conversion module')
    return 0

def cleanup(config):
    pass
