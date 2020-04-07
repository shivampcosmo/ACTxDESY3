from cosmosis.datablock import names, option_section
from cosmosis.datablock import option_section, names

import pickle

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')
    
    
import numpy as np

def setup(options):
    config=dict()
    config['mead_v'] = options.get_string(option_section, "mead_v", default = "new")
    return config

def execute(block, config):
    print ('MEAD module checks.')
    
    # save the outputs of old and new mead modules so one can plot them--
    mead = dict()
    
    mead['um_1']  = block[names.matter_power_nl,'um_1']
    mead['um_5']  = block[names.matter_power_nl,'um_5']
    mead['um_10'] = block[names.matter_power_nl,'um_10']
    mead['um_15'] = block[names.matter_power_nl,'um_15']
    mead['um_20'] = block[names.matter_power_nl,'um_20']
    mead['um_25'] = block[names.matter_power_nl,'um_25']
    mead['um_30'] = block[names.matter_power_nl,'um_30']
    mead['um_35'] = block[names.matter_power_nl,'um_35']
    mead['um_40'] = block[names.matter_power_nl,'um_40']
    mead['um_45'] = block[names.matter_power_nl,'um_45']
    mead['um_50'] = block[names.matter_power_nl,'um_50']
    mead['um_55'] = block[names.matter_power_nl,'um_55']
    mead['um_60'] = block[names.matter_power_nl,'um_60']
    mead['um_65'] = block[names.matter_power_nl,'um_65']
    mead['um_70'] = block[names.matter_power_nl,'um_70']
    mead['um_75'] = block[names.matter_power_nl,'um_75']
    mead['um_80'] = block[names.matter_power_nl,'um_80']
    mead['um_85'] = block[names.matter_power_nl,'um_85']
    mead['um_90'] = block[names.matter_power_nl,'um_90']
    mead['um_95'] = block[names.matter_power_nl,'um_95']
    mead['um_100']= block[names.matter_power_nl,'um_100']


    try:
        mead['ind_lut']  = block[names.matter_power_nl,'ind_lut']
    except:
        pass
    try:
        mead['bt_out']= block[names.matter_power_nl,'bt_out']
    except:
        pass
    try:
        
        mead['g_1']  = block[names.matter_power_nl,'g_1']
        mead['g_5']  = block[names.matter_power_nl,'g_5']
        mead['g_10'] = block[names.matter_power_nl,'g_10']
        mead['g_15'] = block[names.matter_power_nl,'g_15']
        mead['g_20'] = block[names.matter_power_nl,'g_20']
        mead['g_25'] = block[names.matter_power_nl,'g_25']
        mead['g_30'] = block[names.matter_power_nl,'g_30']
        mead['g_35'] = block[names.matter_power_nl,'g_35']
        mead['g_40'] = block[names.matter_power_nl,'g_40']
        mead['g_45'] = block[names.matter_power_nl,'g_45']
        mead['g_50'] = block[names.matter_power_nl,'g_50']
        mead['g_55'] = block[names.matter_power_nl,'g_55']
        mead['g_60'] = block[names.matter_power_nl,'g_60']
        mead['g_65'] = block[names.matter_power_nl,'g_65']
        mead['g_70'] = block[names.matter_power_nl,'g_70']
        mead['g_75'] = block[names.matter_power_nl,'g_75']
        mead['g_80'] = block[names.matter_power_nl,'g_80']
        mead['g_85'] = block[names.matter_power_nl,'g_85']
        mead['g_90'] = block[names.matter_power_nl,'g_90']
        mead['g_95'] = block[names.matter_power_nl,'g_95']
        mead['g_100']= block[names.matter_power_nl,'g_100']

    except:
        pass
    
    mead['mass_h']= block[names.matter_power_nl, "mass_h_um"]
    
    if config['mead_v'] =='old':
        save_obj('tests_mead_old', mead)
    else:
         save_obj('tests_mead_new', mead)
    return 0
def cleanup(config):
    pass
