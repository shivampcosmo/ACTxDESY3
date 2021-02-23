import sys, os
from cosmosis.datablock import names, option_section, BlockError
import numpy as np
import dill
def setup(options):
    config = {}
    return config

def execute(block, config):

    suffix = ''
    sec_names = {
        "shear_shear": "shear_cl" + suffix,
        "shear_shear_bb": "shear_cl_bb" + suffix,
        "shear_shear_gg": "shear_cl_gg" + suffix,
        "galaxy_shear": "galaxy_shear_cl" + suffix,
        "shear_intrinsic": "shear_cl_gi" + suffix,
        "galaxy_intrinsic": "galaxy_intrinsic_cl"  + suffix,
        "intrinsic_intrinsic": "shear_cl_ii"  + suffix,
        "intrinsic_intrinsic_bb": "shear_cl_ii_bb"  + suffix,
        "parameters": "intrinsic_alignment_parameters" + suffix,
        "shear_cmbkappa": "shear_cmbkappa_cl" + suffix,
        "intrinsic_cmbkappa": "intrinsic_cmbkappa_cl" + suffix,
    }   

    shear_shear = sec_names['shear_shear']
    shear_shear_bb = sec_names['shear_shear_bb']
    shear_shear_gg = sec_names['shear_shear_gg']
    galaxy_shear = sec_names['galaxy_shear']
    galaxy_intrinsic = sec_names['galaxy_intrinsic']
    shear_intrinsic = sec_names['shear_intrinsic']
    parameters = sec_names['parameters']
    intrinsic_intrinsic = sec_names['intrinsic_intrinsic']
    intrinsic_intrinsic_bb = sec_names['intrinsic_intrinsic_bb']
    shear_cmbkappa = sec_names['shear_cmbkappa']
    intrinsic_cmbkappa = sec_names['intrinsic_cmbkappa']
    out_dict = {}

    A = [1 for i in range(4)]
    out_dict['ell'] = block[shear_shear, 'ell']
    for i in range(4):
        for j in range(i + 1):
            bin_ij = 'bin_{0}_{1}'.format(i + 1, j + 1)
            bin_ji = 'bin_{1}_{0}'.format(i + 1, j + 1)
            out_dict['shear_shear_' + str(i+1) + '_' + str(j+1)] = block[shear_shear_gg, bin_ij]
            out_dict['shear_intrinsic_' + str(i+1) + '_' + str(j+1)] = A[j] * block[shear_intrinsic, bin_ij] + A[i] * block[shear_intrinsic, bin_ji]
            out_dict['intrinsic_intrinsic_' + str(j+1) + '_' + str(i+1)] = A[i] * A[j] * block[intrinsic_intrinsic, bin_ij]  # II


    savefname = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/save_block_testIA.pk'
    dill.dump(out_dict,open(savefname,'wb'))
    import ipdb; ipdb.set_trace() # BREAKPOINT
    
    return 0

def cleanup(config):
    pass
