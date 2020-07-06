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

pi = np.pi


class HOD:
    """
    Sets up the HOD class for the galaxies.
    """

    def __init__(self, hod_params, other_params):
        self.hod_params = hod_params
        self.hod_type = hod_params['hod_type']
        self.z_array = other_params['z_array']

    # Average number of central galaxies
    def get_Nc(self, M_val):
        if self.hod_type == 'Halos':
            Ncm = 0.5 * (np.sign((M_val - 10 ** self.hod_params['logMmin'])) + 1.)
        elif self.hod_type in ['DESI', '2MRS']:
            erfval = sp.special.erf((np.log10(M_val) - self.hod_params['logMmin']) / self.hod_params['sig_logM'])
            Ncm = 0.5 * (1 + erfval)
        elif self.hod_type == 'DES_MICE':
            Ncm =  0.5 * self.hod_params['fcen'] * (1. + sp.special.erf((np.log10(M_val) - self.hod_params['logMmin']) / self.hod_params['sig_logM']))

            # erfval = sp.special.erf((np.log10(M_val) - self.hod_params['logMmin']) / self.hod_params['sig_logM'])
            # Ncm = self.hod_params['fmaxcen'] * (
            #         1.0 - (1.0 - self.hod_params['fmincen'] / self.hod_params['fmaxcen']) / (1.0 + 10 ** (
            #         (2.0 / self.hod_params['fcen_k']) * (
            #         np.log10(M_val) - self.hod_params['log_mdrop'])))) * 0.5 * (1 + erfval)
        elif self.hod_type == 'DES_GGL':
            erfval = sp.special.erf((np.log10(M_val) - self.hod_params['logMmin']) / self.hod_params['sig_logM'])
            Ncm = self.hod_params['fcen'] * 0.5 * (1 + erfval)
        elif self.hod_type == 'EVOLVE_HOD':
            Ncm_interp = dill.load(open('/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/mice_maglim_hod_zM_interp_zhres.pk','rb'))['fcen_interp']
            Ncm = np.zeros_like(M_val)
            for j in range(len(self.z_array)):
                Ncm[j,:] = np.exp(Ncm_interp((self.z_array[j]), np.log(M_val[j,:]),grid=False))
        else:
            print('give correct HOD type')
            sys.exit(1)
        return Ncm

    # Average number of satellite galaxies
    def get_Ns(self, M_val):
        if self.hod_type == 'Halos':
            Nsm = np.zeros(M_val.shape)
        elif self.hod_type in ['DESI', '2MRS']:
            M0 = 10 ** (self.hod_params['logM0'])
            M1 = 10 ** (self.hod_params['logM1'])
            val = 0.5 * (np.sign((M_val - M0)) + 1.) * ((M_val - M0) / M1)
            Nsm = np.power(val, self.hod_params['alpha_g'])
        elif self.hod_type == 'DES_MICE':
            M1 = 10 ** (self.hod_params['logM1'])
            Nsm = ((M_val / M1) ** self.hod_params['alpha_g'])
            # Nsm = (M_val / (10 ** (self.hod_params['logM1']))) ** self.hod_params['alpha_g'])
            # erfval = sp.special.erf((np.log10(M_val) - self.hod_params['logMmin']) / self.hod_params['sig_logM'])
            # Ncerf = 0.5 * (1 + erfval)
            # Ncm = get_Nc(self, M_val)
            # M1 = 10 ** (self.hod_params['logM1'])
            # Nsm = (Ncerf / Ncm) * ((M_val / M1) ** self.hod_params['alpha_g'])
        elif self.hod_type == 'DES_GGL':
            M1 = 10 ** (self.hod_params['logM1'])
            Nsm = ((M_val / M1) ** self.hod_params['alpha_g'])
        elif self.hod_type == 'EVOLVE_HOD':
            Nsm_interp = dill.load(open('/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/mice_maglim_hod_zM_interp_zhres.pk', 'rb'))['fsat_interp']
            Nsm = np.zeros_like(M_val)
            for j in range(len(self.z_array)):
                Nsm[j, :] = np.exp(Nsm_interp((self.z_array[j]), np.log(M_val[j, :]), grid=False))
        else:
            print('give correct HOD type')
            sys.exit(1)
        return Nsm

    # total number of galaxies = (Ncm*(1+Nsm))
    def get_Ntotal(self, M_val):
        Ncm = self.get_Nc(M_val)
        Nsm = self.get_Ns(M_val)
        ntm = Ncm * (1 + Nsm)
        return ntm
