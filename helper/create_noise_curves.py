#%%

import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
import pandas as pd
import healpy as hp
import sys, os
import pdb
import scipy as sp
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

import dill
import pickle as pk



#%%

noise_Cl_filename = '../forecasts/forecast_data/S4/S4_190604d_2LAT_T_default_noisecurves_deproj0_SENS0_mask_16000_ell_TT_yy.txt'
noise_yy_Cl_file = np.loadtxt(noise_Cl_filename)
l_noise_yy_file, Cl_noise_yy_file = noise_yy_Cl_file[:, 0], noise_yy_Cl_file[:, 2]

l_cl = l_noise_yy_file

#%%

theta_fwhm = 10.0 * (np.pi/180.0) * (1./60.0)
sigb = theta_fwhm/(np.sqrt(8.*np.log(2)))
B_l = np.exp(-1.*l_cl * (l_cl + 1) * (sigb**2)/2.)



Cl_noise_yy_file_S4_deconv = Cl_noise_yy_file/(B_l**2)


fig, ax_all = plt.subplots(1, 1, figsize=(8, 6), sharey = True)
fig.subplots_adjust(wspace = 0.)
ax = ax_all

ax.plot(l_cl, l_cl * (l_cl + 1.) * Cl_noise_yy_file / (2 * np.pi), color='blue', marker='',linestyle='-', label=r'S4 Noise')
ax.plot(l_cl, l_cl * (l_cl + 1.) * Cl_noise_yy_file_S4_deconv / (2 * np.pi), color='red', marker='',linestyle='-', label=r'S4 Noise Beamed')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel(r'$\ell$', size = 22)
ax.set_ylabel(r'$\ell \ (\ell + 1) \ N_\ell/ 2 \pi$', size=22)
ax.legend(fontsize=15, frameon=False, loc='upper left')
ax.set_ylim(1e-16,1e-1)
# ax.set_xlim(1,7000)
ax.tick_params(axis='y', which='major', labelsize=15)
ax.tick_params(axis='y', which='minor', labelsize=15)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='x', which='minor', labelsize=15)
fig.tight_layout()
fig.savefig('../forecasts/plots/Cl_S4_noise_beamed_comp.png')


np.savetxt('../forecasts/forecast_data/S4/S4_190604d_2LAT_T_default_noisecurves_deproj0_SENS0_mask_16000_ell_TT_yy_10arcminbeam.txt',np.array([l_cl, np.ones(l_cl.shape),Cl_noise_yy_file_S4_deconv]).T,header='ell  Cl_yy_noise')


