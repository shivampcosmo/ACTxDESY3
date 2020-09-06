
'''
 Load data. Input files defined in yaml files (destest_bpz.yaml, destest_metacal.yaml,destest_gold.yaml)
 It requires destest : 
 https://github.com/des-science/destest
 
 catalogs (to be specified in the yaml files):
 
 Latest version '/g
 lobal/cscratch1/sd/troxel/cats_des_y3/Y3_mastercat_v2_6_20_18.h5'
 Latest subsampled version (for quick tests): '/global/cscratch1/sd/troxel/cats_des_y3/Y3_mastercat_v2_6_20_18_subsampled.h5'
 
 I didn't manage to run the destest Calibrator. This means that if you directly load R1 and R2 from the catalog,
 they are only the respones of the sample and neglect the response of the selection (which is usually few %).
 The solution I got was to load the columns without selection (uncut=True)
 and apply the response and selection by myself.
'''


import dill
import pyfits as pf
import healpy as hp
import numpy as np

import h5py
import healpy as hp
import pymaster as nmt
from pixell import enmap
from pixell import reproject
import matplotlib
import matplotlib.pyplot as pl
Color = ['k', '#000075', '#a9a9a9','#9A6324', '#808000','#aaffc3', '#fffac8'  ,'#800000', '#ffd8b1',]

font = {'size'   : 18}
matplotlib.rc('font', **font)
# # Latex stuff
pl.rc('text', usetex=False)
pl.rc('font', family='serif')



def IndexToDeclRa(index, nside,ge=False,nest=False):
    theta,phi=hp.pixelfunc.pix2ang(nside ,index,nest)
    if ge:
        r = hp.rotator.Rotator(coord=['G','E'])
        theta, phi = r(theta,phi) 
    return -np.degrees(theta-np.pi/2.),np.degrees(phi)



from astropy.coordinates import SkyCoord
from astropy import units as u
def data_coord_cov(ra_in,dec_in, icrs2gal=False, gal2icrs=False):
    if icrs2gal:
        c_icrs = SkyCoord(ra=ra_in * u.degree, dec=dec_in * u.degree, frame='icrs')
        c_gal = c_icrs.galactic
        l_out, b_out = (c_gal.l).value, (c_gal.b).value
        return l_out, b_out
    if gal2icrs:
        c_gal = SkyCoord(l=ra_in * u.degree, b=dec_in * u.degree, frame='galactic')
        c_icrs = c_gal.icrs
        ra_out, dec_out = (c_icrs.ra).value, (c_icrs.dec).value
        return ra_out, dec_out
    
    
def to_ra_dec(theta,phi):
    ra = phi*180./np.pi
    dec = 90. - theta*180./np.pi
    return ra, dec

def IndexToDeclRa(index, nside,ge=False,nest=False):
    theta,phi=hp.pixelfunc.pix2ang(nside ,index,nest)
    if ge:
        r = hp.rotator.Rotator(coord=['G','E'])
        theta, phi = r(theta,phi) 
    return -np.degrees(theta-np.pi/2.),np.degrees(phi)



    ipix = np.arange(0,npix)
    theta, phi = hp.pixelfunc.pix2ang(nside, ipix)
    ra, dec = to_ra_dec(theta,phi)
    kappa = map
    return ra,dec,kappa

import sys
#sys.path.insert(0, '/global/homes/m/mgatti/Mass_Mapping/systematic_checks/')
from routines import *
import numpy as np

# ******************************************************************
#                              INPUT
# ******************************************************************
nside_mask = 1024  
nside_fast_corr = 1024
nside_cmb = 2048

# output folders ***************
fold1= '/global/cscratch1/sd/mgatti/Cosmic_shear/output_tsz/'  
name_folder_x = '/global/cscratch1/sd/mgatti/Cosmic_shear/output_tsz/rerun_mastercat_4_20_newACT/'

if not os.path.exists(fold1):
    os.mkdir(fold1)
if not os.path.exists(name_folder_x):
    os.mkdir(name_folder_x)

print ("done")



bins = ['0.43_0.63','0.63_0.9','0.9_1.3','0.2_1.3']
nside = 1024
bins_min = [0.2,0.43,0.63,0.9]
bins_max = [0.43,0.63,0.9,1.3]

import sys, os
sys.path.insert(0,  '/global/cscratch1/sd/mgatti/Mass_Mapping/TSZ_ACT/cosmosis//ACTxDESY3/src/cosmosis_code')
os.environ['COSMOSIS_SRC_DIR'] = '/global/cscratch1/sd/mgatti/Mass_Mapping/TSZ_ACT/cosmosis/'



import numpy as np
import copy
import pdb
import ast
import scipy as sp
from scipy import interpolate





import numpy as np
import pyfits as pf
import healpy as hp
import h5py as h5
import os
import sys
import yaml
sys.path.insert(0,  '/global/cfs/cdirs/des/shivamp/destest')
import destest
import treecorr

rad2deg = 180./np.pi


def ang2eq(theta, phi):
    ra = phi * 180. / np.pi
    dec = 90. - theta * 180. / np.pi
    return ra, dec


def eq2ang(ra, dec):
    phi = ra * np.pi / 180.
    theta = (np.pi / 2.) - dec * (np.pi / 180.)
    return theta, phi





# this routine is used to generate random rotations of the shapes - useful to estimate shape noise!
def apply_random_rotation(e1_in, e2_in):
    np.random.seed() # CRITICAL in multiple processes !
    rot_angle = np.random.rand(len(e1_in))*2*np.pi #no need for 2?
    cos = np.cos(rot_angle)
    sin = np.sin(rot_angle)
    e1_out = + e1_in * cos + e2_in * sin
    e2_out = - e1_in * sin + e2_in * cos
    return e1_out, e2_out
# This routine converts healpy pixel index to dec ra coordinates.
def IndexToDeclRa(index, nside,nest= False):
    theta,phi=hp.pixelfunc.pix2ang(nside ,index,nest=nest)
    return -np.degrees(theta-np.pi/2.),np.degrees(phi)
# basic dict props - needed for destest
destest_dict_ = {
    'output_exists' : True,
    'use_mpi'       : False,
    'source'        : 'hdf5',
    'dg'            : 0.01
    }

rad2deg = 180./np.pi
rad2deg = 180./np.pi
ldir = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/planck_data/'
f1 = 'HFI_Mask_PointSrc_2048_R2.00.fits'
f2 = 'LFI_Mask_PointSrc_2048_R2.00.fits'

from astropy.io import fits
df1 = fits.open(ldir + f1)[1].data
maskhfi = hp.reorder(df1['F100'] * df1['F143'] * df1['F217'] * df1['F353'] * df1['F545'] * df1['F857'],n2r=True)

from astropy.io import fits
df2 = fits.open(ldir + f2)
m30 = hp.read_map(ldir + f2,hdu='LFI_030_PsMask')
m44 = hp.read_map(ldir + f2,hdu='LFI_044_PsMask')
m70 = hp.read_map(ldir + f2,hdu='LFI_070_PsMask')
masklfi = m30 * m44 * m70

mask_ps_pl = masklfi * maskhfi

mask_gal = fits.open(ldir + 'HFI_Mask_GalPlane-apo2_2048_R2.00.fits')[1].data['GAL060']
mask_gal = hp.reorder(mask_gal,n2r=True)

def ind2eq(index):
    theta,phi=hp.pixelfunc.pix2ang(2048,index)
    ra = phi*180./np.pi
    dec = 90. - theta*180./np.pi
    return ra, dec



mask = mask_gal
npix = hp.nside2npix(2048)
#conversion to normal ra,dec
indall = np.arange(0,npix)
yLall, yBall = ind2eq(indall)
rayall, decyall  = data_coord_cov(yLall, yBall, gal2icrs=True)
ra_m, dec_m = rayall, decyall
theta_mask_all, phi_mask_all = eq2ang(ra_m, dec_m)
nside = 2048
ind_mask = hp.ang2pix(nside, theta_mask_all, phi_mask_all)
mask_input = np.copy(mask)
ind_masked = ind_mask[np.where(mask_input < 1e-4)[0]]
ind_unmasked = ind_mask[np.where(mask_input > 1e-4)[0]]




sdir = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/simsgty/'


cat_ACT = dill.load(open(sdir + 'cat_Plgal_allbins.pk','rb'))


import dill
import os.path

Nbins = 20
min_theta = 2.5/60.
max_theta = 250./60.
number_of_cores = 64
bin_slope = 0.05
conf = {'nbins': Nbins,
            'min_sep': min_theta,
            'max_sep': max_theta,
            'sep_units': 'degrees',
            'bin_slop': bin_slope,
            'nodes': number_of_cores  # parameter for treecorr
            }
sdir = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/simsgty/'

fymap = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/planck_data/COM_CompMap_Compton-SZMap-nilc-ymaps_2048_R2.00.fits'
ymap = hp.read_map(fymap)
ra_y, dec_y = np.copy(rayall), np.copy(decyall)

theta_datapoint_all, phi_datapoint_all = eq2ang(ra_y, dec_y)
ind_datapoints = hp.ang2pix(nside, theta_datapoint_all, phi_datapoint_all)
int_ind = np.in1d(ind_datapoints, ind_unmasked)
selection_f = np.where(int_ind == True)[0]

for jb in range(4):
    print('doing bin' + str(jb + 1))
    save_fname = sdir + 'planck_des_gty_sbin_' + str(jb+1) + '_only_gal_mask.pk'
    cat_a = treecorr.Catalog(ra=cat_ACT[jb][2], dec=cat_ACT[jb][3], g1=cat_ACT[jb][0], g2=cat_ACT[jb][1],
                             w=cat_ACT[jb][5],ra_units='deg', dec_units='deg') 
    cat_b = treecorr.Catalog(ra=ra_y[selection_f], dec=dec_y[selection_f], ra_units='deg', dec_units='deg',
                                         k=ymap[selection_f])
    kk = treecorr.KGCorrelation(conf)
    kk.process(cat_b, cat_a)
    print(kk.xi)
    dill.dump(kk,open(save_fname,'wb'))






