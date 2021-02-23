import sys, platform, os
# import sys, os
sys.path.insert(0,'/global/u1/s/spandey/kmeans_radec/')
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pdb
import healpy as hp
from astropy.io import fits
from kmeans_radec import KMeans, kmeans_sample
import time
import math
from scipy import interpolate
import treecorr
import pickle as pk
import configparser
from pixell import enmap
from pixell import reproject
import ast
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
sys.path.insert(0,'/global/u1/s/spandey/kmeans_radec/')
import kmeans_radec
import h5py as h5
import argparse

import healpy as hp
import os
root_planck = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/planck_data'
# data_dir = '/global/cfs/cdirs/des/xuod/DES_ACT_gxy/data'
r = hp.Rotator(coord=['G','C'])
# mask = fits.open(os.path.join(root_planck, 'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits'))[1].data['M1']
# mask_rotated = r.rotate_map_alms(mask)
# hp.write_map(os.path.join(data_dir, 'COM_CompMap_Compton-SZMap-masks_2048_R2.01_celestial.fits'), mask_rotated, overwrite=True)
# print("Mask rotated and saved")
print('rotating CIB 857')
ymap = (hp.read_map(os.path.join(root_planck, 'COM_CompMap_CIB-GNILC-F857_2048_R2.00.fits')))
ymap_rotated = r.rotate_map_alms(ymap)
hp.write_map('/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/planck_data/GNILC_CIB857_rotated_celestial_DES_coords.fits', ymap_rotated, overwrite=True) 

print('rotating CIB 545')
ymap = (hp.read_map(os.path.join(root_planck, 'COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits')))
ymap_rotated = r.rotate_map_alms(ymap)
hp.write_map('/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/planck_data/GNILC_CIB545_rotated_celestial_DES_coords.fits', ymap_rotated, overwrite=True) 

print('rotating Temp 545')
ymap = (hp.read_map(os.path.join(root_planck, 'HFI_SkyMap_545-field-Int_2048_R3.00_full.fits')))
ymap_rotated = r.rotate_map_alms(ymap)
hp.write_map('/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/planck_data/SkyMap_545_rotated_celestial_DES_coords.fits', ymap_rotated, overwrite=True) 

print('rotating Temp 857')
ymap = (hp.read_map(os.path.join(root_planck, 'HFI_SkyMap_857-field-Int_2048_R3.00_full.fits')))
ymap_rotated = r.rotate_map_alms(ymap)
hp.write_map('/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/planck_data/SkyMap_857_rotated_celestial_DES_coords.fits', ymap_rotated, overwrite=True) 

