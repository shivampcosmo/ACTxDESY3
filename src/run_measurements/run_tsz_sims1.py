
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
ldir = '/global/cscratch1/sd/msyriac/data/depot/tilec/v1.2.0_20200324/'
fdir = 'map_joint_v1.2.0_sim_baseline_00_000' + str(0) + '_deep56/'
filename_mask = ldir + fdir + 'tilec_mask.fits'
imapm =  enmap.read_map(filename_mask)
decs,ras = imapm.posmap()
ra_m, dec_m, mask = np.array(rad2deg * ras.flatten()), np.array(rad2deg * decs.flatten()), np.array(imapm.flatten())
theta_mask_all, phi_mask_all = eq2ang(ra_m, dec_m)
nside = 512
ind_mask = hp.ang2pix(nside, theta_mask_all, phi_mask_all)
mask_input = np.copy(mask)
ind_masked = ind_mask[np.where(mask_input < 1e-4)[0]]
ind_unmasked = ind_mask[np.where(mask_input > 1e-4)[0]]


sdir = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/simsgty/'


cat_ACT = dill.load(open(sdir + 'cat_ACT_allbins.pk','rb'))



# # Populates a full destest yaml dict for each catalog selection based on the limited catalog input info provided in the common cats.yaml file
# def create_destest_yaml( params, name, cal_type, group, table, select_path ):
#     """
#     Creates the input dictionary structure from a passed dictionary rather than reading froma yaml file.
#     """
#     destest_dict = destest_dict_.copy()
#     destest_dict['load_cache'] = params['load_cache']
#     destest_dict['output'] = params['output']
#     destest_dict['name'] = name
#     destest_dict['filename'] = '/project/projectdirs/des/www/y3_cats/Y3_mastercat_03_31_20.h5'
#     destest_dict['param_file'] = params['param_file']
#     destest_dict['cal_type'] = cal_type
#     destest_dict['group'] = group
#     destest_dict['table'] = table
#     destest_dict['select_path'] = select_path
#     destest_dict['e'] = ['e_1','e_2']
#     destest_dict['Rg'] = ['R11','R22']
#     destest_dict['w'] = 'weight'
#     return destest_dict
# # Build selector (and calibrator) classes from destest for the catalog.
# def load_catalog(pipe_params, name, cal_type, group, table, select_path, inherit=None, return_calibrator=None):
#     """
#     Loads data access and calibration classes from destest for a given yaml setup file.
#     """
#     # Input yaml file defining catalog
#     params = create_destest_yaml(pipe_params, name, cal_type, group, table, select_path)
#     # Load destest source class to manage access to file
#     source = destest.H5Source(params)
#     # Load destest selector class to manage access to data in a structured way
#     if inherit is None:
#         sel = destest.Selector(params,source)
#     else:
#         sel = destest.Selector(params,source,inherit=inherit)
#     # Load destest calibrator class to manage calibration of the catalog
#     if return_calibrator is not None:
#         cal = return_calibrator(params,sel)
#         return sel, cal
#     else:
#         return sel
    

# # Beginning of the code! **********************************************************
# # Read yaml file that defines all the catalog selections used
# params = yaml.load(open('/global/cfs/cdirs/des/shivamp/xcorr/src/cats.yaml'))
# params['param_file'] = '/global/cfs/cdirs/des/shivamp/xcorr/src/cats.yaml'
# # Source catalog
# source_selector, source_calibrator = load_catalog(
#     params, 'mcal', 'mcal', params['source_group'], params['source_table'], params['source_path'], return_calibrator=destest.MetaCalib)
# # Gold catalog
# gold_selector = load_catalog(
#     params, 'gold', 'mcal', params['gold_group'], params['gold_table'], params['gold_path'], inherit=source_selector)
# # BPZ (or DNF) catalog, depending on paths in cats.yaml file (exchange bpz and dnf)
# pz_selector = load_catalog(
#     params, 'pz', 'mcal', params['pz_group'], params['pz_table'], params['pz_path'], inherit=source_selector)
# # I create a dictionary where I'll store ra,dec,e1,e2 and the maps ****
# cat_ACT = dict()
# # run over 4 tomographic bins
# # Get some source photo-z binning information, cut to range 0.1<z_mean<1.3                                                  
# for i in range(4):
#     print (i)
#     pzbin = pz_selector.get_col('bhat') # 5-tuple for metacal (un)sheared versions                                         
#     mask = [pzbin[j] == i for j in range(5)] # First tomographic bin                                                               
#     # Note that get_col() returns a tuple. If its a catalog like gold, it will have length 0, but for something like metacal, it will have length 5 (in the order of the table variable list passed in cats.yaml, i.e., 'unsheared', 'sheared_1p', 'sheared_1m', 'sheared_2p', 'sheared_2m')                                                               
#     # Note that get_col() applies the index mask specified by the 'path' variable in the cats.yaml file automatically.         # Get responses (c, which doesn't exist for our catalogs), and weights                                                      
#     R1,c,w = source_calibrator.calibrate('e_1', mask=mask,return_full_w=True) # Optionally pass an additional mask to use when calculating the selection response. The returned R1 is <Rg_1 + Rs_1>. To get an array of R's, use return_wRg=True to get [Rg_1+Rg_2]/2 for each object or return_wRgS=True to include the selection response. return_full=True returns the non-component-averaged version of the full response.
# #     print(len(w))
#     #R2,c,w = source_calibrator.calibrate('e_2', mask=mask)
#     #print(R2,c,w)
#     g1 = source_selector.get_col('e_1')[0][mask[0]]
#     g2 = source_selector.get_col('e_2')[0][mask[0]]
#     ra = gold_selector.get_col('ra')[0][mask[0]]
#     dec = gold_selector.get_col('dec')[0][mask[0]]
#     wa = source_calibrator.calibrate('e_1', mask=mask,weight_only=True) # Optionally pass an additional mask to use when calculating the selection response. The returned R1 is <Rg_1 + Rs_1>. To get an array of R's, use return_wRg=True to get [Rg_1+Rg_2]/2 for each object or return_wRgS=True to include the selection response. return_full=True returns the non-component-averaged version of the full response.
#     #g2=source_selector.get_col('e_2')[0]
#     R1,c,w = source_calibrator.calibrate('e_1',mask=mask) # Optionally pass an additional mask to use when calculating the selection response. The returned R1 is <Rg_1 + Rs_1>. To get an array of R's, use return_wRg=True to get [Rg_1+Rg_2]/2 for each object or return_wRgS=True to include the selection response. return_full=True returns the non-component-averaged version of the full response.
#     R2,c,w = source_calibrator.calibrate('e_2',mask=mask)
#     # these are the g1,g2 components for the i-th tomographic bin. I already subtracted the mean.
#     # remeber each galaxy comes with a weight 'w'
#     g1 =(g1 - np.mean(g1*wa)/np.mean(wa))/R1
#     g2 =(g2 - np.mean(g2*wa)/np.mean(wa))/R2
#     g1r,g2r= apply_random_rotation(g1,g2)  
    

#     theta_datapoint_all, phi_datapoint_all = eq2ang(ra, dec)
#     ind_datapoints = hp.ang2pix(nside, theta_datapoint_all, phi_datapoint_all)
#     int_ind = np.in1d(ind_datapoints, ind_unmasked)
#     ind = np.where(int_ind == True)[0]
    
#     print('orignal and new numbers : ' + str(len(ra)) + ', ' +  str(len(ind)))
    
#     cat_ACT[i] = [g1[ind],g2[ind], ra[ind],dec[ind], 1, w[ind]]
    






import dill
import os.path
# index_all = np.arange(560)
# index_all = np.array([0])
index_all = np.arange(100)
# map_$DATACOMB_test_sim_baseline_00_00$INDEX_$REGION
ldir = '/global/cscratch1/sd/msyriac/data/depot/tilec/v1.2.0_20200324/'

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

# kk_res = {}
for ji in range(len(index_all)):
    indv = index_all[ji]
    save_fname = sdir + 'gty_obj_sim_' + str(indv) + '_bin4y.pkl'
    if not os.path.isfile(save_fname):
        print('ind=' + str(index_all[ji]))
        if indv < 10:
            fdir = 'map_joint_v1.2.0_sim_baseline_00_000' + str(indv) + '_deep56/'
            fname = 'tilec_single_tile_deep56_comptony_map_joint_v1.2.0_sim_baseline_00_000' + str(indv) + '.fits'
        if indv > 10 and indv < 100:
            fdir = 'map_joint_v1.2.0_sim_baseline_00_00' + str(indv) + '_deep56/'
            fname = 'tilec_single_tile_deep56_comptony_map_joint_v1.2.0_sim_baseline_00_00' + str(indv) + '.fits'
        if indv > 100:
            fdir = 'map_joint_v1.2.0_sim_baseline_00_0' + str(indv) + '_deep56/'
            fname = 'tilec_single_tile_deep56_comptony_map_joint_v1.2.0_sim_baseline_00_0' + str(indv) + '.fits'
        filename = ldir + fdir + fname
        imapy = enmap.read_map(filename)
        decs,ras = imapy.posmap()
        ra_y, dec_y, ymap = np.array(rad2deg * ras.flatten()), np.array(rad2deg * decs.flatten()), np.array(imapy.flatten())

        theta_datapoint_all, phi_datapoint_all = eq2ang(ra_y, dec_y)
        ind_datapoints = hp.ang2pix(nside, theta_datapoint_all, phi_datapoint_all)
        int_ind = np.in1d(ind_datapoints, ind_unmasked)
        selection_f = np.where(int_ind == True)[0]

        for jb in [3]:
            cat_a = treecorr.Catalog(ra=cat_ACT[jb][2], dec=cat_ACT[jb][3], g1=cat_ACT[jb][0], g2=cat_ACT[jb][1],
                                     w=cat_ACT[jb][5],ra_units='deg', dec_units='deg') 
            cat_b = treecorr.Catalog(ra=ra_y[selection_f], dec=dec_y[selection_f], ra_units='deg', dec_units='deg',
                                                 k=ymap[selection_f])
            kk = treecorr.KGCorrelation(conf)
            kk.process(cat_b, cat_a)
            print(kk.xi)
            dill.dump(kk,open(save_fname,'wb'))






