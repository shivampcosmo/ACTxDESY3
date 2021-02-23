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
import ast
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
sys.path.insert(0,'/global/u1/s/spandey/kmeans_radec/')
import kmeans_radec
import h5py as h5
import argparse
sys.path.insert(0,'/global/cfs/cdirs/des/shivamp/cosmosis/y3kp-bias-model/3d_stats/process_measure_data/')
import process_cats_class as pcc
cosmo_params_dict = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
from astropy import constants as const
import scipy.signal as spsg
import scipy.integrate as integrate

def ang2eq(theta, phi):
    ra = phi * 180. / np.pi
    dec = 90. - theta * 180. / np.pi
    return ra, dec


def eq2ang(ra, dec):
    phi = ra * np.pi / 180.
    theta = (np.pi / 2.) - dec * (np.pi / 180.)
    return theta, phi

def get_zmean(zcent,delz,nz_bin):
    prob_zcent = nz_bin
    zmean = (np.sum(prob_zcent*zcent*delz))/(np.sum(prob_zcent*delz))
    return zmean


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat', required=True, type=str, help='Cat type')
    parser.add_argument('--mask', required=True, type=str, help='mask type')
    parser.add_argument('--do_gg', type=bool, default=True)
    parser.add_argument('--nside', default=4096)
    parser.add_argument('--logMmin', default=13.0)
    parser.add_argument('--logMmax', default=13.5)
    parser.add_argument('--beam', default=0)
    args_all = parser.parse_args()
    return args_all

if __name__ == "__main__":
    args = parse_arguments()

    # redmagic, maglim
    cat_tocorr = args.cat
    # cat_tocorr = 'redmagic'
    # None, cib, cmb
    mask_type = args.mask
    do_gg = args.do_gg
    logMmin = args.logMmin
    logMmax = args.logMmax
    beam = args.beam
    nside = args.nside
#     do_gy = args.do_gy
    
    do_jk = True
    do_randy_sub = True
    if mask_type == 'act':
        njk = 500

    if mask_type == 'oct':
        njk = 1000

    if mask_type == 'diff':
        njk = 300

    njk_radec = njk
    njk_z = 1
    # min_r = 0.1
    # max_r = 80.
    nthreads = 64
    bin_slop = 0.0

    minrad = 0.5
    maxrad = 40.0
    nrad = 12

    save_dir = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/'
    #     save_filename_jk_obj = 'jkobj_MICE_' + '_' + '_njk_' + str(njk) + '.pk'
    save_filename_jk_obj = 'jkobj_MICE_' + '_' + '_njk_' + str(njk) + '_mask_' + str(mask_type) + '_v22Feb21.pk'

    ydir = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/'
    if beam == 0:
        true_y_file = ydir + 'ymap_mice_splitCOMBINED_NM100_Nz100_nside' + str(nside) + '_highacc_v01.fits'
    else:
        true_y_file = ydir + 'ymap_mice_splitCOMBINED_NM50_Nz16_nside' + str(nside) + '_v01_beam_' + str(beam) + '_arcmin.fits'

    print('opening ymap and mask')
    ymap_truth = hp.read_map(true_y_file)
    nside = hp.npix2nside(len(ymap_truth))
    nside_ymap = nside
    mask_final = []
    npix_ymap = len(ymap_truth)
    nside_ymap = hp.npix2nside(npix_ymap)
    pix_theta, pix_phi = hp.pix2ang(nside_ymap, np.arange(npix_ymap))
    pix_ra, pix_dec = ang2eq(pix_theta, pix_phi)    

    if mask_type == 'act':
        if nside == 4096:
            filename_mask = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/ACT_D56_mask_inMICE.fits'
        if nside == 2048:
            filename_mask = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/ACT_D56_mask_ns2048_inMICE.fits'

    if mask_type == 'oct':
        if nside == 4096:
            filename_mask = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/octant_mask_inMICE.fits'
        if nside == 2048:
            filename_mask = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/octant_mask_ns2048_inMICE.fits'

    if mask_type == 'diff':
        if nside == 2048:
            filename_mask = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/octant_minus_ACT_D56_mask_inMICE.fits'

    gal_mask_input_orig = hp.read_map(filename_mask)
    mask_input = np.copy(gal_mask_input_orig)
    ind_masked = np.where(mask_input < 1e-4)[0]
    # theta_datapoint_all, phi_datapoint_all = eq2ang(datapoint_ra_all, datapoint_dec_all)
    ind_y = hp.ang2pix(nside, pix_theta, pix_phi)
    int_ind = np.in1d(ind_y, ind_masked)
    selection_mask = np.where(int_ind == False)[0] 
    pix_ra, pix_dec, ymap_truth =  pix_ra[selection_mask], pix_dec[selection_mask], ymap_truth[selection_mask]



    other_params_dict = {}
    if cat_tocorr == 'maglim':
        zmin_bins = np.array([0.20, 0.40, 0.55, 0.70, 0.85, 0.95])
        zmax_bins = np.array([0.40, 0.55, 0.70, 0.85, 0.95, 1.05 ])
        other_params_dict['bin_n_array'] = [1,2,3,4,5,6]
        other_params_dict['bin_array'] = ['bin1','bin2','bin3','bin4','bin5','bin6']

    if cat_tocorr == 'redmagic':
        zmin_bins = np.array([0.15, 0.35, 0.5, 0.65, 0.8])
        zmax_bins = np.array([0.35, 0.5, 0.65, 0.8, 0.9])
        other_params_dict['bin_n_array'] = [1,2,3,4,5]
        other_params_dict['bin_array'] = ['bin1','bin2','bin3','bin4','bin5']

    if cat_tocorr == 'halos':
        zmin_bins = np.array([0.20, 0.40, 0.55, 0.70, 0.85, 0.95])
        zmax_bins = np.array([0.40, 0.55, 0.70, 0.85, 0.95, 1.05 ])
        other_params_dict['bin_n_array'] = [1,2,3,4,5,6]
        other_params_dict['bin_array'] = ['bin1','bin2','bin3','bin4','bin5','bin6']


    other_params_dict['zmin_bins'] = zmin_bins
    other_params_dict['zmax_bins'] = zmax_bins
    other_params_dict['njk_radec'] = njk_radec
    other_params_dict['njk_z'] = njk_z

    gnf = pcc.general_funcs(cosmo_params_dict)
    z_array = np.linspace(0, 1.5, 10000)
    chi_array = np.zeros(len(z_array))
    for j in range(len(z_array)):
        chi_array[j] = gnf.get_Dcom(z_array[j])
    other_params_dict['chi_interp'] = interpolate.interp1d(z_array, chi_array)

    chi_array = np.linspace(0, 4000, 50000)
    z_array = np.zeros(len(chi_array))
    for j in range(len(z_array)):
        z_array[j] = gnf.get_z_from_chi(chi_array[j])
    other_params_dict['z_interp'] = interpolate.interp1d(chi_array, z_array)

    z_min, z_max = 0.1, 1.1
    nzbins_total = 1000
    zarray_all = np.linspace(z_min, z_max, nzbins_total)
    zarray_edges = (zarray_all[1:] + zarray_all[:-1]) / 2.
    zarray = zarray_all[1:-1]
    chi_array_r = gnf.get_Dcom_array(zarray)
    dchi_dz_array_r = (const.c.to(u.km / u.s)).value / (gnf.get_Hz(zarray))
    chi_max = gnf.get_Dcom_array([z_max])[0]
    chi_min = gnf.get_Dcom_array([z_min])[0]
    VT = (4 * np.pi / 3) * (chi_max ** 3 - chi_min ** 3)
    dndz = (4 * np.pi) * (chi_array_r ** 2) * dchi_dz_array_r / VT

    print('opening galaxy catalog')
    if cat_tocorr == 'maglim':
        fname = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/mice_maglim_data.fits'
        df = fits.open(fname)
        # datapoint_z_all = df[1].data['z_dnf_mean_sof']
        datapoint_z_all = df[1].data['z_cgal']
        datapoint_ra_all = df[1].data['ra_gal']
        datapoint_dec_all = df[1].data['dec_gal']

    if cat_tocorr == 'redmagic':
        fname = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/mice_redmagic_hdens_wmag.fits'
        df = fits.open(fname)
        datapoint_z_all = df[1].data['zredmagic']
        datapoint_ra_all = df[1].data['ra']
        datapoint_dec_all = df[1].data['dec']

    if cat_tocorr == 'halos':
        fname = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/mice_all_halos.fits'
        df = fits.open(fname)
        datapoint_z_all = df[1].data['z_cgal']
        datapoint_ra_all = df[1].data['ra_gal']
        datapoint_dec_all = df[1].data['dec_gal']
        datapoint_M_all = df[1].data['lmhalo']
        ind_sel = np.where((datapoint_M_all > logMmin) & (datapoint_M_all < logMmax))[0]
        datapoint_ra_all, datapoint_dec_all, datapoint_z_all = datapoint_ra_all[ind_sel], datapoint_dec_all[ind_sel], datapoint_z_all[ind_sel]
    del df

    print('making randoms catalog')
    CF_datapoint_all = pcc.Catalog_funcs(datapoint_ra_all, datapoint_dec_all, datapoint_z_all ,cosmo_params_dict,other_params_dict)
    nz_unnorm, z_edge = np.histogram(datapoint_z_all, zarray_edges)
    nz_unnorm_smooth =  spsg.savgol_filter(nz_unnorm, 21, 5)
    nz_normed = nz_unnorm/(integrate.simps(nz_unnorm,zarray))
    nz_normed_smooth = nz_unnorm_smooth/(integrate.simps(nz_unnorm_smooth,zarray))
    rand_ra_all, rand_dec_all, rand_z_all = CF_datapoint_all.create_random_cat_uniform_esutil(zarray=zarray, nz_normed=nz_normed_smooth,nrand_fac=10, ra_min=0, ra_max=90, dec_min=0, dec_max=90)
    del CF_datapoint_all

    mask_input = np.copy(gal_mask_input_orig)
    ind_masked = np.where(mask_input < 1e-4)[0]
    theta_datapoint_all, phi_datapoint_all = eq2ang(datapoint_ra_all, datapoint_dec_all)
    ind_datapoints = hp.ang2pix(nside, theta_datapoint_all, phi_datapoint_all)
    int_ind = np.in1d(ind_datapoints, ind_masked)
    selection_mask = np.where(int_ind == False)[0]        

    rand_theta_all, rand_phi_all = eq2ang(rand_ra_all, rand_dec_all)
    ind_rand = hp.ang2pix(nside, rand_theta_all, rand_phi_all)
    del rand_theta_all, rand_phi_all
    int_ind_rand = np.in1d(ind_rand, ind_masked)
    selection_mask_rand = np.where(int_ind_rand == False)[0]



    for jz in range(len(zmin_bins)):
        print('doing z bin:' + str(jz+1))
        minz = zmin_bins[jz]
        maxz = zmax_bins[jz]

        file_suffix_save = '_cat_' + str(cat_tocorr) + '_z_' + str(minz) + '_' + str(maxz) + '_' + 'dojk_' + str(do_jk) + '_njk_' + str(njk)  + '_' + 'desy3' + '_w' + str(int(1)) + '_beam' + str(beam)

        if cat_tocorr == 'halos':
            file_suffix_save += '_logMmin_' + str(logMmin) + '_' + str(logMmax)

        if do_gg:
            filename = save_dir + 'dy/dy_dd_' + 'MICEy'  + '_' + 'nobeam' + '_nside' + str(nside_ymap) + '_mask_' + str(mask_type) + '_' + file_suffix_save + '_ns' + str(nside) + '_v22Feb21_truez.pk'
        else:
            filename = save_dir + 'dy/dy_' + 'MICEy'  + '_' + 'nobeam' + '_nside' + str(nside_ymap) + '_mask_' + str(mask_type) + '_' + file_suffix_save + '_ns' + str(nside) + '_v22Feb21_truez.pk'

        if not os.path.isfile(filename):

            # Restrict to datapoint selection
            selection_z = np.where((datapoint_z_all > minz) & (datapoint_z_all < maxz))[0]
            print("num in selection = ", selection_z.shape)
            selection_f = np.intersect1d(selection_z, selection_mask)


            datapoint_ra = datapoint_ra_all[selection_f]
            datapoint_dec = datapoint_dec_all[selection_f]
            datapoint_z = datapoint_z_all[selection_f]

            ndatapoint = len(datapoint_ra)

            selection_z_rand = np.where((rand_z_all > minz) & (rand_z_all < maxz))[0]
            selection_rand = np.intersect1d(selection_z_rand, selection_mask_rand)
            rand_ra, rand_dec = rand_ra_all[selection_rand], rand_dec_all[selection_rand]

            nrand = len(rand_ra)

            rand_w = np.ones_like(rand_ra)

            print(ndatapoint,nrand)

            zmean_j = (minz + maxz)/2.

            tmp_dir = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/process_cats/tmp_dir/'

            if os.path.isfile(save_dir + save_filename_jk_obj):
                datapoint_cat = treecorr.Catalog(ra=datapoint_ra, dec=datapoint_dec, w=np.ones_like(datapoint_ra), ra_units='degrees',
                                            dec_units='degrees', patch_centers=save_dir + save_filename_jk_obj)           

            else:
                datapoint_cat = treecorr.Catalog(ra=datapoint_ra, dec=datapoint_dec, w=np.ones_like(datapoint_ra), ra_units='degrees',
                                                dec_units='degrees', npatch=njk)
                datapoint_cat.write_patch_centers(save_dir + save_filename_jk_obj)
            rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, w=rand_w, ra_units='degrees', dec_units='degrees', patch_centers=save_dir + save_filename_jk_obj,save_patch_dir=tmp_dir)




            ytruth_cat = treecorr.Catalog(ra=pix_ra, dec=pix_dec, k=ymap_truth, ra_units='degrees', dec_units='degrees')

            dytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0,num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')
            randytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0,num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')

            print('doing dataxy calculation')
            dytruth.process(datapoint_cat, ytruth_cat)

            print('doing auto randomxy calculation')
            randytruth.process(rand_cat, ytruth_cat, low_mem=True)
            dytruth.calculateXi(rk=randytruth)

            xi_dy_full = dytruth.xi
            r_dy = np.exp(dytruth.meanlogr)
            cov_dy = dytruth.cov
            print(r_dy)
            print(xi_dy_full)
            print(np.sqrt(np.diag(cov_dy)))             


            g_g = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                            num_threads=nthreads, bin_slop=bin_slop, sep_units='arcmin', var_method='jackknife')
            g_rg = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                            num_threads=nthreads, bin_slop=bin_slop, sep_units='arcmin', var_method='jackknife')
            rg_rg = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                            num_threads=nthreads, bin_slop=bin_slop, sep_units='arcmin', var_method='jackknife')
            
            print('doing auto dataxdata calculation')
            g_g.process(datapoint_cat, datapoint_cat)

            print('doing auto randomsxrandoms calculation')
            rg_rg.process(rand_cat, rand_cat, low_mem=True)

            print('doing auto dataxrandoms calculation')
            g_rg.process(datapoint_cat, rand_cat, low_mem=True)

            g_g.calculateXi(rr=rg_rg, dr=g_rg)
            xi_gg_full = g_g.xi
            r_gg = np.exp(g_g.meanlogr)
            cov_gg = g_g.cov
            print(r_gg)
            print(xi_gg_full)
            print(np.sqrt(np.diag(cov_gg)))             

            cov_total = treecorr.estimate_multi_cov([g_g,dytruth], 'jackknife')   

            # save_data = {
            #             'dytruth': dytruth,'randytruth': randytruth,
            #             'xi_dy': xi_dy_full,'r_dy': r_dy, 'cov_dy': cov_dy, 'g_g':g_g, 
            #             'rg_rg':rg_rg, 'g_rg':g_rg, 'xi_gg':xi_gg_full, 'r_gg':r_gg, 'cov_gg':cov_gg,
            #             'cov_total':cov_total,
            #             'do_jk': do_jk, 'njk': njk, 'ndatapoint': len(datapoint_ra),
            #             'nrand': len(rand_ra)
            #             }
            save_data = {
                        'xi_dy': xi_dy_full,'r_dy': r_dy, 'cov_dy': cov_dy, 'xi_gg':xi_gg_full, 'r_gg':r_gg, 'cov_gg':cov_gg,
                        'cov_total':cov_total,
                        'do_jk': do_jk, 'njk': njk, 'ndatapoint': len(datapoint_ra),
                        'nrand': len(rand_ra)
                        }                        

            pk.dump(save_data, open(filename, "wb"), protocol = 2)

    pdb.set_trace()

