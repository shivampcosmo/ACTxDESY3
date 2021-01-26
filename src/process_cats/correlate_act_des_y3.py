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
    parser.add_argument('--dp', required=True, type=str, help='deproj type')
    args_all = parser.parse_args()
    return args_all

if __name__ == "__main__":
    args = parse_arguments()

    # redmagic, maglim
    cat_tocorr = args.cat
    # cat_tocorr = 'redmagic'
    # None, cib, cmb
    deproj = args.dp
    do_jk = True
    put_weights_datapoints = True
    do_randy_sub = True
    njk = 100
    # min_r = 0.1
    # max_r = 80.
    nthreads = 10
    bin_slop = 0.0

    minrad = 0.5
    maxrad = 150.0
    nrad = 25
    # mastercatv = 'Y3_mastercat_03_31_20'

    save_dir = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/'
    save_filename_jk_obj = 'jkobj_DES_' + 'v_1_2_0_unblind' + '_' + '_njk_' + str(njk) + '.pk'

    # ydir = '/global/project/projectdirs/des/shivamp/ACTxDESY3_data/act_ymap_releases/v1.0.0/'
    ydir = '/global/cscratch1/sd/msyriac/data/depot/tilec/v1.2.0_20200324/map_v1.2.0_joint_deep56/'
    if deproj == 'None':
        true_y_file = ydir + 'tilec_single_tile_deep56_comptony_map_v1.2.0_joint.fits'
    if deproj == 'cib':
        true_y_file = ydir + 'tilec_single_tile_deep56_comptony_deprojects_cib_map_v1.2.0_joint.fits'
    if deproj == 'cmb':
        true_y_file = ydir + 'tilec_single_tile_deep56_comptony_deprojects_cmb_map_v1.2.0_joint.fits'

    if cat_tocorr == 'maglim':
        # minz = 0.2
        # maxz = 1.05
        zmin_bins = np.array([0.20, 0.40, 0.55, 0.70, 0.85, 0.95])
        zmax_bins = np.array([0.40, 0.55, 0.70, 0.85, 0.95, 1.05 ])

    if cat_tocorr == 'redmagic':
        # minz = 0.15
        # maxz = 0.9
        zmin_bins = np.array([0.15, 0.35, 0.5, 0.65, 0.8])
        zmax_bins = np.array([0.35, 0.5, 0.65, 0.8, 0.9])

    rad2deg = 180./np.pi
    print('opening ymap and mask')
    imapy = enmap.read_map(true_y_file)
    decs,ras = imapy.posmap()
    ra_y, dec_y, ymap_truth = np.array(rad2deg * ras.flatten()), np.array(rad2deg * decs.flatten()), np.array(imapy.flatten())    
    # ymap_truth = hp.read_map(true_y_file)
    # nside = hp.npix2nside(len(ymap_truth))
    # nside_ymap = nside
    # mask_final = []
    
    filename_mask = ydir + 'tilec_mask.fits'
    imapm =  enmap.read_map(filename_mask)
    decs,ras = imapm.posmap()
    ra_m, dec_m, mask = np.array(rad2deg * ras.flatten()), np.array(rad2deg * decs.flatten()), np.array(imapm.flatten())
    theta_mask_all, phi_mask_all = eq2ang(ra_m, dec_m)
    nside = 512
    ind_mask = hp.ang2pix(nside, theta_mask_all, phi_mask_all)
    mask_input = np.copy(mask)
    ind_masked = ind_mask[np.where(mask_input < 1e-4)[0]]
    ind_unmasked = ind_mask[np.where(mask_input > 1e-4)[0]]   

    theta_datapoint_all, phi_datapoint_all = eq2ang(ra_y, dec_y)
    ind_datapoints = hp.ang2pix(nside, theta_datapoint_all, phi_datapoint_all)
    int_ind = np.in1d(ind_datapoints, ind_unmasked)
    selection_f = np.where(int_ind == True)[0]     
    ra_y, dec_y, ymap_truth = ra_y[selection_f], dec_y[selection_f], ymap_truth[selection_f]
    # gal_mask_input_orig = hp.read_map(filename_mask)

    print('opening DES catalog')

    # fname = '/global/cscratch1/sd/troxel/cats_des_y3/Y3_mastercat___UNBLIND___final_v1.0_DO_NOT_USE_FOR_2PT.h5'
    fname = '/project/projectdirs/des/www/y3_cats/Y3_mastercat___UNBLIND___final_v1.1_12_22_20.h5'
    with h5.File(fname,'r') as cat:
        if cat_tocorr == 'maglim':
            ind_sel = cat['index/maglim']['select'][()]
            datapoint_z_all = cat['catalog/dnf/unsheared/zmean_sof'][()][ind_sel]
            datapoint_ra_all = cat['catalog/maglim/ra'][()][ind_sel]
            datapoint_dec_all = cat['catalog/maglim/dec'][()][ind_sel]
            datapoint_weight_all = cat['catalog/maglim/weight'][()][ind_sel]

            rand_ra_all = cat['randoms/maglim/ra'][()]
            rand_dec_all = cat['randoms/maglim/dec'][()]
            rand_z_all = cat['randoms/maglim/z'][()]

        if cat_tocorr == 'redmagic':
            ind_sel = cat['index/redmagic/combined_sample_fid/select'][()]
            datapoint_z_all = cat['catalog/redmagic/combined_sample_fid/zredmagic'][()][ind_sel]
            datapoint_ra_all = cat['catalog/redmagic/combined_sample_fid/ra'][()][ind_sel]
            datapoint_dec_all = cat['catalog/redmagic/combined_sample_fid/dec'][()][ind_sel]
            datapoint_weight_all = cat['catalog/redmagic/combined_sample_fid/weight'][()][ind_sel]

            ind_selr = cat['index/redmagic/combined_sample_fid/random_select'][()]
            rand_ra_all = cat['randoms/redmagic/combined_sample_fid/ra'][()][ind_selr]
            rand_dec_all = cat['randoms/redmagic/combined_sample_fid/dec'][()][ind_selr]
            rand_z_all = cat['randoms/redmagic/combined_sample_fid/z'][()][ind_selr]
            rand_weight_all = np.ones_like(rand_z_all)


    for jz in range(len(zmin_bins)):

        print('doing z bin:' + str(jz+1))
        minz = zmin_bins[jz]
        maxz = zmax_bins[jz]

        file_suffix_save = '_cat_' + str(cat_tocorr) + '_z_' + str(minz) + '_' + str(maxz) + '_' + 'dojk_' + str(do_jk) + '_njk_' + str(njk)  + '_' + 'desy3'

        filename = save_dir + 'dy/dy_' + 'act_deprojects_' + str(deproj) + '_v1.2.0_' + 'wbeam' + '_pixell' + '_' + file_suffix_save + '_hrespixell_v16Jan21.pk'
        if not os.path.isfile(filename):

            # Restrict to datapoint selection
            selection_z = np.where((datapoint_z_all > minz) & (datapoint_z_all < maxz))[0]
            print("num in selection = ", selection_z.shape)
            
            # import pdb; pdb.set_trace()

            # mask_input = np.copy(gal_mask_input_orig)
            # ind_masked = np.where(mask_input < 1e-4)[0]
            # theta_datapoint_all, phi_datapoint_all = eq2ang(datapoint_ra_all, datapoint_dec_all)
            # ind_datapoints = hp.ang2pix(nside, theta_datapoint_all, phi_datapoint_all)
            # int_ind = np.in1d(ind_datapoints, ind_masked)
            # selection_mask = np.where(int_ind == False)[0]
            # selection_f = np.intersect1d(selection_z, selection_mask)
            selection_f = selection_z

            datapoint_ra = datapoint_ra_all[selection_f]
            datapoint_dec = datapoint_dec_all[selection_f]
            datapoint_z = datapoint_z_all[selection_f]
            # theta_datapoint = theta_datapoint_all[selection_f]
            # costheta_datapoint = np.cos(theta_datapoint)
            # phi_datapoint = phi_datapoint_all[selection_f]
            datapoint_w = datapoint_weight_all[selection_f]
            ndatapoint = len(datapoint_ra)

            selection_z_rand = np.where((rand_z_all > minz) & (rand_z_all < maxz))[0]
            # rand_theta_all, rand_phi_all = eq2ang(rand_ra_all, rand_dec_all)
            # ind_rand = hp.ang2pix(nside, rand_theta_all, rand_phi_all)
            # int_ind_rand = np.in1d(ind_rand, ind_masked)
            # selection_mask_rand = np.where(int_ind_rand == False)[0]
            # selection_rand = np.intersect1d(selection_z_rand, selection_mask_rand)
            selection_rand = selection_z_rand

            # rand_theta, rand_phi = rand_theta_all[selection_rand], rand_phi_all[selection_rand]
            rand_ra, rand_dec = rand_ra_all[selection_rand], rand_dec_all[selection_rand]

            nrand = len(rand_ra)

            # rand_theta, rand_phi = eq2ang(rand_ra, rand_dec)
            rand_w = np.ones_like(rand_ra)

            # mask = np.zeros(hp.nside2npix(nside))

            # mask[ind_datapoints] = 1.

            print(ndatapoint,nrand)

            # index_rand = hp.ang2pix(nside_ymap, rand_theta, rand_phi)
            # pix_area = hp.nside2pixarea(nside_ymap, degrees=True)
            # catalog_area = (len(np.unique(index_rand))) * pix_area

            zmean_j = (minz + maxz)/2.

            # print('total data points : ' + str(len(datapoint_ra)))
            # print('total random points : ' + str(len(rand_ra)))
            # print('catalog area sq deg : ' + str(catalog_area))


            if os.path.isfile(save_dir + save_filename_jk_obj):
                datapoint_cat = treecorr.Catalog(ra=datapoint_ra, dec=datapoint_dec, w=datapoint_w, ra_units='degrees',
                                            dec_units='degrees', patch_centers=save_dir + save_filename_jk_obj)           
                
            else:
                datapoint_cat = treecorr.Catalog(ra=datapoint_ra, dec=datapoint_dec, w=datapoint_w, ra_units='degrees',
                                                dec_units='degrees', npatch=njk)
                datapoint_cat.write_patch_centers(save_dir + save_filename_jk_obj)
            rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, w=rand_w, ra_units='degrees', dec_units='degrees', patch_centers=save_dir + save_filename_jk_obj)

            # npix_ymap = len(ymap_truth)
            # nside_ymap = hp.npix2nside(npix_ymap)
            # pix_theta, pix_phi = hp.pix2ang(nside_ymap, np.arange(npix_ymap))
            # pix_ra, pix_dec = ang2eq(pix_theta, pix_phi)


            # ytruth_cat = treecorr.Catalog(ra=pix_ra, dec=pix_dec, k=ymap_truth, ra_units='degrees', dec_units='degrees')
            ytruth_cat = treecorr.Catalog(ra=ra_y, dec=dec_y, k=ymap_truth, ra_units='degrees', dec_units='degrees')

            dytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0,num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')
            randytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0,num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')

            print('doing dataxy calculation')
            dytruth.process(datapoint_cat, ytruth_cat)

            print('doing auto randomxy calculation')
            randytruth.process(rand_cat, ytruth_cat)
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
            rg_rg.process(rand_cat, rand_cat)

            print('doing auto dataxrandoms calculation')
            g_rg.process(datapoint_cat, rand_cat)

            g_g.calculateXi(rr=rg_rg, dr=g_rg)
            xi_gg_full = g_g.xi
            r_gg = np.exp(g_g.meanlogr)
            cov_gg = g_g.cov
            print(r_gg)
            print(xi_gg_full)
            print(np.sqrt(np.diag(cov_gg)))             

            cov_total = treecorr.estimate_multi_cov([g_g,dytruth], 'jackknife')   

            save_data = {
                        'dytruth': dytruth,'randytruth': randytruth,
                        'xi_dy': xi_dy_full,'r_dy': r_dy, 'cov_dy': cov_dy, 'g_g':g_g, 
                        'rg_rg':rg_rg, 'g_rg':g_rg, 'xi_gg':xi_gg_full, 'r_gg':r_gg, 'cov_gg':cov_gg,
                        'cov_total':cov_total,
                        'do_jk': do_jk, 'njk': njk, 'ndatapoint': len(datapoint_ra),
                        'nrand': len(rand_ra)
                        }

            pk.dump(save_data, open(filename, "wb"), protocol = 2)


    pdb.set_trace()


