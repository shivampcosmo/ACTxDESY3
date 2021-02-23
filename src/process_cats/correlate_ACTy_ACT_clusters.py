import sys, platform, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pdb
import healpy as hp
from astropy.io import fits
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
    parser.add_argument('--njk', default=100, type=int, help='Cat type')
    parser.add_argument('--Mmin', type=float, default=0.0, help='Mmin (1e14 M_sun for M500c)')
    parser.add_argument('--Mmax', type=float, default=100.0, help='Mmax (1e14 M_sun for M500c)')
    parser.add_argument('--zmin', type=float, default=0.0, help='rmin')
    parser.add_argument('--zmax', type=float, default=2.0, help='rmax')
    parser.add_argument('--smin', type=float, default=4, help='SNR min')
    parser.add_argument('--smax', type=float, default=100, help='SNR max')

    args_all = parser.parse_args()
    return args_all


if __name__ == "__main__":
    args = parse_arguments()

    # redmagic, maglim
    njk = args.njk
    Mmin = args.Mmin
    Mmax = args.Mmax
    minz = args.zmin
    maxz = args.zmax
    mins = args.smin
    maxs = args.smax
    ydir = '/global/project/projectdirs/des/shivamp/ACTxDESY3_data/act_ymap_releases/v1.0.0/'

    # true_y_file = ydir + 'tilec_single_tile_deep56_comptony_map_v1.0.0_rc_joint_healpix.fits'
    true_y_file = ydir + 'tilec_single_tile_deep56_comptony_deprojects_cib_map_v1.0.0_rc_joint_healpix.fits'


    ymap_truth = hp.read_map(true_y_file)

    print('opening Cluster catalog')
    df = fits.open('/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/process_cats/DR5_cluster-catalog_v1.0.fits.txt')[1].data   

    df2 = pk.load(open('/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/process_cats/randoms_DR5_cluster-catalog_v1.0.pk','rb'))   

        
    ra_all, dec_all, snr_all, z_all, M500c_all = df['RADeg'], df['decDeg'], df['SNR'], df['redshift'], df['M500c']

    ind_sel = np.where((M500c_all > Mmin) & (M500c_all < Mmax) & (snr_all > mins) & (snr_all < maxs) )[0]
    datapoint_z_all, datapoint_ra_all, datapoint_dec_all = z_all[ind_sel], ra_all[ind_sel], dec_all[ind_sel]
    datapoint_weight_all = np.ones_like(datapoint_z_all)

    rand_ra_all = df2['ra']
    rand_dec_all = df2['dec']
    rand_z_all = df2['z']

    # datapoint_ra_all, datapoint_dec_all = data_coord_cov(datapoint_ra_all, datapoint_dec_all,icrs2gal=True)
    # rand_ra_all, rand_dec_all = data_coord_cov(rand_ra_all, rand_dec_all,icrs2gal=True)


    # Restrict to datapoint selection
    selection_z = np.where((datapoint_z_all > minz) & (datapoint_z_all < maxz))[0]
    print("num in selection = ", selection_z.shape)
    # import pdb; pdb.set_trace()

    print('opening ymap and mask')
    ymap_truth = hp.read_map(true_y_file)
    nside = hp.npix2nside(len(ymap_truth))
    nside_ymap = nside
    mask_final = []
    filename_mask = '/global/project/projectdirs/des/shivamp/ACTxDESY3_data/act_ymap_releases/v1.0.0/tilec_mask_healpix.fits'
    gal_mask_input_orig = hp.read_map(filename_mask)

    # nside = hp.npix2nside(len(ymap_truth))
    # mask_final = []
    # gal_mask_input_orig = hp.read_map(true_y_file,field=3)

    mask_input = gal_mask_input_orig

    ind_masked = np.where(mask_input < 1e-4)[0]
    theta_datapoint_all, phi_datapoint_all = eq2ang(datapoint_ra_all, datapoint_dec_all)
    ind_datapoints = hp.ang2pix(nside, theta_datapoint_all, phi_datapoint_all)
    int_ind = np.in1d(ind_datapoints, ind_masked)
    selection_mask = np.where(int_ind == False)[0]
    selection_f = np.intersect1d(selection_z, selection_mask)

    datapoint_ra = datapoint_ra_all[selection_f]
    datapoint_dec = datapoint_dec_all[selection_f]
    datapoint_z = datapoint_z_all[selection_f]
    datapoint_w = datapoint_weight_all[selection_f]
    theta_datapoint = theta_datapoint_all[selection_f]
    costheta_datapoint = np.cos(theta_datapoint)
    phi_datapoint = phi_datapoint_all[selection_f]

    ndatapoint = len(datapoint_ra)

    selection_z_rand = np.where((rand_z_all > minz) & (rand_z_all < maxz))[0]
    rand_theta_all, rand_phi_all = eq2ang(rand_ra_all, rand_dec_all)
    ind_rand = hp.ang2pix(nside, rand_theta_all, rand_phi_all)
    int_ind_rand = np.in1d(ind_rand, ind_masked)
    selection_mask_rand = np.where(int_ind_rand == False)[0]
    selection_rand = np.intersect1d(selection_z_rand, selection_mask_rand)

    rand_theta, rand_phi = rand_theta_all[selection_rand], rand_phi_all[selection_rand]
    rand_ra, rand_dec = ang2eq(rand_theta, rand_phi)

    nrand = len(rand_ra)

    rand_theta, rand_phi = eq2ang(rand_ra, rand_dec)
    rand_w = np.ones_like(rand_ra)
    mask = np.zeros(hp.nside2npix(nside))

    mask[ind_datapoints] = 1.

    print(ndatapoint,nrand)

    do_jk = True
    nside_ymap = nside
    put_weights_datapoints = True
    do_randy_sub = True

    index_rand = hp.ang2pix(128, rand_theta, rand_phi)
    pix_area = hp.nside2pixarea(128, degrees=True)
    catalog_area = (len(np.unique(index_rand))) * pix_area

    zmin = minz
    zmax = maxz

    zmean_j = (minz + maxz)/2.

    print('total data points : ' + str(len(datapoint_ra)))
    print('total random points : ' + str(len(rand_ra)))
    print('catalog area sq deg : ' + str(catalog_area))
    print('bin zmean : ' + str(zmean_j))

    min_r = 0.1
    max_r = 8.0
    minrad = 0.5
    maxrad = 8.0
    bin_slop = 0.0
    nthreads = 40
    nrad = 12
    save_dir = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/process_cats/'
    save_filename_jk_obj = 'JK_centers_ACTarea_clusters_y_njk_' + str(njk) + '.txt'

    if os.path.isfile(save_dir + save_filename_jk_obj):
        datapoint_cat = treecorr.Catalog(ra=datapoint_ra, dec=datapoint_dec, w=datapoint_w, ra_units='degrees',
                                    dec_units='degrees', patch_centers=save_dir + save_filename_jk_obj)           
        
    else:
        datapoint_cat = treecorr.Catalog(ra=datapoint_ra, dec=datapoint_dec, w=datapoint_w, ra_units='degrees',
                                        dec_units='degrees', npatch=njk)
        datapoint_cat.write_patch_centers(save_dir + save_filename_jk_obj)
    rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, w=rand_w, ra_units='degrees', dec_units='degrees', patch_centers=save_dir + save_filename_jk_obj)

    npix_ymap = len(ymap_truth)
    nside_ymap = hp.npix2nside(npix_ymap)
    pix_theta, pix_phi = hp.pix2ang(nside_ymap, np.arange(npix_ymap))
    pix_ra, pix_dec = ang2eq(pix_theta, pix_phi)

    ytruth_cat = treecorr.Catalog(ra=pix_ra, dec=pix_dec, k=ymap_truth - np.mean(ymap_truth),
                                ra_units='degrees', dec_units='degrees')

    # perform correlation measurement
    dytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0,num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')
    randytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0,num_threads=nthreads, bin_slop=bin_slop, var_method='jackknife')

    dytruth.process(datapoint_cat, ytruth_cat)
    randytruth.process(rand_cat, ytruth_cat)
    dytruth.calculateXi(rk=randytruth)

    xi_dy_full = dytruth.xi
    r_dy = np.exp(dytruth.meanlogr)
    cov_dy = dytruth.cov
    print(r_dy)
    print(xi_dy_full)
    print(np.sqrt(np.diag(cov_dy)))  

    save_data = { 'xi_dy': xi_dy_full,
                'r_dy':r_dy, 'minz': minz,
                'maxz': maxz, 'Mmin':Mmin, 'Mmax':Mmax,
                'do_jk': do_jk, 'njk': njk, 'cov_dy':cov_dy}

    # Save output
    data_output_dir = '/global/project/projectdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/'

    file_suffix_save = '_cat_' + str('ACTclusters') + '_z_' + str(minz) + '_' + str(maxz) + '_M_' + str(Mmin) + '_' + str(Mmax) + '_' + 'dojk_' + str(do_jk) + '_njk_' + str(njk) 

    filename = data_output_dir + 'dy/dy_' + 'ACTdeprojCIB_' + 'fwhm_2p4arcmin' + '_nside' + str(nside_ymap) + '_' + file_suffix_save + '.pk'

    pk.dump(save_data, open(filename, "wb"), protocol = 2)

    ipdb.set_trace()

