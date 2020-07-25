import sys, platform, os
sys.path.insert(0,'/global/u1/s/spandey/kmeans_radec/')
import numpy as np
import scipy as sp
import scipy.integrate as integrate
import scipy.signal as spsg
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
import astropy.constants as const
import kmeans_radec
import h5py as h5
import argparse
sys.path.insert(0,'/global/cfs/cdirs/des/shivamp/cosmosis/y3kp-bias-model/3d_stats/process_measure_data/')
import process_cats_class as pcc
cosmo_params_dict = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}

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
#     parser.add_argument('--do_gg', required=True, type=str, help='Do gg corr')
#     parser.add_argument('--do_gy', required=True, type=str, help='Do gy corr')
    parser.add_argument('--mask', required=True, type=str, help='mask type')
    parser.add_argument('--logMmin', default=13.0)
    parser.add_argument('--logMmax', default=13.5)
    parser.add_argument('--do_gg', default=0)
    parser.add_argument('--beam', default=2.4)
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
#     do_gy = args.do_gy
    
    do_jk = True
    put_weights_datapoints = False
    do_randy_sub = True
    if mask_type == 'act':
        njk = 150
    
    if mask_type == 'oct':
        njk = 500

    njk_radec = njk
    njk_z = 1
    # min_r = 0.1
    # max_r = 80.
    nthreads = 64
    bin_slop = 0.0

    minrad = 0.5
    maxrad = 168.0
    nrad = 26

    save_dir = '/global/project/projectdirs/des/shivamp/cosmosis/ACTxDESY3/src/results/'
    save_filename_jk_obj = 'jkobj_MICE_' + '_' + '_njk_' + str(njk) + '.pk'

    ydir = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/'
    if beam == 0:
        true_y_file = ydir + 'ymap_mice_splitCOMBINED_NM50_Nz16_nside4096_v01.fits'
    else:
        true_y_file = ydir + 'ymap_mice_splitCOMBINED_NM50_Nz16_nside4096_v01_beam_' + str(beam) + '_arcmin.fits'

    print('opening ymap and mask')
    ymap_truth = hp.read_map(true_y_file)
    nside = hp.npix2nside(len(ymap_truth))
    nside_ymap = nside
    mask_final = []
    
    if mask_type == 'act':
        filename_mask = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/ACT_D56_mask_inMICE.fits'
    
    if mask_type == 'oct':
        filename_mask = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/octant_mask_inMICE.fits'
        
    gal_mask_input_orig = hp.read_map(filename_mask)
    
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
        datapoint_z_all = df[1].data['z_dnf_mean_sof']
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
        
    for jz in range(len(zmin_bins)):

        print('doing z bin:' + str(jz))
        minz = zmin_bins[jz]
        maxz = zmax_bins[jz]

        file_suffix_save = '_cat_' + str(cat_tocorr) + '_z_' + str(minz) + '_' + str(maxz) + '_' + 'dojk_' + str(do_jk) + '_njk_' + str(njk)  + '_' + 'desy3' + '_w' + str(int(put_weights_datapoints)) + '_beam' + str(beam)
        
        if cat_tocorr == 'halos':
            file_suffix_save += '_logMmin_' + str(logMmin) + '_' + str(logMmax)
        
        if do_gg:
            filename = save_dir + 'dy/dy_dd_' + 'MICEy'  + '_' + 'nobeam' + '_nside' + str(nside_ymap) + '_mask_' + str(mask_type) + '_' + file_suffix_save + '_ns4096.pk'
        else:
            filename = save_dir + 'dy/dy_' + 'MICEy'  + '_' + 'nobeam' + '_nside' + str(nside_ymap) + '_mask_' + str(mask_type) + '_' + file_suffix_save + '_ns4096.pk'
            
        if not os.path.isfile(filename):

            # Restrict to datapoint selection
            selection_z = np.where((datapoint_z_all > minz) & (datapoint_z_all < maxz))[0]
            print("num in selection = ", selection_z.shape)
            # import pdb; pdb.set_trace()

            mask_input = np.copy(gal_mask_input_orig)
            ind_masked = np.where(mask_input < 1e-4)[0]
            theta_datapoint_all, phi_datapoint_all = eq2ang(datapoint_ra_all, datapoint_dec_all)
            ind_datapoints = hp.ang2pix(nside, theta_datapoint_all, phi_datapoint_all)
            int_ind = np.in1d(ind_datapoints, ind_masked)
            selection_mask = np.where(int_ind == False)[0]
            selection_f = np.intersect1d(selection_z, selection_mask)

#             selection_f = selection_z

            datapoint_ra = datapoint_ra_all[selection_f]
            datapoint_dec = datapoint_dec_all[selection_f]
            datapoint_z = datapoint_z_all[selection_f]
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
            
#             selection_rand = selection_z_rand

            # if len(selection_rand) < 15* ndatapoint:
            rand_theta, rand_phi = rand_theta_all[selection_rand], rand_phi_all[selection_rand]
            rand_ra, rand_dec = ang2eq(rand_theta, rand_phi)
            # else:
            #     rand_theta_all, rand_phi_all = rand_theta_all[selection_rand], rand_phi_all[selection_rand]
            #     rand_index_rand = np.unique(np.random.randint(0, len(rand_theta_all), 15 * ndatapoint))
            #     rand_theta, rand_phi = rand_theta_all[rand_index_rand], rand_phi_all[rand_index_rand]
            #     rand_ra, rand_dec = ang2eq(rand_theta, rand_phi)

            nrand = len(rand_ra)

#             rand_theta, rand_phi = eq2ang(rand_ra, rand_dec)
#             mask = np.zeros(hp.nside2npix(nside))
#             mask[ind_datapoints] = 1.

            print(ndatapoint,nrand)


            print('getting JK')
            if do_jk:
                datapoint_radec = np.transpose([datapoint_ra, datapoint_dec])
                # jkobj_map = kmeans_radec.kmeans_sample(datapoint_radec, njk)
                if os.path.isfile(save_dir + save_filename_jk_obj):
                    try:
                        jkobj_map_radec_centers = pk.load(open(save_dir + save_filename_jk_obj, 'rb'))[
                            'jkobj_map_radec_centers']
                    except:
                        jkobj_map_radec_centers = pk.load(open(save_dir + save_filename_jk_obj, 'rb'), encoding='latin1')[
                            'jkobj_map_radec_centers']
                    jkobj_map = KMeans(jkobj_map_radec_centers)
                else:
                    # datapoint_radec = np.transpose([datapoint_ra, datapoint_dec])
                    jkobj_map = kmeans_radec.kmeans_sample(datapoint_radec, njk)
                    jk_dict = {'jkobj_map_radec_centers': jkobj_map.centers}
                    pk.dump(jk_dict, open(save_dir + save_filename_jk_obj, 'wb'), protocol=2)

                datapoint_jk = jkobj_map.find_nearest(datapoint_radec)

                if do_randy_sub:

                    if len(rand_ra) > 2000000:
                        ind_binh = np.arange(len(rand_ra))
                        nsplit = 10
                        ind_binh_split = np.array_split(ind_binh, nsplit)
                        rand_jk = np.zeros_like(rand_ra)
                        for js in range(nsplit):
                            if np.mod(js,5) == 0:
                                print('processing split' + str(js + 1))
                            ind_binh_js = ind_binh_split[js]
                            rand_jk[ind_binh_js] = jkobj_map.find_nearest(
                                np.array([rand_ra[ind_binh_js], rand_dec[ind_binh_js]]).T)
                    else:
                        rand_jk = jkobj_map.find_nearest(np.transpose([rand_ra, rand_dec]))

            index_rand = hp.ang2pix(nside_ymap, rand_theta, rand_phi)
            pix_area = hp.nside2pixarea(nside_ymap, degrees=True)
            catalog_area = (len(np.unique(index_rand))) * pix_area

            # zmin = 0.0
            # zmax = 1.2
            # nzbins_total = 120

            # delta_z = (zmax - zmin) / nzbins_total
            # zarray_all = np.linspace(zmin, zmax, nzbins_total)
            # zarray_edges = (zarray_all[1:] + zarray_all[:-1]) / 2.
            # zarray = zarray_all[1:-1]
            # gaussian_all = np.zeros((len(zarray), len(datapoint_z)))
            # for j in range(len(datapoint_z)):
            #     z = datapoint_z[j]
            #     sig_gauss = 0.0166 * (1 + z)

            #     gaussian_all[:, j] = (1 / (sig_gauss * np.sqrt(2 * np.pi))) * np.exp(-((zarray - z) ** 2) / (2 * (sig_gauss ** 2)))

            # nzbin_j = np.sum(gaussian_all, axis=1)
            # nzbin_j_norm = nzbin_j / (np.sum(nzbin_j) * delta_z)
            # zmean_j = get_zmean(zarray, delta_z, nzbin_j_norm)
            zmean_j = np.mean(datapoint_z)

            print('total data points : ' + str(len(datapoint_ra)))
            print('total random points : ' + str(len(rand_ra)))
            print('catalog area sq deg : ' + str(catalog_area))
            # print('bin zmean : ' + str(zmean_j))


            Omega_m = 0.283705720011
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=100, Om0=Omega_m)
            Dcom_z = (cosmo.comoving_distance(zmean_j)).value

            # minrad = (min_r / Dcom_z)*(180./np.pi)*(60.)
            # maxrad = (max_r / Dcom_z)*(180./np.pi)*(60.)


            if put_weights_datapoints:
                datapoint_weight = datapoint_weight_all[selection_f]
            else:
                datapoint_weight = np.ones_like(datapoint_ra)
            datapoint_cat = treecorr.Catalog(ra=datapoint_ra, dec=datapoint_dec, w=datapoint_weight, ra_units='degrees', dec_units='degrees')
            npix_ymap = len(ymap_truth)
            nside_ymap = hp.npix2nside(npix_ymap)
            pix_theta, pix_phi = hp.pix2ang(nside_ymap, np.arange(npix_ymap))
            pix_ra, pix_dec = ang2eq(pix_theta, pix_phi)
            rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='degrees', dec_units='degrees')


            if do_randy_sub:
                ytruth_cat = treecorr.Catalog(ra=pix_ra, dec=pix_dec, k=ymap_truth,
                                            ra_units='degrees', dec_units='degrees')
            else:
                ytruth_cat = treecorr.Catalog(ra=pix_ra, dec=pix_dec, k=ymap_truth - np.mean(ymap_truth),
                                            ra_units='degrees', dec_units='degrees')

            # perform correlation measurement

            dytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0,num_threads=nthreads, bin_slop=bin_slop)

            randytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0,num_threads=nthreads, bin_slop=bin_slop)
            
            print('doing dataxy calculation')
            dytruth.process(datapoint_cat, ytruth_cat)

            print('doing randomsxy calculation')
            randytruth.process(rand_cat, ytruth_cat)


            np_dytruth_full = dytruth.npairs
            xi_dytruth_full = dytruth.xi

            if do_randy_sub:
                np_randytruth_full = randytruth.npairs
                xi_randytruth_full = randytruth.xi

            xi_dytruth_big_all = np.zeros((njk, nrad))
            xi_randytruth_big_all = np.zeros((njk, nrad))
            
            if do_gg:

                g_g = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                             num_threads=nthreads, bin_slop=bin_slop, sep_units='arcmin')
                g_rg = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                              num_threads=nthreads, bin_slop=bin_slop, sep_units='arcmin')
                rg_rg = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                               num_threads=nthreads, bin_slop=bin_slop, sep_units='arcmin')
                
                print('doing dataxdata calculation')
                g_g.process(datapoint_cat, datapoint_cat)

                print('doing randomsxrandoms calculation')
                rg_rg.process(rand_cat, rand_cat)

                print('doing dataxrandoms calculation')
                g_rg.process(datapoint_cat, rand_cat)


                g_g_np_norm = g_g.npairs * 1. / (1. * ndatapoint * ndatapoint)

                g_rg_np_norm = g_rg.npairs * 1. / (1. * ndatapoint * nrand)

                rg_rg_np_norm = rg_rg.npairs * 1. / (1. * nrand * nrand)

                xi_gg_full = (g_g_np_norm - 2. * g_rg_np_norm + rg_rg_np_norm) / (1. * rg_rg_np_norm)
                r_gg = np.exp(g_g.meanlogr)

                xigg_big_all = np.zeros((njk, nrad))
                rnom_big_all = np.zeros((njk, nrad))

            for j in range(njk):
                if np.mod(j,10) == 0:
                    print('doing jk ' + str(j))
                datapoint_ind_small = np.where(datapoint_jk == j)[0]
                datapoint_cat_small = treecorr.Catalog(ra=datapoint_ra[datapoint_ind_small],
                                                     dec=datapoint_dec[datapoint_ind_small], ra_units='degrees',
                                                     dec_units='degrees')
                


                dytruth_small = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin',
                                                     num_threads=nthreads, bin_slop=bin_slop,verbose=0)
                dytruth_small.process(datapoint_cat_small, ytruth_cat)

                np_dytruth_small = dytruth_small.npairs

                xi_dytruth_small = dytruth_small.xi

                np_dytruth_big = np_dytruth_full - np_dytruth_small

                xi_dytruth_big_all[j, :] = (
                                                 xi_dytruth_full * np_dytruth_full - np_dytruth_small * xi_dytruth_small) / np_dytruth_big

                if do_randy_sub:
                    rand_ind_small = np.where(rand_jk == j)[0]
                    rand_cat_small = treecorr.Catalog(ra=rand_ra[rand_ind_small], dec=rand_dec[rand_ind_small],
                                                    ra_units='degrees', dec_units='degrees')
                    n_g_s, n_rg_s = len(datapoint_ind_small), len(rand_ind_small)
                    n_g_b, n_rg_b = ndatapoint - n_g_s, nrand - n_rg_s

                    randytruth_small = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin',
                                                            num_threads=nthreads, bin_slop=bin_slop,verbose=0)

                    randytruth_small.process(rand_cat_small, ytruth_cat)

                    np_randytruth_small = randytruth_small.npairs
                    xi_randytruth_small = randytruth_small.xi

                    np_randytruth_big = np_randytruth_full - np_randytruth_small

                    xi_randytruth_big_all[j, :] = (
                                                            xi_randytruth_full * np_randytruth_full - np_randytruth_small * xi_randytruth_small) / np_randytruth_big

            
            
                if do_gg:
                    g_g_s_s = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                                     num_threads=nthreads, bin_slop=bin_slop,sep_units='arcmin')
                    g_g_f_s = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                                     num_threads=nthreads, bin_slop=bin_slop,sep_units='arcmin')

                    rg_rg_s_s = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                                       num_threads=nthreads, bin_slop=bin_slop,sep_units='arcmin')
                    rg_rg_f_s = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                                       num_threads=nthreads, bin_slop=bin_slop,sep_units='arcmin')

                    g_rg_s_s = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                                      num_threads=nthreads, bin_slop=bin_slop,sep_units='arcmin')
                    g_rg_f_s = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                                      num_threads=nthreads, bin_slop=bin_slop,sep_units='arcmin')
                    g_rg_s_f = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                                      num_threads=nthreads, bin_slop=bin_slop,sep_units='arcmin')  
                    
                    g_g_s_s.process(datapoint_cat_small, datapoint_cat_small)
                    g_g_f_s.process(datapoint_cat, datapoint_cat_small)
                    rg_rg_s_s.process(rand_cat_small, rand_cat_small)
                    rg_rg_f_s.process(rand_cat, rand_cat_small)
                    g_rg_s_s.process(datapoint_cat_small, rand_cat_small)
                    g_rg_f_s.process(datapoint_cat, rand_cat_small)
                    g_rg_s_f.process(datapoint_cat_small, rand_cat)

                    g_g_s_s_np, g_g_f_s_np = g_g_s_s.npairs, g_g_f_s.npairs
                    rg_rg_s_s_np, rg_rg_f_s_np = rg_rg_s_s.npairs, rg_rg_f_s.npairs
                    g_rg_s_s_np, g_rg_f_s_np, g_rg_s_f_np = g_rg_s_s.npairs, g_rg_f_s.npairs, g_rg_s_f.npairs

                    g_g_b_b_np_norm = (g_g.npairs - 2. * g_g_f_s_np + g_g_s_s_np) / (1. * n_g_b * n_g_b)
                    rg_rg_b_b_np_norm = (rg_rg.npairs - 2. * rg_rg_f_s_np + rg_rg_s_s_np) / (1. * n_rg_b * n_rg_b)
                    g_rg_b_b_np_norm = (g_rg.npairs - g_rg_f_s_np - g_rg_s_f_np + g_rg_s_s_np) / (1. * n_g_b * n_rg_b)

                    xi_gg_big = (g_g_b_b_np_norm - 2. * g_rg_b_b_np_norm + rg_rg_b_b_np_norm) / (1. * rg_rg_b_b_np_norm)

                    xigg_big_all[j, :] = xi_gg_big
                    rnom_big_all[j, :] = np.exp(g_g_s_s.meanlogr)
                    
                    xi_gg_mean = np.tile(xi_gg_full.transpose(), (njk, 1))
                    xi_gg_sub = xigg_big_all - xi_gg_mean
                    xi_gg_cov = (1.0 * (njk - 1.) / njk) * np.matmul(xi_gg_sub.T, xi_gg_sub)
                    xi_gg_sig = np.sqrt(np.diag(xi_gg_cov))



                    
            if do_gg:
                save_data = {'dytruth': dytruth,
                           'randytruth': randytruth,
                           'xi_dytruth_big_all': xi_dytruth_big_all,
                           'xi_randytruth_big_all': xi_randytruth_big_all, 'minz': minz,
                           'maxz': maxz,'xi_gg_full': xi_gg_full, 'r_gg': r_gg, 'xigg_big_all': xigg_big_all,
                           'r_gg_all': rnom_big_all,
                           'cov': xi_gg_cov, 'sig': xi_gg_sig,
                           'do_jk': do_jk, 'njk': njk, 'ndatapoint': len(datapoint_ra),
                           'nrand': len(rand_ra), 'area_datapoint_sqdeg': catalog_area,'zmean':zmean_j,'Dcom_z':Dcom_z}
            else:
                save_data = {'dytruth': dytruth,
                           'randytruth': randytruth,
                           'xi_dytruth_big_all': xi_dytruth_big_all,
                           'xi_randytruth_big_all': xi_randytruth_big_all, 'minz': minz,
                           'maxz': maxz,
                           'do_jk': do_jk, 'njk': njk, 'ndatapoint': len(datapoint_ra),
                           'nrand': len(rand_ra), 'area_datapoint_sqdeg': catalog_area,'zmean':zmean_j,'Dcom_z':Dcom_z}


            

            pk.dump(save_data, open(filename, "wb"), protocol = 2)


    pdb.set_trace()


