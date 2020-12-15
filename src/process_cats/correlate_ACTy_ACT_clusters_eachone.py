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
sys.path.insert(0, os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/helper/')
import LSS_funcs as hmf

import h5py as h5
import argparse
from scipy.integrate import quad
from scipy.optimize import fsolve
import scipy.optimize as op
import scipy as sp
import mycosmo as cosmodef
import colossus
from colossus.cosmology import cosmology
from colossus.lss import bias
from colossus.lss import mass_function
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import concentration

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

class general_funcs:

    def __init__(self, cosmo_params):
        h = cosmo_params['H0'] / 100.
        cosmo_func = cosmodef.mynew_cosmo(h, cosmo_params['Om0'], cosmo_params['Ob0'], cosmo_params['ns'],
                                          cosmo_params['sigma8'])
        self.cosmo = cosmo_func

    def get_Dcom(self, zf):
        c = 3 * 10 ** 5
        Omega_m, Omega_L = self.cosmo.Om0, 1. - self.cosmo.Om0
        res1 = sp.integrate.quad(lambda z: (c / 100) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
        Dcom = res1[0]
        return Dcom

    def get_Dcom_array(self,zarray):
        Omega_m = self.cosmo.Om0
        Omega_L = 1. - Omega_m
        c = 3 * 10 ** 5
        Dcom_array = np.zeros(len(zarray))
        for j in range(len(zarray)):
            zf = zarray[j]
            res1 = sp.integrate.quad(lambda z: (c / 100) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
            Dcom = res1[0]
            Dcom_array[j] = Dcom
        return Dcom_array

    def get_Hz(self,zarray):
        Omega_m = self.cosmo.Om0
        Omega_L = 1 - Omega_m
        Ez = np.sqrt(Omega_m * (1 + zarray) ** 3 + Omega_L)
        Hz = 100. * Ez
        return Hz

    def get_diff(self, zf, chi):
        return chi - self.get_Dcom(zf)

    def root_find(self, init_x, chi):
        nll = lambda *args: self.get_diff(*args)
        result = op.root(nll, np.array([init_x]), args=chi, options={'maxfev': 50}, tol=0.01)
        return result.x[0]

    def get_z_from_chi(self, chi):
        valf = self.root_find(0., chi)
        return valf


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--njk', default=200, type=int, help='Cat type')
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
    datapoint_M_all = M500c_all[ind_sel]

    rand_ra_all = df2['ra']
    rand_dec_all = df2['dec']
    rand_z_all = df2['z']

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
    datapoint_M = datapoint_M_all[selection_f]
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
    rand_z = rand_z_all[selection_rand]
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
    cosmo_params_dict = {'flat': True, 'H0': 70.0, 'Om0': 0.315, 'Ob0': 0.044, 'sigma8': 0.811, 'ns': 0.95}
    cosmology.addCosmology('mock_cosmo', cosmo_params_dict)
    cosmo_colossus = cosmology.setCosmology('mock_cosmo')
    rho_crit_array = cosmo_colossus.rho_c(datapoint_z) * (1000 ** 3)
    hv = cosmo_params_dict['H0']/100.
    datapoint_radius = hmf.get_R_from_M(datapoint_M*hv*1e14, 500.*rho_crit_array)

    gnf = general_funcs(cosmo_params_dict)
    chi_array = np.linspace(0, 5000, 50000)
    z_array = np.zeros(len(chi_array))
    for j in range(len(z_array)):
        z_array[j] = gnf.get_z_from_chi(chi_array[j])
    z_interp = interpolate.interp1d(chi_array, z_array)
    th_thv_min = 0.2
    th_thv_max = 4.0
    npix_ymap = len(ymap_truth)
    nside_ymap = hp.npix2nside(npix_ymap)
    pix_theta, pix_phi = hp.pix2ang(nside_ymap, np.arange(npix_ymap))
    pix_ra, pix_dec = ang2eq(pix_theta, pix_phi)
    nrad = 12
    xi_all_data = np.zeros((len(datapoint_ra), nrad))
    xiVy_all_data = np.zeros((len(datapoint_ra), nrad))
    xiRy_all_data = np.zeros((len(datapoint_ra), nrad))
    theta_all_data = np.zeros((len(datapoint_ra), nrad))
    th_thv_all_data = np.zeros((len(datapoint_ra), nrad))
    thetav_data = np.zeros(len(datapoint_ra))
    Dcomv_data = np.zeros(len(datapoint_ra))

    fac_mult = 200
    fac_multy = 200
    for jv in range(len(datapoint_ra)):
        ra_jv, dec_jv, z_jv, r_jv, w_jv = datapoint_ra[jv], datapoint_dec[jv], datapoint_z[jv], datapoint_radius[jv], datapoint_w[jv]
        Dcom_jv = gnf.get_Dcom(z_jv)
        if Dcom_jv - fac_mult*r_jv > 0:
            zmin_sel_jv = z_interp(Dcom_jv - fac_mult*r_jv)
        else:
            zmin_sel_jv = 0.
        zmax_sel_jv = z_interp(Dcom_jv + fac_mult*r_jv)
        thv_jv = (r_jv/Dcom_jv)*(180.*60./np.pi)
        thv_jv_deg = (r_jv/Dcom_jv)*(180./np.pi)

        rand_ind_sel_void = np.where((rand_z > z_jv - 0.1) & (rand_z < z_jv + 0.1))[0]
        rand_ra_jv, rand_dec_jv, rand_z_jv = rand_ra[rand_ind_sel_void], rand_dec[rand_ind_sel_void], rand_z[rand_ind_sel_void]

        minrad = th_thv_min * thv_jv
        maxrad = th_thv_max * thv_jv

        datapoint_cat = treecorr.Catalog(ra=[ra_jv], dec=[dec_jv], w=[w_jv], ra_units='degrees',
                                            dec_units='degrees')

        y_ind_sel_void = np.where((pix_ra > ra_jv - fac_multy*thv_jv_deg) & (pix_ra < ra_jv + fac_multy*thv_jv_deg) & \
            (pix_dec > dec_jv - fac_multy*thv_jv_deg) & (pix_dec < dec_jv + fac_multy*thv_jv_deg))[0]
        ytruth_catv = treecorr.Catalog(ra=pix_ra[y_ind_sel_void], dec=pix_dec[y_ind_sel_void], k=ymap_truth[y_ind_sel_void],ra_units='degrees', dec_units='degrees')

        rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='degrees', dec_units='degrees')

        dytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad,  sep_units='arcmin', verbose=0)
        randytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad,  sep_units='arcmin', verbose=0)

        randytruth.process(rand_cat, ytruth_catv)
        dytruth.process(datapoint_cat, ytruth_catv)

        xiVy_all_data[jv,:] = dytruth.xi
        xiRy_all_data[jv,:] = randytruth.xi
        xi_all_data[jv,:] = dytruth.xi - randytruth.xi
        theta_all_data[jv,:] = np.exp(dytruth.meanlogr)
        thetav_data[jv] = thv_jv
        th_thv_all_data[jv,:] = dytruth.rnom/thv_jv
        Dcomv_data[jv] = Dcom_jv

        if np.mod(jv,10) == 0:
            print(dytruth.xi - randytruth.xi)
            print('doing the cluster:' + str(jv+1))
            print(zmin_sel_jv, zmax_sel_jv, r_jv)
            print('random points here:' + str(len(rand_ra_jv)))
            save_data = {'ra':datapoint_ra, 'dec':datapoint_dec, 'z':datapoint_z, 'rv':datapoint_radius,
            'theta_thv_all':th_thv_all_data, 'Dcom_all':Dcomv_data, 'thv_data':thetav_data, 'theta_all_data':theta_all_data,
            'xiVy_all_data':xiVy_all_data, 'xiRy_all_data':xiRy_all_data, 'M':datapoint_M}

            data_output_dir = '/global/project/projectdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/'

            file_suffix_save = '_cat_' + str('ACTclusters') + '_z_' + str(minz) + '_' + str(maxz) + '_M_' + str(Mmin) + '_' + str(Mmax) + '_' + 'dojk_' + str(do_jk) + '_njk_' + str(njk) + '_scales_' + str(th_thv_min) + '_' + str(th_thv_max) + '_eachone'

            # filename = data_output_dir + 'dy/dy_' + 'ACT_' + 'fwhm_1p6arcmin' + '_nside' + str(nside_ymap) + '_' + file_suffix_save + '.pk'
            filename = data_output_dir + 'dy/dy_' + 'ACTdeprojCIB_' + 'fwhm_2p4arcmin' + '_nside' + str(nside_ymap) + '_' + file_suffix_save + '.pk'
            pk.dump(save_data, open(filename, "wb"), protocol = 2)


# Save output
save_data = {'ra':datapoint_ra, 'dec':datapoint_dec, 'z':datapoint_z, 'rv':datapoint_radius,
'theta_thv_all':th_thv_all_data, 'Dcom_all':Dcomv_data, 'thv_data':thetav_data, 'theta_all_data':theta_all_data,
             'xiVy_all_data':xiVy_all_data, 'xiRy_all_data':xiRy_all_data, 'M':datapoint_M}

data_output_dir = '/global/project/projectdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/'

file_suffix_save = '_cat_' + str('ACTclusters') + '_z_' + str(minz) + '_' + str(maxz) + '_M_' + str(Mmin) + '_' + str(Mmax) + '_' + 'dojk_' + str(do_jk) + '_njk_' + str(njk) + '_scales_' + str(th_thv_min) + '_' + str(th_thv_max) + '_eachone'

# filename = data_output_dir + 'dy/dy_' + 'ACT_' + 'fwhm_1p6arcmin' + '_nside' + str(nside_ymap) + '_' + file_suffix_save + '.pk'
filename = data_output_dir + 'dy/dy_' + 'ACTdeprojCIB_' + 'fwhm_2p4arcmin' + '_nside' + str(nside_ymap) + '_' + file_suffix_save + '.pk'

pk.dump(save_data, open(filename, "wb"), protocol = 2)

pdb.set_trace()




