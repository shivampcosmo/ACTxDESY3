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
import scipy.interpolate as interpolate
from numpy.linalg import inv
import pdb
import time
from scipy.integrate import quad
from scipy.optimize import fsolve
import scipy.optimize as op
import scipy as sp
import mycosmo as cosmodef

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
    parser.add_argument('--njk', default=100, type=int, help='Cat type')
    parser.add_argument('--rmin', type=float, default=20.0, help='rmin')
    parser.add_argument('--rmax', type=float, default=80.0, help='rmax')
    parser.add_argument('--zmin', type=float, default=0.15, help='rmin')
    parser.add_argument('--zmax', type=float, default=0.6, help='rmax')
    args_all = parser.parse_args()
    return args_all


if __name__ == "__main__":
    args = parse_arguments()

    njk = args.njk
    rmin = args.rmin
    rmax = args.rmax

    zmin = args.zmin
    zmax = args.zmax
    true_y_file = '/global/cfs/cdirs/des/shivamp/actxdes/data_set/planck_data/pl2015/nilc_ymaps.fits'

    ymap_truth = hp.read_map(true_y_file)

    print('opening DES catalog')
    mastercatv = 'Y3_mastercat_03_31_20'
    fname = '/global/cfs/cdirs/des/www/y3_cats/' + mastercatv + '.h5'
    fnamev = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/data/untrimmed_sky_positions_central_Redmagic_highdens_y3a2_v0.5.1.out'

    with h5.File(fname,'r') as cat:
        dfv = np.loadtxt(fnamev)
        void_r_all = dfv[:,3]
        ind_sel = np.where((void_r_all > rmin) & (void_r_all < rmax))[0]
        datapoint_z_all, datapoint_ra_all, datapoint_dec_all = dfv[ind_sel,2], dfv[ind_sel,0], dfv[ind_sel,1]
        datapoint_weight_all = np.ones_like(datapoint_z_all)
        datapoint_radius_all = dfv[ind_sel,3]
        
        ind_sel = cat['index/redmagic/combined_sample_fid/select'][()]
        rm_z_all = cat['catalog/redmagic/combined_sample_fid/zredmagic'][()][ind_sel]
        rm_ra_all = cat['catalog/redmagic/combined_sample_fid/ra'][()][ind_sel]
        rm_dec_all = cat['catalog/redmagic/combined_sample_fid/dec'][()][ind_sel]
        rm_weight_all = cat['catalog/redmagic/combined_sample_fid/weight'][()][ind_sel]

        ind_selr = cat['index/redmagic/combined_sample_fid/random_select'][()]
        rand_ra_all = cat['randoms/redmagic/combined_sample_fid/ra'][()][ind_selr]
        rand_dec_all = cat['randoms/redmagic/combined_sample_fid/dec'][()][ind_selr]
        rand_z_all = cat['randoms/redmagic/combined_sample_fid/z'][()][ind_selr]

    datapoint_ra_all, datapoint_dec_all = data_coord_cov(datapoint_ra_all, datapoint_dec_all,icrs2gal=True)
    rand_ra_all, rand_dec_all = data_coord_cov(rand_ra_all, rand_dec_all,icrs2gal=True)
    rm_ra_all, rm_dec_all = data_coord_cov(rm_ra_all, rm_dec_all,icrs2gal=True)

    # Restrict to datapoint selection
    selection_z = np.where((datapoint_z_all > zmin) & (datapoint_z_all < zmax))[0]
    print("num in selection = ", selection_z.shape)    
    selection_f = selection_z
    
    datapoint_ra = datapoint_ra_all[selection_f]
    datapoint_dec = datapoint_dec_all[selection_f]
    datapoint_z = datapoint_z_all[selection_f]
    datapoint_radius = datapoint_radius_all[selection_f]
    datapoint_weight = datapoint_weight_all[selection_f]

#     theta_datapoint = theta_datapoint_all[selection_f]
#     costheta_datapoint = np.cos(theta_datapoint)
#     phi_datapoint = phi_datapoint_all[selection_f]

    # Restrict to datapoint selection
    selection_z = np.where((rm_z_all > zmin) & (rm_z_all < zmax))[0]
    print("num in selection = ", selection_z.shape)    
    selection_f = selection_z    
    rm_ra = rm_ra_all[selection_f]
    rm_dec = rm_dec_all[selection_f]
    rm_z = rm_z_all[selection_f]
    rm_weight = rm_weight_all[selection_f]
        
    ndatapoint = len(datapoint_ra)

    nsel_rand = 0
    if nsel_rand > 0:
        sel_rand = np.unique(np.random.randint(0, ndatapoint, nsel_rand))
        datapoint_ra, datapoint_dec, datapoint_z, datapoint_radius, datapoint_weight = \
            datapoint_ra[sel_rand], datapoint_dec[sel_rand], datapoint_z[sel_rand], datapoint_radius[sel_rand], datapoint_weight[sel_rand]
        ndatapoint = len(datapoint_ra)

    selection_z_rand = np.where((rand_z_all > zmin) & (rand_z_all < zmax))[0]
    rand_theta_all, rand_phi_all = eq2ang(rand_ra_all, rand_dec_all)
#     ind_rand = hp.ang2pix(nside, rand_theta_all, rand_phi_all)
#     int_ind_rand = np.in1d(ind_rand, ind_masked)
#     selection_mask_rand = np.where(int_ind_rand == False)[0]
#     selection_rand = np.intersect1d(selection_z_rand, selection_mask_rand)
    selection_rand = selection_z_rand
#     if len(selection_rand) < 150*ndatapoint:
#         rand_theta, rand_phi = rand_theta_all[selection_rand], rand_phi_all[selection_rand]
#         rand_ra, rand_dec = ang2eq(rand_theta, rand_phi)
#         rand_z = rand_z_all[selection_rand]
#     else:
#         rand_theta_all, rand_phi_all = rand_theta_all[selection_rand], rand_phi_all[selection_rand]
#         rand_z_all = rand_z_all[selection_rand]
#         rand_index_rand = np.unique(np.random.randint(0, len(rand_theta_all), 150 * ndatapoint))
#         rand_theta, rand_phi = rand_theta_all[rand_index_rand], rand_phi_all[rand_index_rand]
#         rand_z = rand_z_all[rand_index_rand]
#         rand_ra, rand_dec = ang2eq(rand_theta, rand_phi)

    rand_theta, rand_phi = rand_theta_all[selection_rand], rand_phi_all[selection_rand]
    rand_ra, rand_dec = ang2eq(rand_theta, rand_phi)
    rand_z = rand_z_all[selection_rand]


    nrand = len(rand_ra)
    nrm = len(rm_ra)

#     rand_theta, rand_phi = eq2ang(rand_ra, rand_dec)

#     mask = np.zeros(hp.nside2npix(nside))

#     mask[ind_datapoints] = 1.

    print('Number of datapoints and Random points are:')
    print(ndatapoint,nrand,nrm)

    do_jk = True
#     nside_ymap = nside
    put_weights_datapoints = True
    do_randy_sub = True

#     index_rand = hp.ang2pix(nside_ymap, rand_theta, rand_phi)
#     pix_area = hp.nside2pixarea(nside_ymap, degrees=True)
#     catalog_area = (len(np.unique(index_rand))) * pix_area
#     print('total data points : ' + str(len(datapoint_ra)))
#     print('total random points : ' + str(len(rand_ra)))
#     print('catalog area sq deg : ' + str(catalog_area))
    from astropy.cosmology import FlatLambdaCDM
    # cosmo = FlatLambdaCDM(H0=100, Om0=Omega_m)
    cosmo_params_dict = {'flat': True, 'H0': 70.0, 'Om0': 0.283705720011, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
    gnf = general_funcs(cosmo_params_dict)
    chi_array = np.linspace(0, 4000, 50000)
    z_array = np.zeros(len(chi_array))
    for j in range(len(z_array)):
        z_array[j] = gnf.get_z_from_chi(chi_array[j])
    z_interp = interpolate.interp1d(chi_array, z_array)
    th_thv_min = 0.1
    th_thv_max = 2.5
#     npix_ymap = len(ymap_truth)
#     nside_ymap = hp.npix2nside(npix_ymap)
#     pix_theta, pix_phi = hp.pix2ang(nside_ymap, np.arange(npix_ymap))
#     pix_ra, pix_dec = ang2eq(pix_theta, pix_phi)
#     ytruth_cat = treecorr.Catalog(ra=pix_ra, dec=pix_dec, k=ymap_truth,ra_units='degrees', dec_units='degrees')
    nrad = 13
    xi_all_data = np.zeros((len(datapoint_ra), nrad))
    theta_all_data = np.zeros((len(datapoint_ra), nrad))
    th_thv_all_data = np.zeros((len(datapoint_ra), nrad))
    thetav_data = np.zeros(len(datapoint_ra))
    Dcomv_data = np.zeros(len(datapoint_ra))
    

    if do_jk:
        datapoint_radec = np.transpose([datapoint_ra, datapoint_dec])
        jkobj_map = kmeans_radec.kmeans_sample(np.transpose([datapoint_ra, datapoint_dec]), njk)
        datapoint_jk = jkobj_map.find_nearest(datapoint_radec)
        
        fac_mult = 5
#         fac_multy = 9
        for jv in range(len(datapoint_ra)):
            ra_jv, dec_jv, z_jv, r_jv, w_jv = datapoint_ra[jv], datapoint_dec[jv], datapoint_z[jv], datapoint_radius[jv], datapoint_weight[jv]
            Dcom_jv = gnf.get_Dcom(z_jv)
            zmin_sel_jv = z_interp(Dcom_jv - fac_mult*r_jv)
            zmax_sel_jv = z_interp(Dcom_jv + fac_mult*r_jv)
            thv_jv = (r_jv/Dcom_jv)*(180.*60./np.pi)
            thv_jv_deg = (r_jv/Dcom_jv)*(180./np.pi)
            
            rand_ind_sel_void = np.where((rand_ra > ra_jv - fac_mult*thv_jv_deg) & (rand_ra < ra_jv + fac_mult*thv_jv_deg) & \
             (rand_dec > dec_jv - fac_mult*thv_jv_deg) & (rand_dec < dec_jv + fac_mult*thv_jv_deg) &  \
             (rand_z > zmin_sel_jv) & (rand_z < zmax_sel_jv))[0]
            rand_ra_jv, rand_dec_jv, rand_z_jv = rand_ra[rand_ind_sel_void], rand_dec[rand_ind_sel_void], rand_z[rand_ind_sel_void]
            
            rm_ind_sel_void = np.where((rm_ra > ra_jv - fac_mult*thv_jv_deg) & (rm_ra < ra_jv + fac_mult*thv_jv_deg) & \
             (rm_dec > dec_jv - fac_mult*thv_jv_deg) & (rm_dec < dec_jv + fac_mult*thv_jv_deg) &  \
             (rm_z > zmin_sel_jv) & (rm_z < zmax_sel_jv))[0]
            rm_ra_jv, rm_dec_jv, rm_z_jv, rm_w_jv = rm_ra[rm_ind_sel_void], rm_dec[rm_ind_sel_void], rm_z[rm_ind_sel_void], rm_weight[rm_ind_sel_void]
            
            minrad = th_thv_min * thv_jv
            maxrad = th_thv_max * thv_jv

            datapoint_cat = treecorr.Catalog(ra=[ra_jv], dec=[dec_jv], w=[w_jv], ra_units='degrees',
                                                dec_units='degrees')
            rand_cat = treecorr.Catalog(ra=rand_ra_jv, dec=rand_dec_jv, ra_units='degrees', dec_units='degrees')
            rm_cat = treecorr.Catalog(ra=rm_ra_jv, dec=rm_dec_jv, w=rm_w_jv, ra_units='degrees', dec_units='degrees')


#             print('doing the void:' + str(jv+1))
#             print('random points here:' + str(len(rand_ra_jv)))
            # perform correlation measurement
            vgtruth = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad,  sep_units='arcmin', verbose=0)
            vrtruth = treecorr.NNCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad,  sep_units='arcmin', verbose=0)
    
            vgtruth.process(datapoint_cat, rm_cat)
            vrtruth.process(datapoint_cat, rand_cat)
            
            vg_np_norm = vgtruth.weight * 1. / (1. * len(rm_ra_jv))
            vr_np_norm = vrtruth.weight * 1. / (1. * len(rand_ra_jv))
            
            xi_jv = (vg_np_norm/vr_np_norm) - 1
            xi_all_data[jv,:] = xi_jv
            theta_all_data[jv,:] = vgtruth.rnom
            thetav_data[jv] = thv_jv
            th_thv_all_data[jv,:] = vgtruth.rnom/thv_jv
            Dcomv_data[jv] = Dcom_jv
            
            if np.mod(jv,10) == 0:
                print('doing the void:' + str(jv+1))
                print('random points here:' + str(len(rand_ra_jv)))
                print('redmagic points here:' + str(len(rm_ra_jv)))
                print('xi here:', xi_jv)



#     import ipdb; ipdb.set_trace() # BREAKPOINT

    # Save output
    save_data = {'ra':datapoint_ra, 'dec':datapoint_dec, 'z':datapoint_z, 'rv':datapoint_radius,'jk':datapoint_jk,
    'xi_all':xi_all_data, 'theta_thv_all':th_thv_all_data, 'Dcom_all':Dcomv_data, 'thv_data':thetav_data, 'theta_all_data':theta_all_data}

    data_output_dir = '/global/project/projectdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/'
    file_suffix_save = '_cat_' + str('void') + '_z_' + str(zmin) + '_' + str(zmax) + '_R_' + str(rmin) + '_' + str(rmax) + '_' + 'dojk_' + str(do_jk) + '_njk_' + str(njk)  + '_' + 'desy3' + '_w' + str(int(put_weights_datapoints))
    filename = data_output_dir + 'dy/void_rm_' + '_' + file_suffix_save + '_loghres.pk'

    pk.dump(save_data, open(filename, "wb"), protocol = 2)

    pdb.set_trace()
