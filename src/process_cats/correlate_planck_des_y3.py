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
    parser.add_argument('--cat', required=True, type=str, help='Cat type')
    parser.add_argument('--njk', required=True, type=int, help='Cat type')
    args_all = parser.parse_args()
    return args_all


if __name__ == "__main__":
  args = parse_arguments()

  # redmagic, maglim
  cat_tocorr = args.cat
  njk = args.njk
  # cat_tocorr = 'redmagic'
  # None, cib, cmb
  # deproj = args.dp
  true_y_file = '/global/cfs/cdirs/des/shivamp/actxdes/data_set/planck_data/pl2015/nilc_ymaps.fits'

  if cat_tocorr == 'maglim':
    minz = 0.2
    maxz = 1.05
  if cat_tocorr == 'redmagic':
    minz = 0.15
    maxz = 0.9

  ymap_truth = hp.read_map(true_y_file)

  fname = '/project/projectdirs/des/www/y3_cats/Y3_mastercat_12_3_19.h5'


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

  datapoint_ra_all, datapoint_dec_all = data_coord_cov(datapoint_ra_all, datapoint_dec_all,icrs2gal=True)
  rand_ra_all, rand_dec_all = data_coord_cov(rand_ra_all, rand_dec_all,icrs2gal=True)



  # Restrict to datapoint selection
  selection_z = np.where((datapoint_z_all > minz) & (datapoint_z_all < maxz))[0]
  print("num in selection = ", selection_z.shape)
  # import pdb; pdb.set_trace()

  nside = hp.npix2nside(len(ymap_truth))
  mask_final = []
  gal_mask_input_orig = hp.read_map(true_y_file,field=3)

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

  if len(selection_rand) < 15* ndatapoint:
      rand_theta, rand_phi = rand_theta_all[selection_rand], rand_phi_all[selection_rand]
      rand_ra, rand_dec = ang2eq(rand_theta, rand_phi)
      # nrand = len(rand_ra)
  else:
      rand_theta_all, rand_phi_all = rand_theta_all[selection_rand], rand_phi_all[selection_rand]
      rand_index_rand = np.unique(np.random.randint(0, len(rand_theta_all), 15 * ndatapoint))
      rand_theta, rand_phi = rand_theta_all[rand_index_rand], rand_phi_all[rand_index_rand]
      rand_ra, rand_dec = ang2eq(rand_theta, rand_phi)
      # nrand = len(rand_ra)

  nrand = len(rand_ra)

  rand_theta, rand_phi = eq2ang(rand_ra, rand_dec)

  mask = np.zeros(hp.nside2npix(nside))

  mask[ind_datapoints] = 1.

  print(ndatapoint,nrand)

  do_jk = True
  nside_ymap = nside
  put_weights_datapoints = True
  do_randy_sub = True
  

  if do_jk:

      ind_sel = np.random.randint(0,len(datapoint_ra),200000)
      datapoint_radec = np.transpose([datapoint_ra, datapoint_dec])
      jkobj_map = kmeans_radec.kmeans_sample(np.transpose([datapoint_ra[ind_sel], datapoint_dec[ind_sel]]), njk)

      datapoint_jk = jkobj_map.find_nearest(datapoint_radec)

      if do_randy_sub:        
        if len(rand_ra) > 10000000:
          rand_jk = np.zeros_like(rand_ra)
          nsplit = 30
          ind_binh_split = np.array_split(np.arange(len(rand_ra)),nsplit)
          for js in range(nsplit):
              print('processing split' + str(js+1))
              ind_binh_js = ind_binh_split[js]
              rand_jk[ind_binh_js] = jkobj_map.find_nearest(np.array([rand_ra[ind_binh_js], rand_dec[ind_binh_js]]).T)
        else:
          rand_radec = np.transpose([rand_ra, rand_dec])
          rand_jk = jkobj_map.find_nearest(rand_radec)

  index_rand = hp.ang2pix(nside_ymap, rand_theta, rand_phi)
  pix_area = hp.nside2pixarea(nside_ymap, degrees=True)
  catalog_area = (len(np.unique(index_rand))) * pix_area

  zmin = 0.0
  zmax = 1.2
  nzbins_total = 120

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
  zmean_j = (minz + maxz)/2.

  print('total data points : ' + str(len(datapoint_ra)))
  print('total random points : ' + str(len(rand_ra)))
  print('catalog area sq deg : ' + str(catalog_area))
  print('bin zmean : ' + str(zmean_j))

  min_r = 0.1
  max_r = 80.

  minrad = 0.1
  maxrad = 250.0

  nrad = 15

  Omega_m = 0.283705720011
  from astropy.cosmology import FlatLambdaCDM
  cosmo = FlatLambdaCDM(H0=100, Om0=Omega_m)
  Dcom_z = (cosmo.comoving_distance(zmean_j)).value

  # minrad = (min_r / Dcom_z)*(180./np.pi)*(60.)
  # maxrad = (max_r / Dcom_z)*(180./np.pi)*(60.)


  if put_weights_datapoints:

      datapoint_weight = datapoint_weight_all[selection_f]

      datapoint_cat = treecorr.Catalog(ra=datapoint_ra, dec=datapoint_dec, w=datapoint_weight, ra_units='degrees',
                                       dec_units='degrees')
  else:
      datapoint_cat = treecorr.Catalog(ra=datapoint_ra, dec=datapoint_dec, ra_units='degrees', dec_units='degrees')
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
  dytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0)
  randytruth = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin', verbose=0)

  dytruth.process(datapoint_cat, ytruth_cat)
  randytruth.process(rand_cat, ytruth_cat)


  np_dytruth_full = dytruth.npairs
  xi_dytruth_full = dytruth.xi

  if do_randy_sub:
      np_randytruth_full = randytruth.npairs
      xi_randytruth_full = randytruth.xi

  xi_dytruth_big_all = np.zeros((njk, nrad))
  xi_randytruth_big_all = np.zeros((njk, nrad))

  for j in range(njk):
      print('doing jk ' + str(j))
      datapoint_ind_small = np.where(datapoint_jk == j)[0]
      datapoint_cat_small = treecorr.Catalog(ra=datapoint_ra[datapoint_ind_small],
                                             dec=datapoint_dec[datapoint_ind_small], ra_units='degrees',
                                             dec_units='degrees')


      dytruth_small = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin',
                                             verbose=0)
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

          randytruth_small = treecorr.NKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, sep_units='arcmin',
                                                    verbose=0)

          randytruth_small.process(rand_cat_small, ytruth_cat)

          np_randytruth_small = randytruth_small.npairs
          xi_randytruth_small = randytruth_small.xi

          np_randytruth_big = np_randytruth_full - np_randytruth_small

          xi_randytruth_big_all[j, :] = (
                                                    xi_randytruth_full * np_randytruth_full - np_randytruth_small * xi_randytruth_small) / np_randytruth_big





  save_data = { 'dytruth': dytruth,
               'randytruth': randytruth,
               'xi_dytruth_big_all': xi_dytruth_big_all,
               'xi_randytruth_big_all': xi_randytruth_big_all, 'minz': minz,
               'maxz': maxz,
               'do_jk': do_jk, 'njk': njk, 'ndatapoint': len(datapoint_ra),
               'nrand': len(rand_ra), 'area_datapoint_sqdeg': catalog_area,'zmean':zmean_j,'Dcom_z':Dcom_z}



  # Save output

  data_output_dir = '/global/project/projectdirs/des/shivamp/cosmosis/ACTxDESY3/src/results/'

  file_suffix_save = '_cat_' + str(cat_tocorr) + '_z_' + str(minz) + '_' + str(maxz) + '_' + 'dojk_' + str(do_jk) + '_njk_' + str(njk)  + '_' + 'desy3' + '_w' + str(int(put_weights_datapoints))

  filename = data_output_dir + 'dy/dy_' + 'planck_' + 'fwhm_10arcmin' + '_nside' + str(nside_ymap) + '_' + file_suffix_save + '.pk'

  pk.dump(save_data, open(filename, "wb"), protocol = 2)


  pdb.set_trace()










































