import sys, os
sys.path.append(os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/src/cosmosis_code/2pt')
import twopoint
import pickle as pk
from astropy.io import fits as pf
from astropy.io import fits
import scipy as sp
# load template *****
fiducial = pf.open('/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/data/fiducial_maglim_cov_sourcesv040.fits')
# load n(z) ******
import numpy as np
from scipy.signal import savgol_filter
nzsamp = np.load('/global/cscratch1/sd/alexalar/desy3data/Nz_samples/v0.50/3sdir_fid_zsamplefid//nz_samples.npy')
nzsamp = savgol_filter(nzsamp,5,2,axis=2)
nz_tot=np.mean(nzsamp,axis=0)
z_samp = np.load('/global/cscratch1/sd/alexalar/desy3data/Nz_samples/sv_sn_test1///bin_centers.npy')
z_edge_samp = np.load('/global/cscratch1/sd/alexalar/desy3data/Nz_samples/sv_sn_test1///bin_edges.npy')
# make n(z)
Nz = []
from scipy.interpolate import interp1d
for i in range(4):
    f = interp1d(z_samp,nz_tot[i])
    nz_e = np.zeros(len(fiducial[6].data['Z_MID']))
    mask = fiducial[6].data['Z_MID'] > z_samp[0]
    nz_e[mask] = f(fiducial[6].data['Z_MID'][mask])
    Nz.append(nz_e)
nz_full  = twopoint.NumberDensity("nz_source", fiducial['nz_source'].data['Z_LOW'], fiducial['nz_source'].data['Z_MID'], fiducial['nz_source'].data['Z_HIGH'], Nz)

zmin_bins = np.array([0.20, 0.40, 0.55, 0.70, 0.85, 0.95])
zmax_bins = np.array([0.40, 0.55, 0.70, 0.85, 0.95, 1.05 ])

fname = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/MICE_data/mice_maglim_data.fits'
df = fits.open(fname)
zcgal_all = df[1].data['z_cgal']
# datapoint_z_all = df[1].data['z_dnf_mean_sof']
datapoint_z_all = df[1].data['z_cgal']
zc_edges_all = np.hstack((fiducial['nz_lens'].data['Z_LOW'],fiducial['nz_lens'].data['Z_HIGH'][-1]))
Nz = []
from scipy.interpolate import interp1d
for i in range(len(zmin_bins)):
    ind_bin = np.where((datapoint_z_all > zmin_bins[i]) & (datapoint_z_all < zmax_bins[i]))[0]
    zc_bin = zcgal_all[ind_bin]
    
    nz_hist, hist_edges = np.histogram(zc_bin, bins=zc_edges_all)
    nz_hist_norm = nz_hist / (sp.integrate.simps(nz_hist,fiducial['nz_lens'].data['Z_MID']))
    Nz.append(nz_hist_norm)
nz_LENS  = twopoint.NumberDensity("nz_lens", fiducial['nz_lens'].data['Z_LOW'], fiducial['nz_lens'].data['Z_MID'], fiducial['nz_lens'].data['Z_HIGH'], Nz)
# this is to use the same theta as in the theory code - 
import math

theta = np.array([0.61323842,  0.88152532,  1.27314055,  1.83433814,  2.64195147,
        3.8078711 ,  5.484779  ,  7.90280216, 11.38533148, 16.40379712,
       23.63340958, 34.04841214])

import pickle
def load_obj(name):
        try:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)#, encoding='latin1')
        except:
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f, encoding='latin1')
from enum34 import Enum
class Types(Enum):
    """
    This is an enumeration - a list of possible values with names and code values
    that we can use in FITS headers.
    It enumerates the different quantities that a two-point measurement can correlate.
    For example, CMB T,E,B, lensing E and B, galaxy position and magnification.
    It specifies the quantity and whether we are in Fourier space or real space.
    One special case is xi_{-} and xi_{+} in galaxy shear.  These values are already
    correlations of combinations of E and B modes. We denote this as xi_{++} and xi_{--} here.
    """
    galaxy_position_fourier = "GPF"
    galaxy_shear_emode_fourier = "GEF"
    galaxy_shear_bmode_fourier = "GBF"
    galaxy_position_real = "GPR"
    galaxy_shear_plus_real = "G+R"
    galaxy_shear_minus_real = "G-R"
    cmb_kappa_real = "CKR"
    compton = 'compton'
    @classmethod
    def lookup(cls, value):
        for T in cls:
            if T.value == value:
                return T



store_gg = dict()
store_tsz = dict()
count =0

save_dir = os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/src/results/'
cat_tocorr = 'maglim'
do_jk = True
njk = 500
put_weights_datapoints = False
nside_ymap = 4096
mask_type = 'act'
minz_all = np.array([0.20, 0.40, 0.55, 0.7, 0.85, 0.95])
maxz_all = np.array([0.40, 0.55, 0.7, 0.85, 0.95, 1.05 ])
do_gg = 1
labels = ['Bin1', 'Bin2', 'Bin3', 'Bin4','Bin5', 'Bin6']

linestyles = ['-.','-','--',':','-.','-','--']

factor = 1.



filenames = []
for j in range(len(minz_all)):
    minz = minz_all[j]
    maxz = maxz_all[j]
    # file_suffix_save = '_cat_' + str(cat_tocorr) + '_z_' + str(minz) + '_' + str(maxz) + '_' + 'dojk_' + str(do_jk) + '_njk_' + str(njk)  + '_' + 'desy3' + '_w' + str(int(put_weights_datapoints)) + '_beam' + str(2.4)   

    # if do_gg:
    #     filename = save_dir + 'dy/dy_dd_' + 'MICEy'  + '_' + 'nobeam' + '_nside' + str(nside_ymap) + '_mask_' + str(mask_type) + '_' + file_suffix_save + '_ns4096.pk'
    # else:
    #     filename = save_dir + 'dy/dy_' + 'MICEy'  + '_' + 'nobeam' + '_nside' + str(nside_ymap) + '_mask_' + str(mask_type) + '_' + file_suffix_save + '_ns4096.pk'
    # filename = save_dir + 'dy/dy_dd_MICEy_nobeam_nside4096_mask_act__cat_maglim_z_' + str(minz) + '_' + str(maxz) + '_dojk_True_njk_500_desy3_w1_beam0_ns4096_v16Jan21.pk'
    filename = save_dir + 'dy/dy_dd_MICEy_nobeam_nside4096_mask_act__cat_maglim_z_' + str(minz) + '_' + str(maxz) + '_dojk_True_njk_500_desy3_w1_beam0_ns4096_v22Feb21_truez.pk'
    filenames.append(filename)

nbins = len(zmin_bins)
ntheta = len(theta)
n_tot = nbins * ntheta * 2
cov_total = np.zeros((n_tot,n_tot)) 

for l in range(nbins): 
    print('opening bin ' + str(l+1))
    filename = filenames[l]
    try:
        haloydata = pk.load(open(filename, "rb"))
    except:
        haloydata = pk.load(open(filename, "rb"),encoding='latin1')

    ggcorr = haloydata['xi_gg']
    store_gg["{0} {1}".format(l,l)] = ggcorr
    # dytruth = haloydata['dytruth']
    # randytruth = haloydata['randytruth']
    cov_totalj = haloydata['cov_total']
    xi_dytruth = haloydata['xi_dy']
    store_tsz["{0}".format(l)] = dict()
    store_tsz["{0}".format(l)]["tsz"] = xi_dytruth
    print(cov_totalj.shape)
    cov_total[l*ntheta:(l+1)*ntheta,l*ntheta:(l+1)*ntheta] = cov_totalj[0:12,0:12]
    cov_total[(l+nbins)*ntheta:(l+1+nbins)*ntheta,(l+nbins)*ntheta:(l+1+nbins)*ntheta] = cov_totalj[12:,12:]
    cov_total[l*ntheta:(l+1)*ntheta,(l+nbins)*ntheta:(l+1+nbins)*ntheta] = cov_totalj[0:12,:][:,12:]
    cov_total[(l+nbins)*ntheta:(l+1+nbins)*ntheta,l*ntheta:(l+1)*ntheta] = cov_totalj[12:,:][:,0:12]

angular_bins = len(theta)
tsz_dv = np.zeros(angular_bins*nbins)
tsz_bin1 = np.zeros(angular_bins*nbins) 
tsz_bin2 = np.zeros(angular_bins*nbins) 
tsz_angular_bin = np.zeros(angular_bins*nbins) 
tsz_angle = np.zeros(angular_bins*nbins) 
count = 0
for l in range(nbins):
    tsz_dv[count*angular_bins:(count+1)*angular_bins] = store_tsz["{0}".format(l)]["tsz"]
    tsz_bin1[count*angular_bins:(count+1)*angular_bins] = l+1
    tsz_bin2[count*angular_bins:(count+1)*angular_bins] = l+1
    tsz_angular_bin[count*angular_bins:(count+1)*angular_bins] = np.arange(len(theta))
    tsz_angle[count*angular_bins:(count+1)*angular_bins] = theta
    count +=1

angular_bins = len(theta)
gg_dv = np.zeros(angular_bins*nbins)
gg_bin1 = np.zeros(angular_bins*nbins) 
gg_bin2 = np.zeros(angular_bins*nbins) 
gg_angular_bin = np.zeros(angular_bins*nbins) 
gg_angle = np.zeros(angular_bins*nbins) 
count = 0
for l in range(nbins):
    gg_dv[count*angular_bins:(count+1)*angular_bins] = store_gg["{0} {1}".format(l,l)]
    gg_bin1[count*angular_bins:(count+1)*angular_bins] = l+1
    gg_bin2[count*angular_bins:(count+1)*angular_bins] = l+1
    gg_angular_bin[count*angular_bins:(count+1)*angular_bins] = np.arange(len(theta))
    gg_angle[count*angular_bins:(count+1)*angular_bins] = theta
    count +=1


wtheta = twopoint.SpectrumMeasurement('wtheta', (gg_bin1, gg_bin2),
                                                     (twopoint.Types.galaxy_position_real,
                                                      twopoint.Types.galaxy_position_real),
                                                     ['no_nz', 'no_nz'], 'SAMPLE', gg_angular_bin, gg_dv,angle=gg_angle, angle_unit='arcmin')
tsz_m = twopoint.SpectrumMeasurement('compton_galaxy', (tsz_bin1, tsz_bin2),
                                                     (twopoint.Types.galaxy_position_real,Types.compton),
                                                     ['no_nz', 'no_nz'], 'SAMPLE', tsz_angular_bin, tsz_dv,angle=tsz_angle, angle_unit='arcmin')
print ('done')


obj = twopoint.TwoPointFile([wtheta,tsz_m], [nz_full,nz_LENS], windows=None, covmat_info=None)
names = [s.name for s in obj.spectra]
lengths = [len(s) for s in obj.spectra]
n = sum(lengths)
obj.covmat_info = twopoint.CovarianceMatrixInfo("COVMAT", names, lengths, cov_total)

path_results = os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/src/results/'

import os
try:
    os.remove(path_results + 'Maglim_ACT_MICE_actarea_JKcov_v22Feb21_truez.fits')
except:
    pass
obj.to_fits(path_results + 'Maglim_ACT_MICE_actarea_JKcov_v22Feb21_truez.fits') 




# path_results = os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/src/results/'
# dd = pickle.load(open(path_results +'DV_obj_temp_all_MICE_maglim_gg_gy_zevhod_actbeam_wcov_corrnbar.pk','rb'),fix_imports=True,encoding='latin')
# ind_th_sel = np.where((dd.fftcovtot_dict['gy_gy']['theta'] > 0.55) & (dd.fftcovtot_dict['gy_gy']['theta'] < 157.0))[0]
# n_tot = angular_bins*nbins+angular_bins*nbins
# cov_theory = np.zeros((n_tot,n_tot))     
# comb = []
# for l in range(nbins):
#     comb.append([l,l,'gg','gg'])
# for l in range(nbins):
#     comb.append([l,-1,'gg','gy'])
# for i,c1 in enumerate(comb):
#     for j,c2 in enumerate(comb):
#         count = False
#         c_mute = np.zeros(((angular_bins),(angular_bins)))
#         try:
#             c_mute = dd.fftcovtot_dict['{0}_{1}'.format(c1[3],c2[3])]['bin_' + str(c1[1]+1) + '_' + str(c1[0]+1) + '_' + str(c2[1]+1) + '_' + str(c2[0]+1)][ind_th_sel, :][:,ind_th_sel]
#             count = True
#         except:
#             pass
#         try:
#             c_mute = dd.fftcovtot_dict['{0}_{1}'.format(c2[3],c1[3])]['bin_' + str(c2[1]+1) + '_' + str(c2[0]+1) + '_' + str(c1[1]+1) + '_' + str(c1[0]+1)][ind_th_sel, :][:,ind_th_sel]
#             count = True
#         except:
#             pass
#         try:
#             c_mute = dd.fftcovtot_dict['{0}_{1}'.format(c1[3],c2[3])]['bin_' + str(c1[0]+1) + '_' + str(c1[1]+1) + '_' + str(c2[0]+1) + '_' + str(c2[1]+1)][ind_th_sel, :][:,ind_th_sel]
#             count = True
#         except:
#             pass
#         try:
#             c_mute = dd.fftcovtot_dict['{0}_{1}'.format(c2[3],c1[3])]['bin_' + str(c2[0]+1) + '_' + str(c2[1]+1) + '_' + str(c1[0]+1) + '_' + str(c1[1]+1)][ind_th_sel, :][:,ind_th_sel]
#             count = True
#         except:
#             pass
#         try:
#             c_mute = dd.fftcovtot_dict['{0}_{1}'.format(c2[3],c1[3])]['bin_' + str(c1[0]+1) + '_' + str(c1[1]+1) + '_' + str(c2[0]+1) + '_' + str(c2[1]+1)][ind_th_sel, :][:,ind_th_sel]
#             count = True
#         except:
#             pass
#         try:
#             c_mute = dd.fftcovtot_dict['{0}_{1}'.format(c1[3],c2[3])]['bin_' + str(c1[0]+1) + '_' + str(c1[1]+1) + '_' + str(c2[1]+1) + '_' + str(c2[0]+1)][ind_th_sel, :][:,ind_th_sel]
#             count = True
#         except:
#             pass
#         try:
#             c_mute = dd.fftcovtot_dict['{0}_{1}'.format(c2[3],c1[3])]['bin_' + str(c1[1]+1) + '_' + str(c1[0]+1) + '_' + str(c2[0]+1) + '_' + str(c2[1]+1)][ind_th_sel, :][:,ind_th_sel]
#             count = True
#         except:
#             pass
#         try:
#             c_mute = dd.fftcovtot_dict['{0}_{1}'.format(c2[3],c1[3])]['bin_' + str(c2[0]+1) + '_' + str(c2[1]+1) + '_' + str(c1[1]+1) + '_' + str(c1[0]+1)][ind_th_sel, :][:,ind_th_sel]
#             count = True
#         except:
#             pass
#         cov_theory[i*angular_bins:(i+1)*angular_bins,j*angular_bins:(j+1)*angular_bins] = c_mute
# obj = twopoint.TwoPointFile([wtheta,tsz_m], [nz_full,nz_LENS], windows=None, covmat_info=None)
# names = [s.name for s in obj.spectra]
# lengths = [len(s) for s in obj.spectra]
# n = sum(lengths)
# obj.covmat_info = twopoint.CovarianceMatrixInfo("COVMAT", names, lengths, cov_theory)
# import os
# try:
#     os.remove(path_results + 'Maglim_ACT_MICE_actarea_theorycov_corrnbar.fits')
# except:
#     pass
# obj.to_fits(path_results + 'Maglim_ACT_MICE_actarea_theorycov_corrnbar.fits') 

