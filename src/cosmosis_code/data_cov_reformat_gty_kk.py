import sys, platform, os
sys.path.insert(0, '/global/cfs/cdirs/des/shivamp/cosmosis/ACTxDESY3/src/cosmosis_code/')
os.environ['COSMOSIS_SRC_DIR'] = '/global/cfs/cdirs/des/shivamp/cosmosis'
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
from astropy.io import fits
import time
import math
from scipy import interpolate
import pickle as pk
import dill
import sys, os




try:
    df = pk.load(open('/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/data/results_25_6_20.pkl','rb'))
except:
    df = pk.load(open('/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/data/results_25_6_20.pkl', 'rb'),
                 encoding='latin1')

data_year='y3'
cov_type = 'theory'

ntheta = len(df['Yshear_'+data_year][str(0)]['theta'])
bins_array = [1,2,3,4]
nbins = len([*df['Yshear_'+data_year].keys()])

Cl_shearshear = np.zeros(ntheta*nbins)
Cl_sheary = np.zeros(ntheta*nbins)

cov_shearshear = np.zeros((ntheta*nbins,ntheta*nbins))
cov_sheary = np.zeros((ntheta*nbins,ntheta*nbins))
cov_shearshear_sheary = np.zeros((ntheta*nbins,ntheta*nbins))

if cov_type == 'theory':
    df_name = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/DV_obj_temp_gty_plancky3_beamed_wmeadv4_20_mod_M8_17_nM64_nz256_imead1_nodndm.pk'
    DV = dill.load(open(df_name, 'rb'))
    ind_th_sel = np.where((DV.fftcovtot_dict['ky_ky']['theta'] > 2.5) & (DV.fftcovtot_dict['ky_ky']['theta'] < 250.0))[
        0]
    print(DV.fftcovtot_dict['ky_ky']['theta'][ind_th_sel])
    print(df['Yshear_'+data_year][str(0)]['theta'] * 180. / np.pi)

theta_sheary = np.array([])
theta_shearshear = np.array([])

bin_combs_kk_cs = np.array([[1,1],[1,2],[1,3],[1,4],[2,2],[2,3],[2,4],[3,3],[3,4],[4,4]])
bin_combs_gty = np.array([[1,0],[2,0],[3,0],[4,0]])

for j1 in range(len(bins_array)):

    kk_g = df['shearshear_' + data_year][str(j1) + '_' + str(j1)]['xip']
    kk_th_g = df['shearshear_' + data_year][str(j1) + '_' + str(j1)]['theta'] * 180. / np.pi
#    kk_err = df['shearshear_' + data_year][str(j1) + '_' + str(j1)]['err']
    Cl_shearshear[ntheta * (j1):ntheta * (j1 + 1)] = kk_g
    for j2 in range(len(bins_array)):
        try:
            cov_kk_binj2 = DV.fftcovtot_dict['kk_kk']['bin_' + str(j1+1) + '_' + str(j1+1) + '_' + str(j2+1) + '_' + str(j2+1)][ind_th_sel, :][:,
                                                                        ind_th_sel]

        except:
            cov_kk_binj2 = DV.fftcovtot_dict['kk_kk']['bin_' + str(j2+1) + '_' + str(j2+1) + '_' + str(j1+1) + '_' + str(j1+1)][ind_th_sel, :][:,
                                                                        ind_th_sel]

#    kk_cov = df['shearshear_' + data_year][str(j1) + '_' + str(j1)]['cov']
        cov_shearshear[ntheta * (j1):ntheta * (j1 + 1), ntheta * (j2):ntheta * (j2 + 1)] = cov_kk_binj2 

    if len(theta_shearshear) == 0:
        theta_shearshear = df['shearshear_' + data_year][str(j1) + '_' + str(j1)]['theta'] * 180. / np.pi
    else:
        theta_shearshear = np.vstack(
            (theta_shearshear, df['shearshear_' + data_year][str(j1) + '_' + str(j1)]['theta'] * 180. / np.pi))
    
    if cov_type == 'jk':
        yk_cov = df['Yshear_' + data_year][str(j1)]['cov']
        cov_sheary[ntheta * (j1):ntheta * (j1 + 1), ntheta * (j1):ntheta * (j1 + 1)] = yk_cov
    yk_g = df['Yshear_' + data_year][str(j1)]['xip']
    yk_th_g = df['Yshear_' + data_year][str(j1)]['theta'] * 180. / np.pi
#    yk_err = df['Yshear_' + data_year][str(j1)]['err']
    Cl_sheary[ntheta * (j1):ntheta * (j1 + 1)] = yk_g

    if len(theta_sheary) == 0:
        theta_sheary = df['Yshear_' + data_year][str(j1)]['theta'] * 180. / np.pi
    else:
        theta_sheary = np.vstack((theta_sheary, df['Yshear_' + data_year][str(j1)]['theta'] * 180. / np.pi))
    if cov_type == 'theory':
        for j2 in range(len(bins_array)):
            cov_sheary[ntheta * j1:ntheta * (j1 + 1), ntheta * j2:ntheta * (j2 + 1)] = DV.fftcovtot_dict['gty_gty'][
                                                                            'bin_' + str(j1 + 1) + '_0_' + str(
                                                                                j2 + 1) + '_0'][ind_th_sel, :][:,
                                                                        ind_th_sel]

    if cov_type == 'theory':
        for j2 in range(len(bins_array)):
            cov_shearshear_sheary[ntheta * j1:ntheta * (j1 + 1), ntheta * j2:ntheta * (j2 + 1)] = DV.fftcovtot_dict['gty_kk'][
                                                                            'bin_' + str(j1 + 1) + '_0_' + str(
                                                                                j2 + 1) + '_' + str(j2+1)][ind_th_sel, :][:,
                                                                        ind_th_sel]
    
    


cov_total = np.zeros((2*ntheta*nbins,2*ntheta*nbins))
cov_total[0:ntheta*nbins,0:ntheta*nbins] = cov_shearshear
cov_total[ntheta*nbins:2*ntheta*nbins,ntheta*nbins:2*ntheta*nbins] = cov_sheary
cov_total[0:ntheta*nbins,ntheta*nbins:2*ntheta*nbins] = cov_shearshear_sheary
cov_total[ntheta*nbins:2*ntheta*nbins,0:ntheta*nbins] = cov_shearshear_sheary.T


corrf_comb = np.hstack((Cl_shearshear,Cl_sheary))

theta_comb = np.vstack((theta_shearshear,theta_sheary))


results_dict = {}
results_dict['xcoord_all'] = theta_comb
results_dict['mean'] = corrf_comb
results_dict['cov_total'] = cov_total
# import pdb; pdb.set_trace()
filename_save = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/actxdes_shear/' + 'planck_des' + data_year + '_kk_gty_reformat_autobinonly_cov_' + cov_type + '_25_6_20.pk'
pk.dump(results_dict, open(filename_save, 'wb'), protocol=2)


