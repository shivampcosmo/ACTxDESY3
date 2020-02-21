import sys, platform, os
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

try:
    df = pk.load(open('/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/actxdes_shear/results.pkl','rb'))
except:
    df = pk.load(open('/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/actxdes_shear/results.pkl', 'rb'),
                 encoding='latin1')

ntheta = len(df['Yshear_y3'][str(0)]['theta'])
bins_array = [1,2,3,4]
nbins = len([*df['Yshear_y3'].keys()])

Cl_shearshear = np.zeros(ntheta*nbins)
Cl_sheary = np.zeros(ntheta*nbins)

cov_shearshear = np.zeros((ntheta*nbins,ntheta*nbins))
cov_sheary = np.zeros((ntheta*nbins,ntheta*nbins))



theta_sheary = np.array([])
theta_shearshear = np.array([])

for j2 in range(len(bins_array)):
    # j2 = bins_array[j]

    kk_cov = df['shearshear_y3'][str(j2)+ '_' + str(j2)]['cov']
    kk_g = df['shearshear_y3'][str(j2) + '_' + str(j2)]['xip']
    kk_th_g = df['shearshear_y3'][str(j2) + '_' + str(j2)]['theta'] * 180. / np.pi
    kk_err = df['shearshear_y3'][str(j2) + '_' + str(j2)]['err']
    Cl_shearshear[ntheta * (j2):ntheta * (j2 + 1)] = kk_g
    cov_shearshear[ntheta*(j2):ntheta*(j2+1),ntheta*(j2):ntheta*(j2+1)] = kk_cov

    if len(theta_shearshear) == 0:
        theta_shearshear = df['shearshear_y3'][str(j2)+ '_' + str(j2)]['theta'] * 180. / np.pi
    else:
        theta_shearshear = np.vstack((theta_shearshear,df['shearshear_y3'][str(j2)+ '_' + str(j2)]['theta'] * 180. / np.pi))

    yk_cov = df['Yshear_y3'][str(j2)]['cov']
    yk_g = df['Yshear_y3'][str(j2)]['xip']
    yk_th_g = df['Yshear_y3'][str(j2)]['theta'] * 180. / np.pi
    yk_err = df['Yshear_y3'][str(j2)]['err']
    Cl_sheary[ntheta*(j2):ntheta*(j2+1)] = yk_g
    cov_sheary[ntheta*(j2):ntheta*(j2+1),ntheta*(j2):ntheta*(j2+1)] = yk_cov

    if len(theta_sheary) == 0:
        theta_sheary = df['Yshear_y3'][str(j2)]['theta'] * 180. / np.pi
    else:
        theta_sheary = np.vstack((theta_sheary,df['Yshear_y3'][str(j2)]['theta'] * 180. / np.pi))



cov_total = np.zeros((2*ntheta*nbins,2*ntheta*nbins))
cov_total[0:ntheta*nbins,0:ntheta*nbins] = cov_shearshear
cov_total[ntheta*nbins:2*ntheta*nbins,ntheta*nbins:2*ntheta*nbins] = cov_sheary

corrf_comb = np.hstack((Cl_shearshear,Cl_sheary))

theta_comb = np.vstack((theta_shearshear,theta_sheary))


results_dict = {}
results_dict['xcoord_all'] = theta_comb
results_dict['mean'] = corrf_comb
results_dict['cov_total'] = cov_total

filename_save = '/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/actxdes_shear/' + 'planck_desy3_kk_gty_reformat_autobinonly.pk'
pk.dump(results_dict, open(filename_save, 'wb'), protocol=2)


