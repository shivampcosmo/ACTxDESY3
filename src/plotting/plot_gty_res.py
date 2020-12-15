import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import dill
import sys, os
from astropy.io import fits
import scipy.interpolate as interpolate
sys.path.insert(0, '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/cosmosis_code/')
os.environ['COSMOSIS_SRC_DIR'] = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis'
font = {'size': 18}
matplotlib.rc('font', **font)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import scipy.interpolate as interpolate


Color = ['k', '#000075', '#a9a9a9','#9A6324', '#808000','#aaffc3', '#fffac8'  ,'#800000', '#ffd8b1',]

import matplotlib
import matplotlib.pyplot as pl
font = {'size'   : 18}
matplotlib.rc('font', **font)
# # Latex stuff
pl.rc('text', usetex=True)
pl.rc('font', family='serif')


import pickle as pk


# cuts_min = [[20,40],[10,40],[7,40],[7,40]]
# cuts_max = [[30,250],[30,250],[30,250],[30,250]]

cuts_min = [[20,40],[10,40],[7,60],[7,60]]
cuts_max = [[30,250],[30,250],[30,250],[30,250]]


# cuts_min = [[20],[10],[7],[7]]
# cuts_max = [[250],[250],[250],[250]]

def get_chi2(binv,cuts_min, cuts_max, angv,data, theory,cov_bin):
    if len(cuts_min) > 1:
        selec1 = np.where((angv > cuts_min[0]) & (angv < cuts_max[0]))[0]
        selec2 = np.where((angv > cuts_min[1]) & (angv < cuts_max[1]))[0]
        selection = np.hstack((selec1, selec2))
    else:
        selection = np.where((angv > cuts_min[0]) & (angv < cuts_max[0]))[0]
    
    cov_selec = (cov_bin[:, selection])[selection, :]
    data_selec = data[selection]
    theory_selec = theory[selection]
    diff_selec = (data_selec - theory_selec)
    inv_cov_selec = np.linalg.inv(cov_selec)
    chi2 = np.dot(diff_selec, np.dot(inv_cov_selec, diff_selec))
    return np.round(chi2,2), len(selection)


fdir = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/results/'
ytype = 'planck'
# ytype = 'act'
do_residuals = 1
show_1h2h = 1
show_chi2 = 1

if ytype == 'act':
    fnames = [
        'DV_obj_MAP_values_gty_only_HM_delz_m_IA_P0A_P0z_betaA_betaz_almlow_almhigh_al1_PLcosmo_finalrun.pk'
    ]
    yt = 2
    from astropy.io import fits
    actf = df = fits.open('/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/actxdes_shear/DES_planck_ACT_theorycov.fits')   
    
    bin1_gty = actf['compton1_shear'].data['BIN1']
    bin2_gty = actf['compton1_shear'].data['BIN2']
    gty_csf = actf['compton1_shear'].data['VALUE']
    gty_ang = actf['compton1_shear'].data['ANG']
    gty_cov = actf['COVMAT'].data[480:,:][:,480:]
    gty_sig = np.sqrt(np.diag(actf['COVMAT'].data)[480:])
    label_yx = 'ACT x Y3'
    ylims = [-1.2,1.2]
    sc = [20,10,1.1,1.1]

if ytype == 'planck':
    fnames = [
        'DV_obj_MAP_values_gty_only_HM_delz_m_IA_P0A_P0z_betaA_betaz_almlow_almhigh_al1_PLcosmo_finalrun.pk'
    ]
    yt = 1
    from astropy.io import fits
    actf = df = fits.open('/global/cfs/cdirs/des/shivamp/ACTxDESY3_data/actxdes_shear/DES_planck_ACT_theorycov.fits')   

    bin1_gty = actf['compton_shear'].data['BIN1']
    bin2_gty = actf['compton_shear'].data['BIN2']
    gty_csf = actf['compton_shear'].data['VALUE']
    gty_ang = actf['compton_shear'].data['ANG']
    gty_sig = np.sqrt(np.diag(actf['COVMAT'].data)[400:480])
    gty_cov = actf['COVMAT'].data[400:480,:][:,400:480]
    label_yx = 'Planck x Y3'
    ylims = [-0.65,0.65]
    
    sc = [20,10,7.1,7.1]
    
    
labels = ['Best Fit','OWLS Ref','OWLS AGN']
# labels = ['Mead',r'$R_{\rm max}/R_{\rm 200c} = 3$',r'$R_{\rm max}/R_{\rm 200c} = 2$']
colors = ['r','b','g']
nbins = 4
bins = (np.arange(4) + 1).astype(int)
fig, ax = pl.subplots(1,4, figsize = (18,5),sharex=True,sharey='row')


for j2 in range(4):
    for jf in range(len(fnames)):

        DV = dill.load(open(fdir + fnames[jf],'rb'))  

        jc = 0
        texts_kk = [r'1,1',r'2,2',r'3,3',r'4,4']
        texts_ky = [r'1,y',r'2,y',r'3,y',r'4,y']
        theta_array = DV.xi_result_dict['gty']['theta']
        bin_str = 'yt_' + str(yt) + 'bin_' + str(j2+1) + '_' + str(0)

        Cl_j1_tot = DV.xi_result_dict['gty'][bin_str]['tot']
        sel_ind = np.where((bin1_gty == j2+1) & (bin2_gty == j2+1))[0]
        yk_g = gty_csf[sel_ind]
        yk_th_g = gty_ang[sel_ind]
        yk_err = gty_sig[sel_ind]
        cov_bin = gty_cov[sel_ind,:][:,sel_ind]
        inv_cov_bin = np.linalg.inv(cov_bin)

        if do_residuals:
            Cl_interp = interpolate.interp1d(np.log(theta_array), np.log(Cl_j1_tot),fill_value='extrapolate')

            Cl_d_th = np.exp(Cl_interp(np.log(yk_th_g)))
            if jf == 0:
                Cl_d_th_ref = Cl_d_th

            if jf == 0 and j2 == 0:
                ax[j2].errorbar(yk_th_g, (yk_g - Cl_d_th_ref)/Cl_d_th_ref ,yerr=yk_err/Cl_d_th_ref, ls='',marker='s',color='black',label=label_yx)
            else:
                ax[j2].errorbar(yk_th_g, (yk_g - Cl_d_th_ref)/Cl_d_th_ref,yerr=yk_err/Cl_d_th_ref, ls='',marker='s',color='black')
            
            if ((jf in [1,2]) and (j2 == 0)) or ((jf == 0) and (j2 == 1)):
                ax[j2].errorbar(yk_th_g, (Cl_d_th/Cl_d_th_ref) -1 , linestyle='-', marker='',lw = 2, color =colors[jf],label=labels[jf])  
            else:
                ax[j2].errorbar(yk_th_g, (Cl_d_th/Cl_d_th_ref) -1 , linestyle='-', marker='',lw = 2, color =colors[jf])  
        else:
            if jf == 0:
                ax[j2].errorbar(yk_th_g, yk_g,yerr=yk_err, ls='',marker='s',color='black',label=label_yx)
            else:
                ax[j2].errorbar(yk_th_g, yk_g,yerr=yk_err, ls='',marker='s',color='black')
    

            ax[j2].errorbar(theta_array, Cl_j1_tot, linestyle='-', marker='',lw = 2, color =colors[jf],label=labels[jf])  
            if show_1h2h:
                Cl_j1_1h = DV.xi_result_dict['gty'][bin_str]['1h']
                Cl_j1_2h = DV.xi_result_dict['gty'][bin_str]['2h']
                ax[j2].errorbar(theta_array, Cl_j1_1h, linestyle='--', marker='',lw = 2, color =colors[jf],label='1-halo')  
                ax[j2].errorbar(theta_array, Cl_j1_2h, linestyle=':', marker='',lw = 2, color =colors[jf],label='2-halo')  

        if show_chi2:                
            if j2 == 0:
                print('bin, (chi2, # of points)')
            
            print(j2+1, get_chi2(j2+1,cuts_min[j2], cuts_max[j2], yk_th_g,yk_g, Cl_d_th,cov_bin))


        ax[j2].axvspan(0,sc[j2],alpha=0.08,color='k')
        ax[j2].set_xscale('log')

        if do_residuals:
            ax[j2].set_ylim(ylims)        
        else:
            ax[j2].set_yscale('log')
            ax[j2].set_ylim(1e-11,2e-9)


        ax[j2].set_xlabel(r'$\theta$ (arcmin)', size = 20)
        ticks  = np.array([3,10,30,100])
        labels_bottom = ticks
        ax[j2].set_xticks(ticks)
        ax[j2].set_xticklabels(labels_bottom,  fontsize=15)

        ax[j2].tick_params(axis='both', which='minor', labelsize=15)    
        ax[j2].tick_params(axis='both', which='major', labelsize=15)
        ax[j2].set_xlim((2,250))
        ax[j2].text( 0.75, 0.96,str(j2+1) + ',y', verticalalignment='top', horizontalalignment='left', transform=ax[j2].transAxes, fontsize=15, bbox=dict(facecolor='white', edgecolor='black'))    


        jc += 1

    if do_residuals:
        ax[0].set_ylabel(r'$\Delta \xi_{y\gamma_t}/\xi^{\rm bestfit}_{y\gamma_t}$ ', size = 22)
    else:
        ax[0].set_ylabel(r'$\xi_{y\gamma_t}(\theta)$ ', size = 22)
    ax[0].legend(fontsize=15,loc='upper left')

pl.tight_layout()

fig.savefig('gty_bestfit_try.pdf')





