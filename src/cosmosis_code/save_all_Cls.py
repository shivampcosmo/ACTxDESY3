import sys, os
from cosmosis.datablock import names, option_section
from numpy import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
import scipy.interpolate as interp
import scipy as sp
import ast
import pickle as pk
import copy
import pdb
import os
def save_Cls(block, sec_name,save_plot_dir):
    if not os.path.exists(save_plot_dir):
        os.mkdir(save_plot_dir)
    
    M = block[sec_name, 'M_array']
    print (M)
    nbins_lens = block['nz_lens', 'nbin']
    nbins_source = block['nz_source', 'nbin']
    ell = block[sec_name, 'theory_ell']
    
    try:
        for i in range(nbins_source):
            #for j in range(nbins_source):
                j=i
                out = open(save_plot_dir+'/Cl_kk_{0}_{1}'.format(i+1,j+1),'w')
                out.write('# ell     c_ll     err \n')
                for l in range(len(ell)):
                    Cl = block[sec_name, 'theory_Clkk_bin_' + str(i+1) + '_' + str(j+1)][l]
                    err = np.sqrt(np.diag(block[sec_name, 'cov_total_kk_kk_bin_'+ str(i+1) + '_' + str(j+1)]))[l]
                    out.write('{0}     {1}    {2} \n'.format((1.015**j)*ell[l],Cl,err))
                out.close()
    except:
        pass
    
    try:
        out = open(save_plot_dir+'/ell','w')
        out.write('# ell\n')
        for l in range(len(ell)):
            out.write('{0}\n'.format((1.015**j)*ell[l]))
        out.close()
    except:
        pass

    try:
        out = open(save_plot_dir+'/M','w')
        out.write('# Mass\n')
        for l in range(len(M)):
            out.write('{0}\n'.format((1.015**j)*M[l]))
        out.close()
    except:
        pass
    
    try:
        for i in range(nbins_source):
            #for j in range(nbins_source):
                j=i
                out = open(save_plot_dir+'/Cl_ky_{0}_{1}'.format(i+1,j+1),'w')
                out.write('# ell     c_ll     err \n')
                for l in range(len(ell)):
                    Cl = block[sec_name, 'theory_Clky_bin_' + str(i+1) + '_' + str(j+1)][l]
                    err = np.sqrt(np.diag(block[sec_name, 'cov_total_ky_ky_bin_' + str(i+1) + '_' + str(j+1)]))[l]
                    out.write('{0}     {1}    {2} \n'.format((1.015**j)*ell[l],Cl,err))
                out.close()
    except:
        pass
    try:
        for i in range(nbins_lens):
            #for j in range(nbins_source):
                j=i
                out = open(save_plot_dir+'/Cl_gg_{0}_{1}'.format(i+1,j+1),'w')
                out.write('# ell     c_ll     err \n')
                for l in range(len(ell)):
                    Cl = block[sec_name, 'theory_Clgg_bin_' + str(i+1) + '_' + str(j+1)][l]
                    err = np.sqrt(np.diag(block[sec_name, 'cov_total_gg_gg_bin_' + str(i+1) + '_' + str(j+1)]))[l]
                    out.write('{0}     {1}    {2} \n'.format((1.015**j)*ell[l],Cl,err))
                out.close()
    except:
        pass
    
    
    try:
        for i in range(nbins_lens):
            #for j in range(nbins_source):
                j=i
                out = open(save_plot_dir+'/Cl_gy_{0}_{1}'.format(i+1,j+1),'w')
                out.write('# ell     c_ll     err \n')
                for l in range(len(ell)):
                    Cl = block[sec_name, 'theory_Clgy_bin_' + str(i+1) + '_' + str(j+1)][l]
                    err = np.sqrt(np.diag(block[sec_name, 'cov_total_gy_gy_bin_' + str(i+1) + '_' + str(j+1)]))[l]
                    out.write('{0}     {1}    {2} \n'.format((1.015**j)*ell[l],Cl,err))
                out.close()
    except:
        pass
    
    
    
    try:
       
            #for j in range(nbins_source):
                j=i
                out = open(save_plot_dir+'/Cl_yy','w')
                out.write('# ell     c_ll     err \n')
                for l in range(len(ell)):
                    Cl = block[sec_name, 'theory_Clyy'][l]
                    err = np.sqrt(np.diag(block[sec_name, 'cov_total_yy_yy']))[l]
                    out.write('{0}     {1}    {2} \n'.format((1.015**j)*ell[l],Cl,err))
                out.close()
    except:
        pass
    
    
    
    
    
    
    
    # DMASS SAVING ****************
    
    try:
        for i in range(nbins_source):
            #for j in range(nbins_source):
                j=i
                out = open(save_plot_dir+'/Cl_kk_dM_{0}_{1}'.format(i+1,j+1),'w')
                
                for l in range(len(ell)):
                    st = ''
                    
                    for m in range(len(M)):
                
                        Cl = block[sec_name, 'theory_Clkk1h_dM_bin_' + str(i+1) + '_' + str(j+1)][m,l]
                        
                        st+='{0}  '.format(Cl)
                    out.write('{0}  \n'.format(st))
                out.close()
    except:
        pass
    
    
    
    
    for i in range(nbins_source):
            #for j in range(nbins_source):
                j=i
                out = open(save_plot_dir+'/Cl_ky_dM_{0}_{1}'.format(i+1,j+1),'w')
              
                for l in range(len(ell)):
                    st = ''
                    for m in range(len(M)):
                        Cl = block[sec_name, 'theory_Clky1h_dM_bin_' + str(i+1) + '_' + str(j+1)][m,l]
                        st+='{0}   '.format(Cl)
                    out.write('{0}  \n'.format(st))
                out.close()
    #except:
    #    pass
    try:
        for i in range(nbins_lens):
            #for j in range(nbins_source):
                j=i
                out = open(save_plot_dir+'/Cl_gg_dM_{0}_{1}'.format(i+1,j+1),'w')
                
                for l in range(len(ell)):
                    st = ''
                    for m in range(len(M)):
                        Cl = block[sec_name, 'theory_Clgg1h_dM_bin_' + str(i+1) + '_' + str(j+1)][m,l]
                        st+='{0}   '.format(Cl)
                    out.write('{0}  \n'.format(st))
                out.close()
    except:
        pass
    
    
    try:
        for i in range(nbins_lens):
            #for j in range(nbins_source):
                j=i
                out = open(save_plot_dir+'/Cl_gy_dM_{0}_{1}'.format(i+1,j+1),'w')
                
                for l in range(len(ell)):
                    st = ''
                    for m in range(len(M)):
                        Cl = block[sec_name, 'theory_Clgy1h_dM_bin_' + str(i+1) + '_' + str(j+1)][m,l]
                        st+='{0}   '.format(Cl)
                    out.write('{0}  \n'.format(st))
                out.close()
    except:
        pass
    
    
    
    try:
        
 
                out = open(save_plot_dir+'/Cl_yy_dM','w')
                
                for l in range(len(ell)):
                    st = ''
                    for m in range(len(M)):
                        Cl = block[sec_name, 'theory_Clyy1h_dM'][m,l]
                        st+='{0}   '.format(Cl)
                    out.write('{0}  \n'.format(st))
                out.close()
    except:
        pass
    
  
    

def setup(options):
   
    do_plot = options.get_bool(option_section, "do_plot", True)
    save_plot_dir = options.get_string(option_section, "save_plot_dir",
                                       '/home/shivam/Research/cosmosis/ACTxDESY3/src/plots/')
    sec_name = options.get_string(option_section, "sec_save_name",'get_cov')
    return do_plot, save_plot_dir, sec_name


def execute(block, config):
    print ('do_plot1')
    do_plot, save_plot_dir, sec_name = config
    save_Cls(block, sec_name,save_plot_dir)

    return 0

