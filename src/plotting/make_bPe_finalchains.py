# %reset

# %reset
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
sys.path.insert(0, os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/helper/')
import mycosmo as cosmodef
import LSS_funcs as hmf

Color = ['k', '#000075', '#a9a9a9','#9A6324', '#808000','#aaffc3', '#fffac8'  ,'#800000', '#ffd8b1',]

import matplotlib
import matplotlib.pyplot as pl
font = {'size'   : 18}
matplotlib.rc('font', **font)
# # Latex stuff
pl.rc('text', usetex=True)
pl.rc('font', family='serif')
import colossus
from colossus.cosmology import cosmology

import pickle as pk
import numpy as np
import pickle as pk
import matplotlib
import matplotlib.pyplot as pl
from astropy import units
from astropy import constants
# import matplotlib
from astropy.io import fits
import sys, os
import dill
# %pylab inline

os.environ['COSMOSIS_SRC_DIR'] = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis'

# sys.path.insert(0, '../../../helper/')
sys.path.insert(0, '../cosmosis_code/')

from pressure import *
from get_theory_interface import *


def replace_values(params_dict, pressure_params_orig, sec_name='theory_yx'):
    pressure_params_dict = copy.deepcopy(pressure_params_orig)
    for key in params_dict.keys():
        key_sp = key.split('--')
        if key_sp[0] == sec_name:
            param_val = key_sp[1]
            for pressure_keys in pressure_params_dict.keys():
                if param_val == pressure_keys.lower():
                    pressure_params_dict[pressure_keys] = params_dict[key]

    return pressure_params_dict

def weighted_percentile(data_mat, percents, weights=None):
    weighted_mat = np.zeros(data_mat.shape[1])
    for mj in range(data_mat.shape[1]):
        data = data_mat[:, mj]
        if weights is None:
            return np.percentile(data, percents)
        ind=np.argsort(data)
        d=data[ind]
        w=weights[ind]
        p=1.*w.cumsum()/w.sum()*100
        y=np.interp(percents, p, d)
        weighted_mat[mj] = y
    return weighted_mat

def get_nsample(filename):
    with open(filename,"r") as fi:
        for ln in fi:
            if ln.startswith("#nsample="):
                nsamples = int(ln[9:])
    return nsamples

z = 0.25
# h = 0.6774
h = 0.7
# percentiles = [2.5, 97.5]
percentiles = [16.0, 84.0]

class makeplot:
    def __init__(self, params_files_dir,params_file , params_def_file):
        
        ini_info = read_ini(params_files_dir + params_file, ini_def=params_files_dir + params_def_file)
        other_params_dict = copy.deepcopy(ini_info['other_params_dict'])
        cosmo_params_dict = copy.deepcopy(ini_info['cosmo_params_dict'])
        pressure_params_dict = copy.deepcopy(ini_info['pressure_params_dict'])
        hod_params_dict = copy.deepcopy(ini_info['hod_params_dict'])
        cosmology.addCosmology('mock_cosmo', cosmo_params_dict)
        self.cosmo_colossus = cosmology.setCosmology('mock_cosmo')
        #        self.cosmo_colossus = cosmology.setCosmology('planck18')
        h = cosmo_params_dict['H0'] / 100.
        cosmo_func = cosmodef.mynew_cosmo(h, cosmo_params_dict['Om0'], cosmo_params_dict['Ob0'], cosmo_params_dict['ns'], cosmo_params_dict['sigma8'])  
        self.cosmo = cosmo_func
        self.M_array, self.z_array, self.x_array = other_params_dict['M_array'], other_params_dict['z_array'], other_params_dict[
        'x_array']
        self.verbose = other_params_dict['verbose']
        self.nm, self.nz = len(self.M_array), len(self.z_array)
        self.rho_crit_array = self.cosmo_colossus.rho_c(self.z_array) * (1000 ** 3)
        M_mat_mdef = np.tile(self.M_array.reshape(1, self.nm), (self.nz, 1))
        self.M_mat = M_mat_mdef
        self.R_mat = hmf.get_R_from_M_mat(self.M_mat,other_params_dict['pressure_model_delta'] * self.rho_crit_array)  
        self.Pressure_fid = Pressure(cosmo_params_dict, pressure_params_dict, other_params_dict)
        self.dndm_mat, self.bm_mat = other_params_dict['dndm_array'], other_params_dict['bm_array']
        other_params_dict['pressure_fid'] = pressure_params_dict
        self.cosmo_params_dict, self.pressure_params_dict, self.other_params_dict = cosmo_params_dict, pressure_params_dict, other_params_dict
    

        
    def collect_YM(self,chain_file=None, nsamps=100, M_min = 1e12, M_max = 10**15.3, z=0.25, numM = 20):
                
        M_array_500c = np.logspace(np.log10(M_min), np.log10(M_max), num=numM)
        
        pressure = Pressure(self.cosmo_params_dict, self.pressure_params_dict, self.other_params_dict)        
        YM_fid = np.zeros(numM)
        
        for mi in range(numM):
            YM_fid[mi] = pressure.get_Y500sph_singleMz(M_array_500c[mi], z, do_fast=False)
        
        
        if chain_file is not None:
            infile = open(chain_file, 'r')
            first_line = infile.readline()

            data = np.loadtxt(chain_file)
            nsample = nsamps

            data_params = data[-nsample:,:-4]
            weights = data[-nsample:,-1]

            all_params = first_line.split()[:-4]
            all_params[0] = all_params[0].split('#')[1]

            integratedY_mat = np.zeros((nsample,numM))
    #         numM = len(M_array_500c)
            for pi in range(nsample):
                params_dict = {}
                for jp in range(len(all_params)):
                    params_dict[all_params[jp]] = data_params[pi,jp]
                cosmo_params, pressure_params, other_params = copy.deepcopy(
                    self.cosmo_params_dict), copy.deepcopy(self.pressure_params_dict), copy.deepcopy(self.other_params_dict)

                pressure_params = replace_values(params_dict, pressure_params)
                other_params['pressure_fid'] = pressure_params
    #             print(pressure_params)
                pressure = Pressure(cosmo_params, pressure_params, other_params)

                for mi in range(numM):
                    integratedY = pressure.get_Y500sph_singleMz(M_array_500c[mi], z, do_fast=False)
                    integratedY_mat[pi,mi] = integratedY               

            YM_low = weighted_percentile(integratedY_mat, percentiles[0], weights=weights)
            YM_high = weighted_percentile(integratedY_mat, percentiles[1], weights=weights)
            return M_array_500c, YM_low, YM_high, YM_fid, integratedY_mat
        
        else:
            return M_array_500c, YM_fid
        
            
    def collect_bPe_mat(self,chain_file=None, nsamps=100, x_min = 0.1, x_max = 3, numx=30):
            
            x_array = np.logspace(np.log10(x_min), np.log10(x_max), num=numx)
            pressure = Pressure(self.cosmo_params_dict, self.pressure_params_dict, self.other_params_dict)
            x_mat = np.tile(x_array.reshape(1,1,numx),(self.nz,self.nm,1))
            
            Pe_fid = pressure.get_Pe_mat(self.M_mat, x_array, self.other_params_dict['z_array'], self.R_mat)
            ET_fid = (4*np.pi*self.R_mat**3)*sp.integrate.simps((x_mat**2)*Pe_fid,x_array)
            bPe_fid = ((1+self.z_array)**3) * (sp.integrate.simps(ET_fid*self.dndm_mat*self.bm_mat,self.M_array))
            if chain_file is not None:
                infile = open(chain_file, 'r')
                first_line = infile.readline()

                data = np.loadtxt(chain_file)
                nsample = nsamps

                data_params = data[-nsample:,:-4]
                weights = data[-nsample:,-1]

                all_params = first_line.split()[:-4]
                all_params[0] = all_params[0].split('#')[1]
                
                bPe_mat = np.zeros((nsample,self.nz))
        
                for pi in range(nsample):
                    params_dict = {}
                    for jp in range(len(all_params)):
                        params_dict[all_params[jp]] = data_params[pi,jp]
                    cosmo_params, pressure_params, other_params = copy.deepcopy(
                        self.cosmo_params_dict), copy.deepcopy(self.pressure_params_dict), copy.deepcopy(self.other_params_dict)

                    pressure_params = replace_values(params_dict, pressure_params)
                    other_params['pressure_fid'] = pressure_params
                    pressure = Pressure(cosmo_params, pressure_params, other_params)
                    Pe_fid = pressure.get_Pe_mat(self.M_mat, x_array, self.other_params_dict['z_array'], self.R_mat)
                    ET_fid = (4*np.pi*self.R_mat**3)*sp.integrate.simps((x_mat**2)*Pe_fid,x_array)
                    bPe_mat[pi,:] = ((1+self.z_array)**3) * (sp.integrate.simps(ET_fid*self.dndm_mat*self.bm_mat,self.M_array))
                    

                bPe_low = weighted_percentile(bPe_mat, percentiles[0], weights=weights)
                bPe_high = weighted_percentile(bPe_mat, percentiles[1], weights=weights)
            
                return self.z_array, bPe_low, bPe_high, bPe_fid
            else:
                return self.z_array, bPe_fid



pf_dir = os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/src/params_files/'
pf_main = 'final_runs/params_des_ky_planckacty3_beamed_B12_highbroken.ini'

ldir = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/chains/'
chain_f = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/chains/chain_gtyPLonly_only_fidcuts_HM_delz_m_IA_P0A_P0z_betaA_alphigh_highbpl_al1_PLcosmo_finalrun3.txt'
save_data_fname = 'bPe_plot_data_gtyPLonly_only_fidcuts_HM_delz_m_IA_P0A_P0z_betaA_alphigh_highbpl_al1_PLcosmo_finalrun3.pk'
save_plot_fname = 'bPe_gtyPLonly_only_fidcuts_HM_delz_m_IA_P0A_P0z_betaA_alphigh_highbpl_al1_PLcosmo_finalrun3.pdf'

pf_def = 'params_default.ini'
mp = makeplot(pf_dir, pf_main, pf_def)



nsample = get_nsample(chain_f)
nsamps = nsample

z_array, bPe_low, bPe_high, bPe_fid = mp.collect_bPe_mat(chain_file=chain_f, nsamps=nsamps)

savedict = {'z':z_array, 'bPe_low':bPe_low,'bPe_high':bPe_high,'bPe_fid':bPe_fid}

import pickle as pk
pk.dump(savedict, open(save_data_fname,'wb'))

pl.rc('text', usetex=False)

percentiles = [16.0, 84.0]

colors = ['blue']
alpha_list = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2]
labels = [r'Data']
linestyles = ['-']


fig, ax = pl.subplots(1, 1, figsize=(8, 6))


    
ax.fill_between(z_array, bPe_low*(1e-3), bPe_high*(1e-3), color='blue', alpha=alpha_list[0], label = 'Planck + ACT')
# ax.fill_between(M_array, YM_low_pl*scaling, YM_high_pl*scaling, color='green', alpha=alpha_list[0], label = 'Planck')
ax.plot(z_array, bPe_fid*(1e-3), color='k', alpha=1.0, label = 'Battaglia 12')
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel(r'$z$', size=20)
# ax.set_ylabel(r'$ \tilde{Y}_{500} / M^{5/3}_{500} \ [\rm{arcmin}^2 (h/ M_{\odot})^{5/3} ]$', size=20)
ax.set_ylabel(r'$\langle bP_e \rangle$', size=20)
ax.set_xlim(0.0,1.0)
ax.set_ylim(0.0,5e-7)

ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)

# Colors = ['#0072b1','#009d73','#d45e00','k', 'grey','yellow']

# if (0):
#     ax.plot(ym_agn8_data[:,0]*h, ym_agn8_data[:,1], label = r'${\rm OWLS \, AGN}$', lw = 2, color  = Colors[0])
# #     ax.plot(ym_agn85_data[:,0]*h, ym_agn85_data[:,1], label =  r'${\rm AGN8.5}$', lw = 2, ls = 'dashed', color = Colors[1])
#     ax.plot(ym_ref_data[:,0]*h, ym_ref_data[:,1], label =  r'${\rm OWLS \, REF}$', lw = 2, ls = 'dotted', color = Colors[2])
# if (1):
#     ax.plot(ym_agn8_data[:,0]*h, ym_agn8_data[:,1]/(ym_agn8_data[:,0]*h/1.0e15)**(5./3.), label = r'${\rm OWLS \, AGN}$', lw = 3, color  = Colors[2])
# #     ax.plot(ym_agn85_data[:,0]*h, ym_agn85_data[:,1]/(ym_agn85_data[:,0]*h/1.0e15)**(5/3.), label =  r'${\rm AGN8.5}$', lw = 3, ls = 'dashed', color = Colors[1])
#     ax.plot(ym_ref_data[:,0]*h, ym_ref_data[:,1]/(ym_ref_data[:,0]*h/1.0e15)**(5./3.), label =  r'${\rm OWLS \, REF}$', lw = 3, ls = 'dotted', color = Colors[4])

# ax.set_xlim((1.0e12, 1.0e15))
# #ax.set_ylim((2.0e-8, 1.0e-2))
# #ax.set_ylim((1.0e-29, 5.0e-27))


legend = ax.legend(fontsize=18, frameon=False, loc = 'lower right')
# fig.savefig('YM_buzzard_try.pdf')
fig.savefig(save_plot_fname)


