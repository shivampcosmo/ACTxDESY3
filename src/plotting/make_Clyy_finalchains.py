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
import multiprocessing
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


        other_params_dict['ng_value'] = np.zeros_like(other_params_dict['z_array']) + 1e-100
        other_params_dict['ng_zarray'] = other_params_dict['z_array']
        other_params_dict['ng_zarray_source'] = other_params_dict['z_array']

        other_params_dict['ng_value_source'] = np.zeros_like(other_params_dict['z_array']) + 1e-100
        other_params_dict['cosmo_fid'] = cosmo_params_dict
        other_params_dict['hod_fid'] = hod_params_dict
        other_params_dict['binvl'] = 1
        other_params_dict['binvs'] = 1

        other_params_dict['nbar'] = 1e-100

        other_params_dict['noise_kappa'] = 0.0
        other_params_dict['kk_hm_trans'] = 1.0
        other_params_dict['total_Clyy_filename'] = '../data/Planck/planck_yy_total_full_mask60.txt'
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
        self.hod_params_dict = hod_params_dict

        

        
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
        
            
    def collect_Clyy_mat(self,pi_array, return_dict,all_params,data_params,beam_fwhm_arcmin=0.0):

        for pi in pi_array:
            print(pi)
            params_dict = {}
            for jp in range(len(all_params)):
                params_dict[all_params[jp]] = data_params[pi,jp]
            cosmo_params, pressure_params, other_params, hod_params = copy.deepcopy(
                self.cosmo_params_dict), copy.deepcopy(self.pressure_params_dict), copy.deepcopy(self.other_params_dict),copy.deepcopy(self.hod_params_dict)


            pressure_params = replace_values(params_dict, pressure_params)
            other_params['pressure_fid'] = pressure_params
            PrepDV_fid = PrepDataVec(cosmo_params, hod_params, pressure_params, other_params)
            sig_beam = beam_fwhm_arcmin * (1. / 60.) * (np.pi / 180.) * (1. / np.sqrt(8. * np.log(2)))
            Bl = (np.exp(-1. * PrepDV_fid.l_array * (PrepDV_fid.l_array + 1) * (sig_beam ** 2) / 2.)) ** (2) + 1e-200
            PrepDV_params = {}
            PrepDV_params['uyl_zM_dict0'] = PrepDV_fid.uyl_zM_dict
            PrepDV_params['byl_z_dict0'] = PrepDV_fid.byl_z_dict
            PrepDV_params['PrepDV_fid'] = PrepDV_fid
            CalcDV = CalcDataVec(PrepDV_params)
            Cl1h_j1j2 = CalcDV.get_Cl_AB_1h('y', 'y', PrepDV_fid.l_array, PrepDV_params['uyl_zM_dict0'],
                                                PrepDV_params['uyl_zM_dict0'])
            Cl2h_j1j2 = CalcDV.get_Cl_AB_2h('y', 'y', PrepDV_fid.l_array, PrepDV_params['byl_z_dict0'],
                                                PrepDV_params['byl_z_dict0'])
            return_dict[pi] = Bl * CalcDV.get_Cl_AB_tot('y', 'y', Cl1h_j1j2, Cl2h_j1j2)                    



    def get_Clyy_mat(self,chain_file=None, nsamps=100,beam_fwhm_arcmin=0.0,do_multiprocess=True,num_pool=28):   
        PrepDV_fid = PrepDataVec(self.cosmo_params_dict, self.hod_params_dict, self.pressure_params_dict, self.other_params_dict)
        sig_beam = beam_fwhm_arcmin * (1. / 60.) * (np.pi / 180.) * (1. / np.sqrt(8. * np.log(2)))
        Bl = (np.exp(-1. * PrepDV_fid.l_array * (PrepDV_fid.l_array + 1) * (sig_beam ** 2) / 2.)) ** (2) + 1e-200
        PrepDV_params = {}
        PrepDV_params['uyl_zM_dict0'] = PrepDV_fid.uyl_zM_dict
        PrepDV_params['byl_z_dict0'] = PrepDV_fid.byl_z_dict
        PrepDV_params['PrepDV_fid'] = PrepDV_fid
        CalcDV = CalcDataVec(PrepDV_params)
        Cl1h_j1j2 = CalcDV.get_Cl_AB_1h('y', 'y', PrepDV_fid.l_array, PrepDV_params['uyl_zM_dict0'],
                                                PrepDV_params['uyl_zM_dict0'])
        Cl2h_j1j2 = CalcDV.get_Cl_AB_2h('y', 'y', PrepDV_fid.l_array, PrepDV_params['byl_z_dict0'],
                                                PrepDV_params['byl_z_dict0'])
        Clyy_fid = Bl * CalcDV.get_Cl_AB_tot('y', 'y', Cl1h_j1j2, Cl2h_j1j2)

        if chain_file is not None:
            infile = open(chain_file, 'r')
            first_line = infile.readline()

            data = np.loadtxt(chain_file)
            nsample = nsamps

            data_params = data[-nsample:,:-4]
            weights = data[-nsample:,-1]

            all_params = first_line.split()[:-4]
            all_params[0] = all_params[0].split('#')[1]
            
            Clyy_mat = np.zeros((nsample,len(PrepDV_fid.l_array)))
            if do_multiprocess:
                manager = multiprocessing.Manager()
                Clyy_dict = manager.dict()

                processes = []
                if num_pool is None:
                    for pi in range(nsample):
                        p = multiprocessing.Process(target=self.collect_Clyy_mat, args=(
                            [pi],Clyy_dict, all_params,data_params))
                        processes.append(p)
                        p.start()
                else:
                    npool = num_pool
                    pi_array = np.arange(nsample)
                    pi_array_split = np.array_split(pi_array, npool)
                    print(pi_array_split)
                    for j in range(npool):
                        p = multiprocessing.Process(target=self.collect_Clyy_mat, args=(
                            pi_array_split[j], Clyy_dict, all_params,data_params))
                        processes.append(p)
                        p.start()

                for process in processes:
                    process.join()

                for pi in range(nsample):
                    Clyy_mat[pi, :] = Clyy_dict[pi]

                Clyy_low = weighted_percentile(Clyy_mat, percentiles[0], weights=weights)
                Clyy_high = weighted_percentile(Clyy_mat, percentiles[1], weights=weights)
            
                return PrepDV_fid.l_array, Clyy_low, Clyy_high, Clyy_fid
            else:
                return PrepDV_fid.l_array, Clyy_fid



pf_dir = os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/src/params_files/'
pf_main = 'final_runs/params_des_kk_ky_planckacty3_beamed_B12_highbroken.ini'
chain_f = '/global/cfs/cdirs/des/shivamp/nl_cosmosis/cosmosis/ACTxDESY3/src/chains/chain_xipm_gtyPLonly_fidcuts_HM_delz_m_IA_P0A_P0z_P0m_alphigh_highbpl_al1_PLcosmo_finalrun3.txt'
save_data_fname = 'Clyy_plot_data_xipm_gtyPLonly_fidcuts_HM_delz_m_IA_P0A_P0z_P0m_alphigh_highbpl_al1_PLcosmo_finalrun3.txt'
save_plot_fname = 'Clyy_xipm_gtyPLonly_fidcuts_HM_delz_m_IA_P0A_P0z_P0m_alphigh_highbpl_al1_PLcosmo_finalrun3.pdf'

pf_def = 'params_default.ini'
mp = makeplot(pf_dir, pf_main, pf_def)



nsample = get_nsample(chain_f)
nsamps = nsample
# nsamps = 500
l_array, Clyy_low, Clyy_high, Clyy_fid = mp.get_Clyy_mat(chain_file=chain_f, nsamps=nsamps)

savedict = {'l':l_array, 'Clyy_low':Clyy_low,'Clyy_high':Clyy_high,'Clyy_fid':Clyy_fid}

import pickle as pk
pk.dump(savedict, open(save_data_fname,'wb'))

pl.rc('text', usetex=False)

percentiles = [16.0, 84.0]

colors = ['blue']
alpha_list = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2]
labels = [r'Data']
linestyles = ['-']


fig, ax = pl.subplots(1, 1, figsize=(8, 6))


scaling = (1./(2*np.pi))*l_array*(l_array +1)
ax.fill_between(l_array, Clyy_low*(scaling), Clyy_high*(scaling), color='blue', alpha=alpha_list[0], label = 'Planck + ACT')
# ax.fill_between(M_array, YM_low_pl*scaling, YM_high_pl*scaling, color='green', alpha=alpha_list[0], label = 'Planck')
ax.plot(l_array, Clyy_fid*(scaling), color='k', alpha=1.0, label = 'Battaglia 12')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$', size=20)
# ax.set_ylabel(r'$ \tilde{Y}_{500} / M^{5/3}_{500} \ [\rm{arcmin}^2 (h/ M_{\odot})^{5/3} ]$', size=20)
ax.set_ylabel(r'$\ell (\ell+1) C_{\ell}/2\pi$', size=20)
ax.set_xlim(1,7e3) 
ax.set_ylim(1e-15,1e-11)

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


