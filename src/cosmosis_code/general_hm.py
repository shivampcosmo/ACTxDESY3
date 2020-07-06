
import sys, os
import traceback
import pdb
import numpy as np
import scipy as sp
import scipy.special as spsp
from scipy import interpolate
import astropy.units as u
from astropy import constants as const
import colossus
from colossus.cosmology import cosmology
from colossus.lss import bias
from colossus.lss import mass_function
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import concentration
import copy
import itertools
sys.path.insert(0, os.environ['COSMOSIS_SRC_DIR'] + '/ACTxDESY3/helper/')
import mycosmo as cosmodef
from twobessel import *
import LSS_funcs as hmf
import plot_funcs as pf
import multiprocessing
import time
import pdb
import pickle as pk
import dill
from mcfit import Hankel

pi = np.pi


class general_hm:

    def __init__(self, cosmo_params, pressure_params, other_params):
        cosmology.addCosmology('mock_cosmo', cosmo_params)
        self.cosmo_colossus = cosmology.setCosmology('mock_cosmo')
#        self.cosmo_colossus = cosmology.setCosmology('planck18')
        h = cosmo_params['H0'] / 100.

        cosmo_func = cosmodef.mynew_cosmo(h, cosmo_params['Om0'], cosmo_params['Ob0'], cosmo_params['ns'],
                                          cosmo_params['sigma8'])
        self.cosmo = cosmo_func

        self.z_array = other_params['z_array']

        self.dndm_model = other_params['dndm_model']
        self.bias_model = other_params['bias_model']
        self.conc_model = other_params['conc_model']
        self.use_multiprocess = other_params['use_multiprocess']

        self.verbose = other_params['verbose']
        H0 = 100. * (u.km / (u.s * u.Mpc))
        G_new = const.G.to(u.Mpc ** 3 / ((u.s ** 2) * u.M_sun))
        self.rho_m_bar = ((cosmo_params['Om0'] * 3 * (H0 ** 2) / (8 * np.pi * G_new)).to(u.M_sun / (u.Mpc ** 3))).value
        # import pdb; pdb.set_trace()

    # Get the interpolated object of linear power spectrum at given cosmology
    def get_Pklin_zk_interp(self):
        k_array = np.logspace(-5, 2, 400)
        Pklinz_z0_test = hmf.get_Pklinz(0.0, k_array, current_cosmo=self.cosmo)
        sig8h = hmf.sigRz0(8., k_array, Pklinz_z0_test, window='tophat')
        sig8_ratio = ((self.cosmo.sig8 / sig8h) ** 2)
        zmin, zmax = np.min(self.z_array), np.max(self.z_array)
        z_array = np.logspace(np.log10(zmin), np.log10(zmax), 100)
        Pklinz_2d_mat = sig8_ratio * (hmf.get_Pklinzarray(z_array, k_array, current_cosmo=self.cosmo))
        # Pklinz_2d_mat_interp = interpolate.RectBivariateSpline(np.log(z_array), np.log(k_array),
        #                                                        np.log(Pklinz_2d_mat))
        Pklinz_2d_mat_interp = interpolate.RectBivariateSpline((z_array), np.log(k_array),
                                                               np.log(Pklinz_2d_mat))
        return Pklinz_2d_mat_interp


    # get the halo mass function and halo bias using the colossus module
    def get_dndm_bias(self, M_mat, mdef):

        dndm_array_Mz, bm_array_Mz = np.zeros(M_mat.shape), np.zeros(M_mat.shape)

        for j in range(len(self.z_array)):
            M_array = M_mat[j, :]

            dndm_array_Mz[j, :] = (1. / M_array) * mass_function.massFunction(M_array, self.z_array[j],
                                                                              mdef=mdef, model=self.dndm_model,
                                                                              q_out='dndlnM')

            bm_array_Mz[j, :] = bias.haloBias(M_array, self.z_array[j], model=self.bias_model, mdef=mdef)

        return dndm_array_Mz, bm_array_Mz

    # get the halo concentration using the colossus module
    def get_halo_conc_Mz(self, M_mat, mdef):
        halo_conc_array_Mz = np.zeros(M_mat.shape)

        if self.use_multiprocess:
            def get_halo_conc(ind, output_dict):
                M_array = M_mat[ind, :]
                output_dict[self.z_array[ind]] = concentration.concentration(M_array, mdef, self.z_array[ind],model=self.conc_model)

            manager = multiprocessing.Manager()
            halo_conc_dict = manager.dict()
            if self.verbose:
                print('getting halo_conc for each z and M')
            starttime = time.time()
            processes = []
            for j in range(len(self.z_array)):
                p = multiprocessing.Process(target=get_halo_conc, args=(j, halo_conc_dict))
                processes.append(p)
                p.start()

            for process in processes:
                process.join()
            if self.verbose:
                print('That took {} seconds'.format(time.time() - starttime))

            for j in range(len(self.z_array)):
                halo_conc_array_Mz[j, :] = halo_conc_dict[self.z_array[j]]
        else:
            for j in range(len(self.z_array)):
                M_array = M_mat[j, :]

                if mdef == 'fof' and self.conc_model=='bullock01':
                    halo_conc_array_Mz[j, :] = concentration.concentration(M_array, '200c', self.z_array[j],model=self.conc_model)
                else:
                    halo_conc_array_Mz[j, :] = concentration.concentration(M_array, mdef, self.z_array[j],model=self.conc_model)
        return halo_conc_array_Mz

    def get_wplin_interp(self, nu, pkzlin_interp):
        k_array = np.logspace(-5, 2.5, 30000)
        z_array = np.logspace(-3, 1, 100)
        # Pklinz0 = np.exp(pkzlin_interp.ev(np.log(z_array[0]), np.log(k_array)))
        Pklinz0 = np.exp(pkzlin_interp.ev((z_array[0]), np.log(k_array)))
        theta_out, xi_out = Hankel(k_array, nu=nu, q=1.0)(Pklinz0)
        xi_out *= (1 / (2 * np.pi))
        theta_out_arcmin = theta_out * (180. / np.pi) * 60.
        xi_mat = np.zeros((len(z_array), len(theta_out)))

        for j in range(len(z_array)):
            Pklinz = np.exp(pkzlin_interp.ev(np.log(z_array[j]), np.log(k_array)))
            theta_out, xi_out = Hankel(k_array, nu=nu, q=1.0)(Pklinz)
            xi_out *= (1 / (2 * np.pi))
            xi_mat[j, :] = xi_out

        wplin_mat_interp = interpolate.RectBivariateSpline(np.log(z_array), np.log(theta_out),
                                                           np.log(xi_mat))
        # import pdb; pdb.set_trace()
        return wplin_mat_interp

