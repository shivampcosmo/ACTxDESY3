import sys
sys.path.insert(0, './')
import numpy as np
import pdb
from scipy import interpolate
from matplotlib import pyplot as pl
import scipy as sp
import colossus
from colossus.cosmology import cosmology
from colossus.lss import mass_function

sys.path.insert(0, '../helper/')
import mycosmo as cosmodef
import LSS_funcs as hmf
import astropy.units as u
from astropy import constants as const


# See Fig. 2.6 of http://inspirehep.net/record/1654540/files/fdr-science-biblatex.pdf


# returns n(z) per deg^2
def data_to_nz(data, zz):
    zz_table = data[:, 0]
    dndz_table = data[:, 1]
    dndz_func = interpolate.interp1d(zz_table, dndz_table, bounds_error=False, fill_value=0.)
    minz = np.min(zz_table)
    maxz = np.max(zz_table)
    dz = zz[1:] - zz[:-1]
    zcent = 0.5 * (zz[1:] + zz[:-1])
    # nz = dz * dndz_func(zcent)
    nz = dndz_func(zcent)
    return nz


# get nbar in 1/str from n(z) per deg^2
def get_nbar(nz, zz):
    nz_str = nz * (180. / np.pi) ** 2
    val = sp.integrate.simps(nz_str, zz)
    return val


def get_nz_normalized(nz, zz):
    val = sp.integrate.simps(nz, zz)
    return nz / val


def get_desi_specs(sample_name, zz, bgs_filename = 'forecast_data/desi_BGS.dat',elg_filename = 'forecast_data/desi_ELG.dat'):
    fsky_desi_so = 0.23
    # BGS survey
    # bgs_filename = 'forecast_data/desi_BGS.dat'
    bgs_data = np.genfromtxt(bgs_filename, delimiter=',')
    bgs_bDz = 1.34
    bgs_nz = data_to_nz(bgs_data, zz)

    # ELG survey
    # elg_filename = 'forecast_data/desi_ELG.dat'
    elg_data = np.genfromtxt(elg_filename)
    elg_bDz = 0.84
    elg_nz = data_to_nz(bgs_data, zz)

    if sample_name == 'bgs':
        nz_output = np.copy(bgs_nz)
    if sample_name == 'elg':
        nz_output = np.copy(elg_nz)

    return nz_output


def get_nz_tophat(zarray, zmin, zmax):
    ind_z = np.where((zarray > zmin) & (zarray < zmax))[0]
    valf = np.zeros(len(zarray))
    valf[ind_z] = 1. / (zmax - zmin)
    return valf


def get_nz_g_2mrs(z, m, n, z0):
    val = (n / (z0 * sp.special.gamma((m + 1.) / n))) * ((z / z0) ** m) * np.exp(-1. * (z / z0) ** n)
    return val


def get_nz_halo(z_array, cosmo_params, other_params):
    zmin = other_params['zmin_tracer']
    zmax = other_params['zmax_tracer']
    fsky = other_params['fsky_gg']
    completeness_frac = other_params['completeness_frac']

    cosmology.addCosmology('mock_cosmo', cosmo_params)
    cosmology.setCosmology('mock_cosmo')
    M_array = np.logspace(other_params['log_M_min_tracer'], other_params['log_M_max_tracer'],other_params['num_M'])
    dndm_array_Mz = np.zeros((len(z_array), other_params['num_M']))

    for j in range(len(z_array)):

        if (z_array[j] < zmax) and (z_array[j] > zmin):
            dndm_array_Mz[j, :] = (1. / M_array) * mass_function.massFunction(M_array, z_array[j],
                                                                              mdef=other_params['mdef_analysis'],
                                                                              model=other_params['dndm_model'],
                                                                              q_out='dndlnM')
        else:
            dndm_array_Mz[j, :] = np.zeros(len(M_array))

    chi_array = hmf.get_Dcom_array(z_array, cosmo_params['Om0'])
    dchi_dz_array = (const.c.to(u.km / u.s)).value / (hmf.get_Hz(z_array, cosmo_params['Om0']))
    # Number of halos per steradian
    dNh_dz = 4*np.pi * (fsky * completeness_frac) * (chi_array ** 2) * dchi_dz_array * sp.integrate.simps(dndm_array_Mz, M_array)
    N_h = sp.integrate.simps(dNh_dz, z_array)
    dNh_dz_normed = dNh_dz / N_h
    # pdb.set_trace()
    return dNh_dz_normed, N_h


if __name__ == '__main__':
    print "getting desi numbers"
    minz = 0.0
    maxz = 2.0
    numz = 100
    zz = np.linspace(minz, maxz, num=numz)
    zcent = 0.5 * (zz[0:-1] + zz[1:])
    nz = get_desi_specs('bgs', zz)

    fig, ax = pl.subplots(1, 1)
    ax.plot(zcent, nz)
    fig.show()
    pdb.set_trace()
