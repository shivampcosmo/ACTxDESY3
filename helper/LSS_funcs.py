import astropy.units as u
import camb
import copy
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
from astropy.cosmology import Planck15
from camb import model
from scipy.integrate import simps as _simps
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
import pdb
# from numba import vectorize, float64
# from numba import jit

pi = np.pi


def get_Dcom(zf, Omega_m):
    Omega_L = 1. - Omega_m
    c = 3 * 10 ** 5
    res1 = sp.integrate.quad(lambda z: (c / 100) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
    Dcom = res1[0]
    return Dcom


def get_Dcom_array(zarray, Omega_m):
    Omega_L = 1. - Omega_m
    c = 3 * 10 ** 5
    Dcom_array = np.zeros(len(zarray))
    for j in range(len(zarray)):
        zf = zarray[j]
        res1 = sp.integrate.quad(lambda z: (c / 100) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
        Dcom = res1[0]
        Dcom_array[j] = Dcom
    return Dcom_array


def get_Dang_com(zf, Omega_m):
    Omega_L = 1. - Omega_m
    c = 3 * 10 ** 5
    res1 = sp.integrate.quad(lambda z: (c / 100) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
    Dcom = res1[0]
    return Dcom / (1. + zf)


def get_Dang_com_array(zarray, Omega_m):
    Omega_L = 1. - Omega_m
    c = 3 * 10 ** 5
    Dang_com_array = np.zeros(len(zarray))
    for j in range(len(zarray)):
        zf = zarray[j]
        res1 = sp.integrate.quad(lambda z: (c / 100) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
        Dcom = res1[0]
        Dang_com_array[j] = Dcom / (1. + zf)
    return Dang_com_array


def get_Hz(zarray, Omega_m):
    Omega_L = 1 - Omega_m
    Ez = np.sqrt(Omega_m * (1 + zarray) ** 3 + Omega_L)
    Hz = 100. * Ez
    return Hz


def get_Ez(zarray, Omega_m):
    Omega_L = 1 - Omega_m
    Ez = np.sqrt(Omega_m * (1 + zarray) ** 3 + Omega_L)
    return Ez


# cosmo_params = [h,Om0,Ob0,ns,sig8]
def get_Pklinz(z_mean, karr, current_cosmo=Planck15, Pklinz0=None):
    H0 = current_cosmo.H0.value
    h = H0 / 100.0
    Omega_m = current_cosmo.Om0
    Ommh2 = Omega_m * ((H0 / 100) ** 2)
    Omega_b = current_cosmo.Ob0
    Ombh2 = Omega_b * ((H0 / 100) ** 2)
    Omega_c = Omega_m - Omega_b
    Omch2 = Omega_c * ((H0 / 100) ** 2)
    Omega_L = current_cosmo.Ode0
    Omlh2 = Omega_L * ((H0 / 100) ** 2)
    nsval = current_cosmo.ns

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=Ombh2, omch2=Omch2, mnu=0.0)
    pars.InitPower.set_params(ns=nsval)
    pars.set_matter_power(redshifts=[z_mean], kmax=np.amax(karr))

    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    karr, z, pklin = results.get_matter_power_spectrum(minkh=np.amin(karr), maxkh=np.amax(karr), npoints=len(karr))

    Pklin = pklin[0, :]

    return Pklin


# cosmo_params = [h,Om0,Ob0,ns,sig8]
def get_Pklinzarray(z_mean_array, karr, current_cosmo=Planck15, Pklinz0=None):
    H0 = current_cosmo.H0.value
    h = H0 / 100.0
    Omega_m = current_cosmo.Om0
    Ommh2 = Omega_m * ((H0 / 100) ** 2)
    Omega_b = current_cosmo.Ob0
    Ombh2 = Omega_b * ((H0 / 100) ** 2)
    Omega_c = Omega_m - Omega_b
    Omch2 = Omega_c * ((H0 / 100) ** 2)
    Omega_L = current_cosmo.Ode0
    Omlh2 = Omega_L * ((H0 / 100) ** 2)
    nsval = current_cosmo.ns


    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=Ombh2, omch2=Omch2, mnu=0.0)
    pars.InitPower.set_params(ns=nsval)
    pars.set_matter_power(redshifts=z_mean_array, kmax=np.amax(karr))

    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    karr, z, pklin = results.get_matter_power_spectrum(minkh=np.amin(karr), maxkh=np.amax(karr), npoints=len(karr))

    return pklin


def get_rhom(current_cosmo=Planck15):
    H0 = current_cosmo.H0.value
    h = H0 / 100.0
    Omega_m = current_cosmo.Om0
    rhocr = (current_cosmo.critical_density0 / (h ** 2))
    rhom = (Omega_m * rhocr).to(u.solMass / u.Mpc ** 3).value

    return rhom


def sigmz0(m, karr, Pklinz0, rhom=get_rhom(), window='tophat'):
    nk = len(karr)
    R = np.power((3 * m / (4 * np.pi * rhom)), 1. / 3.)
    x = karr * R
    if window == 'tophat':
        wkr = (3 / (x * x * x)) * (np.sin(x) - x * np.cos(x))

    if window == 'gauss':
        wkr = np.exp(-(x ** 2) / 2)

    arrf = (karr * karr) * (wkr * wkr) * Pklinz0
    intval = sp.integrate.simps(arrf, karr)

    sig2m = (1 / (2 * (np.pi ** 2))) * intval
    sigm = np.sqrt(sig2m)

    return sigm


def sigRz0(R, karr, Pklinz0, window='gauss'):
    nk = len(karr)
    x = karr * R
    if window == 'tophat':
        wkr = (3 / (x * x * x)) * (np.sin(x) - x * np.cos(x))

    if window == 'gauss':
        wkr = np.exp(-(x ** 2) / 2)

    arrf = (karr * karr) * (wkr * wkr) * Pklinz0
    intval = sp.integrate.simps(arrf, karr)

    sig2m = (1 / (2 * (np.pi ** 2))) * intval
    sigm = np.sqrt(sig2m)

    return sigm


def get_sigmz0_array(Marr, karr, Pklinz0, rhom=get_rhom(), window='tophat'):
    sigmarr = np.zeros(len(Marr))
    for i in range(len(Marr)):
        M = Marr[i]
        sigmarr[i] = sigmz0(M, karr, Pklinz0, rhom, window)

    return sigmarr


def fst(nu, p=0.3, q=0.707):
    Ap = 1 / (1 + (sp.special.gamma(0.5 - p) / ((2 ** p) * np.sqrt(np.np.pi))))
    fstnu = Ap * np.sqrt(2 * q / np.pi) * (1 + (1 / (q * nu ** 2) ** p)) * nu * (np.exp((-q * nu ** 2 / 2)))
    return fstnu


def fmice(sig, z_mean, A0=0.58, a0=1.37, b0=0.3, c0=1.036, A0pow=-0.13, a0pow=-0.15, b0pow=-0.084, c0pow=-0.024):
    Amice = A0 * ((1 + z_mean) ** A0pow)
    amice = a0 * ((1 + z_mean) ** a0pow)
    bmice = b0 * ((1 + z_mean) ** b0pow)
    cmice = c0 * ((1 + z_mean) ** c0pow)
    valf = Amice * ((sig ** -amice) + bmice) * (np.exp(-cmice / (sig ** 2)))
    return valf


def dndm(Marr, z_mean, karr, Pklinz0, fnm='st', rhom=get_rhom(), Deltac=1.686, A0=0.58, a0=1.37, b0=0.3, c0=1.036,
         A0pow=-0.13, a0pow=-0.15, b0pow=-0.084, c0pow=-0.024, p=0.3, q=0.707):
    Dz = (1. / (1. + z_mean))
    deltac = Deltac / Dz
    pi = np.pi
    dndmarr = np.zeros(len(Marr))
    if fnm == 'st':
        for i in range(len(Marr)):
            M = Marr[i]
            sigmh = sigmz0(M, karr, Pklinz0, rhom, window='tophat')
            nuval = deltac / sigmh
            dndmarr[i] = fst(nuval, p, q)

    if fnm == 'mice':
        for i in range(len(Marr)):
            M = Marr[i]
            sigmh = sigmz0(M, karr, Pklinz0, rhom, window='tophat')
            dndmarr[i] = fmice(sigmh, z_mean, A0, a0, b0, c0, A0pow, a0pow, b0pow, c0pow)

    return dndmarr


def bnLmice(M, n, z_mean, karr, Pklinz0, rhom=get_rhom(), Deltac=1.686, A0=0.58, a0=1.37, b0=0.3, c0=1.036, A0pow=-0.13,
            a0pow=-0.15, b0pow=-0.084, c0pow=-0.024):
    # from mpmath import *
    # mp.dps = 15;
    # mp.pretty = True
    Amice = A0 * ((1 + z_mean) ** A0pow)
    amice = a0 * ((1 + z_mean) ** a0pow)
    bmice = b0 * ((1 + z_mean) ** b0pow)
    cmice = c0 * ((1 + z_mean) ** c0pow)
    Dz = (1. / (1. + z_mean))
    deltac = Deltac / Dz
    sigmh = sigmz0(M, karr, Pklinz0, rhom, window='tophat')
    nuval = deltac / sigmh
    fmiceval = (fmice(sigmh, z_mean, A0, a0, b0, c0, A0pow, a0pow, b0pow, c0pow))
    bnL = ((-1 / sigmh) ** n) * (
        diff(lambda nu: Amice * (((deltac / nu) ** -amice) + bmice) * (exp(-cmice / ((deltac / nu) ** 2))), nuval,
             n)) / fmiceval
    return bnL


def bnLst(M, n, z_mean, karr, Pklinz0, rhom=get_rhom(), Deltac=1.686, p=0.3, q=0.707):
    # from mpmath import *
    # mp.dps = 15
    # mp.pretty = True
    Ap = 1 / (1 + (sp.special.gamma(0.5 - p) / ((2 ** p) * np.sqrt(np.np.pi))))
    Dz = (1. / (1. + z_mean))
    deltac = Deltac / Dz
    sigmh = sigmz0(M, karr, Pklinz0, rhom, window='tophat')
    nuval = deltac / sigmh
    pi = np.pi
    fstval = (fst(nuval, p, q))
    bnL = ((-1 / sigmh) ** n) * (
        diff(lambda nu: Ap * sqrt(2 * q / pi) * (1 + (1 / (q * nu ** 2) ** p)) * nu * (exp((-q * nu ** 2 / 2))), nuval,
             n)) / fstval
    return bnL


def get_local_lag_bias_array(Marrh, z_mean, karr, Pklinz0, fnm='mice', orders=None):
    if orders is None:
        orders = [1, 2, 3]
    b1Larr = np.zeros(len(Marrh))
    b2Larr = np.zeros(len(Marrh))
    b3Larr = np.zeros(len(Marrh))
    if fnm == 'mice':
        for i in range(len(Marrh)):
            Mh = Marrh[i]
            b1Larr[i] = bnLmice(Mh, 1, z_mean, karr, Pklinz0)
            b2Larr[i] = bnLmice(Mh, 2, z_mean, karr, Pklinz0)
            b3Larr[i] = bnLmice(Mh, 3, z_mean, karr, Pklinz0)
    if fnm == 'st':
        for i in range(len(Marrh)):
            Mh = Marrh[i]
            b1Larr[i] = bnLst(Mh, 1, z_mean, karr, Pklinz0)
            b2Larr[i] = bnLst(Mh, 2, z_mean, karr, Pklinz0)
            b3Larr[i] = bnLst(Mh, 3, z_mean, karr, Pklinz0)
    if (1 in orders) & (2 in orders) & (3 in orders):
        return b1Larr, b2Larr, b3Larr

    if (1 in orders) & (2 in orders) & (3 not in orders):
        return b1Larr, b2Larr

    if (1 in orders) & (2 not in orders) & (3 not in orders):
        return b1Larr


def get_local_eul_bias_array(Marrh, z_mean, karr, Pklinz0, fnm='mice', orders=None):
    if orders is None:
        orders = [1, 2, 3]
    if (1 in orders) & (2 in orders) & (3 in orders):
        b1Larr, b2Larr, b3Larr = get_local_lag_bias_array(Marrh, z_mean, karr, Pklinz0, fnm, orders)
        b1Earr = 1. + b1Larr
        b2Earr = b2Larr + (8. / 21.) * b1Larr
        b3Earr = b3Larr - (13. / 7.) * b2Larr - (796. / 1323.) * b1Larr
        return b1Earr, b2Earr, b3Earr

    if (1 in orders) & (2 in orders) & (3 not in orders):
        b1Larr, b2Larr = get_local_lag_bias_array(Marrh, z_mean, karr, Pklinz0, fnm, orders)
        b1Earr = 1. + b1Larr
        b2Earr = b2Larr + (8. / 21.) * b1Larr
        return b1Earr, b2Earr

    if (1 in orders) & (2 not in orders) & (3 not in orders):
        b1Larr = get_local_lag_bias_array(Marrh, z_mean, karr, Pklinz0, fnm, orders)
        b1Earr = 1. + b1Larr
        return b1Earr


# Halofit functions
def _get_spec(k, delta_k, sigma_8):
    if sigma_8 < 1.0 and sigma_8 > 0.6:
        lnr = np.linspace(np.log(0.1), np.log(10.0), 500)
        lnsig = np.empty(500)

        for i, r in enumerate(lnr):
            R = np.exp(r)
            integrand = delta_k * np.exp(-(k * R) ** 2)
            sigma2 = _simps(integrand, np.log(k))
            lnsig[i] = np.log(sigma2)

    else:
        for r in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            integrand = delta_k * np.exp(-(k * r) ** 2)
            sigma2 = _simps(integrand, np.log(k))
            lnsig1 = np.log(sigma2)

            if lnsig1 < 0:
                try:
                    lnsig1 = lnsig_old
                except:
                    print("WARNING: LOWEST R NOT LOW ENOUGH IN _GET_SPEC. ln(sig) starts below 0: ", lnsig1)
                break

            lnsig_old = copy.copy(lnsig1)

        lnr = np.linspace(np.log(0.1 * r), np.log(r), 250)
        lnsig = np.empty(250)

        for i, r in enumerate(lnr):
            R = np.exp(r)
            integrand = delta_k * np.exp(-(k * R) ** 2)
            sigma2 = _simps(integrand, np.log(k))
            lnsig[i] = np.log(sigma2)

    r_of_sig = _spline(lnsig[::-1], lnr[::-1], k=5)
    rknl = 1.0 / np.exp(r_of_sig(0.0))

    sig_of_r = _spline(lnr, lnsig, k=5)
    try:
        dev1, dev2 = sig_of_r.derivatives(np.log(1.0 / rknl))[1:3]
    except Exception as e:
        print(
            "HALOFIT WARNING: Requiring extra iterations to find derivatives of sigma at 1/rknl (this often happens at high redshift).")
        lnr = np.linspace(np.log(0.2 / rknl), np.log(5 / rknl), 100)
        lnsig = np.empty(100)

        for i, r in enumerate(lnr):
            R = np.exp(r)
            integrand = delta_k * np.exp(-(k * R) ** 2)
            sigma2 = _simps(integrand, np.log(k))
            lnsig[i] = np.log(sigma2)
        lnr = lnr[np.logical_not(np.isinf(lnsig))]
        lnsig = lnsig[np.logical_not(np.isinf(lnsig))]
        if len(lnr) < 2:
            raise Exception("Lots of things went wrong in halofit")

        sig_of_r = _spline(lnr, lnsig, k=5)
        dev1, dev2 = sig_of_r.derivatives(np.log(1.0 / rknl))[1:3]

    rneff = -dev1 - 3.0
    rncur = -dev2

    return rknl, rneff, rncur


def halofit(k, delta_k, sigma_8, z=0, cosmo=Planck15, takahashi=1):
    # Get physical parameters
    rknl, neff, rncur = _get_spec(k, delta_k, sigma_8)

    # Only apply the model to higher wavenumbers
    mask = k > 0.005
    plin = delta_k[mask]
    k = k[mask]

    # Define the cosmology at redshift
    omegamz = cosmo.Om(z)
    omegavz = cosmo.Ode(z)

    w = cosmo.w(z)
    fnu = cosmo.Onu0 / cosmo.Om0

    if takahashi:
        a = 10 ** (1.5222 + 2.8553 * neff + 2.3706 * neff ** 2 +
                   0.9903 * neff ** 3 + 0.2250 * neff ** 4 +
                   - 0.6038 * rncur + 0.1749 * omegavz * (1 + w))
        b = 10 ** (-0.5642 + 0.5864 * neff + 0.5716 * neff ** 2 +
                   - 1.5474 * rncur + 0.2279 * omegavz * (1 + w))
        c = 10 ** (0.3698 + 2.0404 * neff + 0.8161 * neff ** 2 + 0.5869 * rncur)
        gam = 0.1971 - 0.0843 * neff + 0.8460 * rncur
        alpha = np.abs(6.0835 + 1.3373 * neff - 0.1959 * neff ** 2 +
                       - 5.5274 * rncur)
        beta = (2.0379 - 0.7354 * neff + 0.3157 * neff ** 2 +
                1.2490 * neff ** 3 + 0.3980 * neff ** 4 - 0.1682 * rncur +
                fnu * (1.081 + 0.395 * neff ** 2))
        xmu = 0.0
        xnu = 10 ** (5.2105 + 3.6902 * neff)

    else:
        a = 10 ** (1.4861 + 1.8369 * neff + 1.6762 * neff ** 2 +
                   0.7940 * neff ** 3 + 0.1670 * neff ** 4 +
                   - 0.6206 * rncur)
        b = 10 ** (0.9463 + 0.9466 * neff + 0.3084 * neff ** 2 +
                   - 0.94 * rncur)
        c = 10 ** (-0.2807 + 0.6669 * neff + 0.3214 * neff ** 2 - 0.0793 * rncur)
        gam = 0.8649 + 0.2989 * neff + 0.1631 * rncur
        alpha = np.abs(1.3884 + 0.3700 * neff - 0.1452 * neff ** 2)
        beta = (0.8291 + 0.9854 * neff + 0.3401 * neff ** 2)
        xmu = 10 ** (-3.5442 + 0.1908 * neff)
        xnu = 10 ** (0.9589 + 1.2857 * neff)

    if np.abs(1 - omegamz) > 0.01:
        f1a = omegamz ** -0.0732
        f2a = omegamz ** -0.1423
        f3a = omegamz ** 0.0725
        f1b = omegamz ** -0.0307
        f2b = omegamz ** -0.0585
        f3b = omegamz ** 0.0743
        frac = omegavz / (1 - omegamz)
        if takahashi:
            f1 = f1b
            f2 = f2b
            f3 = f3b
        else:
            f1 = frac * f1b + (1 - frac) * f1a
            f2 = frac * f2b + (1 - frac) * f2a
            f3 = frac * f3b + (1 - frac) * f3a
    else:
        f1 = f2 = f3 = 1.0

    y = k / rknl

    ph = a * y ** (f1 * 3) / (1 + b * y ** f2 + (f3 * c * y) ** (3 - gam))
    ph = ph / (1 + xmu / y + xnu * y ** -2) * (1 + fnu * (0.977 - 18.015 * (cosmo.Om0 - 0.3)))

    plinaa = plin * (1 + fnu * 47.48 * k ** 2 / (1 + 1.5 * k ** 2))
    pq = plin * (1 + plinaa) ** beta / (1 + plinaa * alpha) * np.exp(-y / 4.0 - y ** 2 / 8.0)
    pnl = pq + ph

    nonlinear_delta_k = delta_k.copy()
    nonlinear_delta_k[mask] = pnl

    return nonlinear_delta_k


def Pkhalofit(k, Pklinz0, Pklinz, z_mean, current_cosmo=Planck15):
    sigma_8_hf = sigRz0(8., k, Pklinz0, window='gauss')

    delta_k_lin = (k ** 3) * Pklinz / (2 * (np.pi ** 2))

    delta_k_hf = halofit(k, delta_k_lin, sigma_8_hf, z=z_mean, cosmo=current_cosmo)
    Pk_halofit = (2 * (np.pi ** 2)) * delta_k_hf / (k ** 3)

    return Pk_halofit


# cosmo_params = [h,Om0,Ob0,ns,sig8]
def get_PkNL_zarray(z_mean_array, karr, current_cosmo=Planck15, Pklinz0=None):
    H0 = current_cosmo.H0.value
    h = H0 / 100.0
    Omega_m = current_cosmo.Om0
    Ommh2 = Omega_m * ((H0 / 100) ** 2)
    Omega_b = current_cosmo.Ob0
    Ombh2 = Omega_b * ((H0 / 100) ** 2)
    Omega_c = Omega_m - Omega_b
    Omch2 = Omega_c * ((H0 / 100) ** 2)
    Omega_L = current_cosmo.Ode0
    Omlh2 = Omega_L * ((H0 / 100) ** 2)

    if current_cosmo == 'MICEcosmo':
        nsval = current_cosmo.ns
    else:
        nsval = 0.965

    Pklinz_z0_test = get_Pklinz(0.0, karr, current_cosmo=current_cosmo)

    sig8h = sigRz0(8., karr, Pklinz_z0_test, window='tophat')

    sig8_ratio = ((current_cosmo.sig8 / sig8h) ** 2)

    Pklinz0 = sig8_ratio * Pklinz_z0_test

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=Ombh2, omch2=Omch2, mnu=0.0)
    pars.InitPower.set_params(ns=nsval)
    pars.set_matter_power(redshifts=z_mean_array, kmax=np.amax(karr))

    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    karr, z_array, pklin_array = results.get_matter_power_spectrum(minkh=np.amin(karr), maxkh=np.amax(karr),
                                                                   npoints=len(karr))

    Pklinz_2d_mat = sig8_ratio * pklin_array

    PkNL_array = []

    for j in range(len(Pklinz_2d_mat)):
        PkNL = Pkhalofit(karr, Pklinz0, Pklinz_2d_mat[j], z_array[j], current_cosmo)
        PkNL_array.append(PkNL)

    return PkNL_array




def get_nbar_z(M_array, dndm_Mz_array, g1_mat, M_mat_cond_inbin):
    valf = sp.integrate.simps(dndm_Mz_array * g1_mat * M_mat_cond_inbin , M_array)
    return valf



def get_rhobar(M_array, dndm_array):
    valf = sp.integrate.simps(dndm_array * M_array, M_array)
    return valf


def get_rhobar_z(M_array, dndm_Mz_array):
    nz, nm = dndm_Mz_array.shape
    M_mat = np.tile(M_array, (nz, 1))
    valf = sp.integrate.simps(np.multiply(dndm_Mz_array, M_mat), M_array)
    return valf


def get_R_from_M(M_val, rhovir):
    rvir = np.power(3 * M_val / (4 * np.pi * rhovir), 1. / 3.)
    return rvir


def get_R_from_M_mat(M_mat, rhovir_array):
    nz, nm = M_mat.shape
    rho_vir_mat = np.tile(rhovir_array.reshape(nz, 1), (1, nm))
    rvir_mat = (3. * M_mat / (4. * np.pi * rho_vir_mat)) ** (1. / 3.)
    return rvir_mat


# fourier transform of halo mass profile : notation adhere to Cooray-Sheth '01
def get_ukm_g(r_max, k_val, conc):
    c = conc
    rs = r_max / c
    coeff = 1 / (np.log(1 + c) - (c / (1 + c)))
    (s1, c1) = sp.special.sici((1 + c) * k_val * rs)
    (s2, c2) = sp.special.sici(k_val * rs)
    sin1 = np.sin(k_val * rs)
    cos1 = np.cos(k_val * rs)
    sin2 = np.sin(c * k_val * rs)
    valf = coeff * (sin1 * (s1 - s2) - (sin2 / ((1 + c) * k_val * rs)) + cos1 * (c1 - c2))
    return valf


# fourier transform of halo mass profile : notation adhere to Cooray-Sheth '01
def get_ukmz_g_mat(r_max_mat, k_array, conc_mat,rsg_rs):
    nz, nm = conc_mat.shape
    k_mat = np.tile(k_array.reshape(nz, 1), (1, nm))
    rs_mat = rsg_rs * r_max_mat / conc_mat
    coeff = 1. / (np.log(1. + conc_mat) - (conc_mat / (1. + conc_mat)))
    (s1, c1) = sp.special.sici((1. + conc_mat) * rs_mat * k_mat)
    (s2, c2) = sp.special.sici(k_mat * rs_mat)
    sin1 = np.sin(k_mat * rs_mat)
    cos1 = np.cos(k_mat * rs_mat)
    sin2 = np.sin(conc_mat * k_mat * rs_mat)
    valf = coeff * (sin1 * (s1 - s2) - (sin2 / ((1. + conc_mat) * k_mat * rs_mat)) + cos1 * (c1 - c2))
    return valf


def get_nz_g_2mrs(z, m, n, z0):
    val = (n / (z0 * sp.special.gamma((m + 1.) / n))) * ((z / z0) ** m) * np.exp(-1. * (z / z0) ** n)
    return val


# NFW halo mass profile
def get_nfw_rm(r, m, halo_conc, rhovir):
    # mstar = 0.2985*np.power(10,13)
    c = halo_conc
    rvir3 = 3 * m / (4 * np.pi * rhovir)
    rvir = np.power(rvir3, 1. / 3.)

    if r < rvir:
        rs = rvir / c
        rhos = m / (4 * np.pi * (rs ** 3) * (np.log(1 + c) - (c / (1 + c))))
        denom = (r / rs) * ((1 + (r / rs)) * (1 + (r / rs)))
        rhorm = rhos / denom

    else:
        rhorm = 0

    return rhorm


# def rhoav(dndm, Marrh):
#     valf = sp.integrate.simps(dndm * Marrh, Marrh)
#
#     return valf


def get_hmf_from_halomod(Marr, z_mean):
    hm = HaloModel(z=z_mean)
    bmarr_hm = hm.bias
    Marr_hm = hm.m
    dndlnm_hm = hm.dndlnm
    dndm_hm = hm.dndm

    dndm_interp = interpolate.splrep(Marr_hm, dndm_hm, s=0)
    dndlnm_interp = interpolate.splrep(Marr_hm, dndlnm_hm, s=0)
    bmarr_interp = interpolate.splrep(Marr_hm, bmarr_hm, s=0)

    dndmarr = interpolate.splev(Marr, dndm_interp, der=0)
    dndlnmarr = interpolate.splev(Marr, dndlnm_interp, der=0)
    bmarr = interpolate.splev(Marr, bmarr_interp, der=0)

    return dndmarr, bmarr


def P2hm_th(k, karr, Marr, dndmarr, b1Larr, b2Larr, b3Larr, z_mean, Pk_halofit, Pb1L_NL, Pb2L_NL, usesig='False',
            sigmarr=None):
    if sigmarr is None:
        sigmarr = []
    if usesig == False:
        sigmarr = np.zeros(len(Marr))
    bkmarrh = np.zeros(len(Marr))
    jh = np.where(karr == k)[0]

    for i in range(len(Marr_h)):
        Mh = Marr[i]

        sigmh = sigmarr[i]

        b1L = b1Larr[i]
        b2L = b2Larr[i]
        b3L = b3Larr[i]

        PmXNL = (1 + b1L + b2L * sigmh ** 2 + b3L * sigmh ** 2) * (Pk_halofit[jh]) + b1L * Pb1L_NL[jh] + b2L * Pb2L_NL[
            jh]
        b1eff = PmXNL / (Pk_halofit[jh])
        bkmarrh[i] = b1eff

    rhobarh = sp.integrate.simps(dndmarr, Marr)
    toint = dndmarr * bkmarrh / rhobarh

    val1 = sp.integrate.simps(toint, Marr)

    valf = val1 * Pk_halofit[jh]

    return valf


def P2hh_th(k, karr, Marr, dndmarr, b1Larr, b2Larr, b3Larr, z_mean, Pk_halofit, Pb1L, Pb1L2, Pb1Lb2L, Pb2L, Pb2L2,
            usesig='False',
            sigmarr=None):
    if sigmarr is None:
        sigmarr = []
    jh = np.where(karr == k)[0]

    if usesig == False:
        sigmarr = np.zeros(len(Marr))
    bkmarrh = np.zeros(len(Marr))

    PXXNL = np.zeros((len(Marr), len(Marr)))
    for i1 in range(len(Marr)):
        for i2 in range(len(Marr)):
            Mh1 = Marr[i1]
            Mh2 = Marr[i2]

            sigmh1 = sigmarr[i1]
            sigmh2 = sigmarr[i2]

            b1L1 = b1Larr[i1]
            b2L1 = b2Larr[i1]
            b3L1 = b3Larr[i1]

            b1L2 = b1Larr[i2]
            b2L2 = b2Larr[i2]
            b3L2 = b3Larr[i2]

            PXXNL[i1, i2] = ((1 + b1L1) * (1 + b1L2) + 2 * (1 + b1L1) * (b2L1 + b3L1) * sigmh1 ** 2 + 2 * (1 + b1L2) * (
                    b2L2 + b3L2) * sigmh2 ** 2) * (Pk_halofit[jh]) \
                            + (b1L1 / 2. + b1L2 / 2.) * Pb1L[jh] + (b1L1 * b1L2) * (Pb1L2[jh]) + (
                                    b1L1 * b2L2 / 2. + b1L2 * b2L1 / 2.) * Pb1Lb2L[jh] \
                            + (b2L1 / 2. + b2L2 / 2.) * Pb2L[jh] + (b2L1 * b2L2) * Pb2L2[jh]

    dndmmat = (dndmarr[:, np.newaxis]) * dndmarr[:, np.newaxis].T

    rhobarh = sp.integrate.simps(dndmarr, Marr)
    toint = np.multiply(dndmmat, PXXNL)

    valf = (sp.integrate.simps(sp.integrate.simps(toint, Marr), Marr)) / (rhobarh ** 2)

    return valf


# def calculteXi_NN3d(xyz_radecr_X1,xyz_radecr_X2,):

def get_corrfunc_realspace(r, karr, Pkarr):
    toint = (karr ** 2) * Pkarr * (np.sin(karr * r) / (karr * r))
    val = sp.integrate.simps(toint, karr)
    valf = (1 / (2 * pi ** 2)) * val
    return valf


def get_wprp(rp, karr, Pkarr):
    val1 = sp.special.jv(0, karr * rp)
    valf = (sp.integrate.simps(karr * Pkarr * val1, karr)) / (2 * pi)
    return valf

def project_corr(rp, x_array, xi_x):
    xi_x_interp = sp.interpolate.interp1d(x_array,xi_x)
    x_new = np.logspace(np.log10(rp+ 0.1),np.log10(np.max(x_array)),1000)
    xi_val = xi_x_interp(x_new)
    val1 = x_new/(np.sqrt(x_new**2 - rp**2))
    valf = 2.* (sp.integrate.simps(xi_val * val1, x_new))
    # pdb.set_trace()
    return valf


def get_Cl(l, theta_array, xitheta_array):
    val1 = sp.special.jv(0, theta_array * l)
    valf = (2 * pi) * (sp.integrate.simps(theta_array * xitheta_array * val1, theta_array))
    return valf


def get_nz_tophat(zarray, zmin, zmax):
    ind_z = np.where((zarray > zmin) & (zarray < zmax))[0]
    valf = np.zeros(len(zarray))
    valf[ind_z] = 1. / (zmax - zmin)
    return valf


def get_Cl_limber(l, zarray, nzbin1, nzbin2, Pkarrz_interp_object, Hz_array, Dcom_array):
    c = 3 * 10 ** 5
    val_int = nzbin1 * nzbin2 * (Hz_array / (c * Dcom_array ** 2)) * (
        Pkarrz_interp_object.ev(zarray, (l + 1. / 2.) / Dcom_array))
    valf = sp.integrate.simps(val_int, zarray)
    return valf


def get_corr(cov):
    corr = np.zeros(cov.shape)
    for ii in range(cov.shape[0]):
        for jj in range(cov.shape[1]):
            corr[ii,jj] = cov[ii,jj]/np.sqrt(cov[ii,ii]*cov[jj,jj])
    return corr

def get_wplin_interp(nu, pkzlin_interp):
    k_array = np.logspace(-5, 3, 100000)
    z_array = np.logspace(-3,1,100)
    Pklinz = (pkzlin_interp.ev(np.log(z_array), np.log(k_array)))
    
    import pdb; pdb.set_trace()
    theta_out, xi_out = Hankel(k_array, nu=nu, q=1.0)(Pklinz, extrap=True)
    for j in range(z_array):
        theta_out, xi_out = Hankel(k_array, nu=nu, q=1.0)(Pklinz, extrap=True)

# @jit(nopython=True,parallel = True)
# @vectorize([float64(float64, float64,float64, float64)])
# def get_integrand_uyl(l,x_mat,y3d_mat,l500c_mat):
#     temp_mat = l * x_mat / l500c_mat
#     val = (x_mat ** 2) * y3d_mat * np.sin(temp_mat) / temp_mat
#     return val

