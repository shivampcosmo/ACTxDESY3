import sys, platform, os
import numpy as np,astropy.units as u
from astropy import constants as const
pi = np.pi

class mynew_cosmo:
    """docstring for MICE_cosmo"""
    def __init__(self, h,Om0,Ob0,ns,sig8):
        
        self.Om0 = Om0
        self.Ob0 = Ob0
        self.Ok0 = 0.0
        self.Ogamma0 = 0.0
        self.Onu0 = 0.0
        self.h = h
        H0 = (100.*h)*(u.km / u.s / u.Mpc)
        self.H0 = H0
        self.Ode0 = 1 - (Om0 + self.Ok0 + self.Ogamma0)
        cd0 = 3*(H0**2)/(8.*pi*const.G)
        self.critical_density0 = cd0.to(u.g / u.cm ** 3)
        self.sig8 = sig8
        self.ns = ns


    # @property
    def H0(self):
        """ Return the Hubble constant as an `~astropy.units.Quantity` at z=0"""
        return self.H0

    def Hz(self,z):
        """ Return the Hubble constant as an `~astropy.units.Quantity` at z=0"""
        valf = self.H0 * np.sqrt((self.Om0) * ((1. + z) ** 3) + self.Ode0)
        return 

    # @property
    def Om0(self):
        """ Omega matter; matter density/critical density at z=0"""
        return self.Om0

    # @property
    def Ode0(self):
        """ Omega dark energy; dark energy density/critical density at z=0"""
        return self.Ode0

    # @property
    def Ob0(self):
        """ Omega baryon; baryonic matter density/critical density at z=0"""
        return self.Ob0

    # @property
    def Odm0(self):
        """ Omega dark matter; dark matter density/critical density at z=0"""
        return self.Odm0

    # @property
    def Ok0(self):
        """ Omega curvature; the effective curvature density/critical density
        at z=0"""
        return self.Ok0
    
    # @property
    def critical_density0(self):
        """ Critical density as `~astropy.units.Quantity` at z=0"""
        return self.critical_density0

    def critical_density(self,z):
        valf = (3*(self.H0 * np.sqrt((self.Om0) * ((1. + z) ** 3) + self.Ode0))**2/(8.*pi*const.G)).to(u.g / u.cm ** 3)
        return valf

    # @property
    def Ogamma0(self):
        """ Omega gamma; the density/critical density of photons at z=0"""
        return self.Ogamma0

    # @property
    def Onu0(self):
        """ Omega nu; the density/critical density of neutrinos at z=0"""
        return self.Onu0

    def w(self,z):
        return -1.0

    def Om(self,z):
        return (self.Om0 * ((1. + z) ** 3))/((self.Om0) * ((1. + z) ** 3) + self.Ode0)

    def Ob(self, z):
        return (self.Ob0 * ((1. + z) ** 3))/((self.Om0) * ((1. + z) ** 3) + self.Ode0)

    def Ode(self, z):
        return self.Ode0/((self.Om0) * ((1. + z) ** 3) + self.Ode0)



