# -*- coding: utf-8 -*-

"""
callable stellar mass functions from the literature
"""

from __future__ import (division, print_function, absolute_import, unicode_literals)
import numpy as np

from astropy.table import Table
from astropy.modeling.models import custom_model

__all__ = ['Tomczak_2014_phi', 'Davidzon_2017_phi']


class Davidzon_2017_phi(object):
    """
    stellar mass function from Davidson et al. 2017, arXiv:1701.02734
    """
    
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        redshift : float
            default is 1.0
        
        type : string
            default is 'all'.  options are: 'all', 'active', 'passive'
        """
        
        self.publication = ['arXiv:1701.02734']
        
        self.littleh = 0.7
        
        if 'redshift' in kwargs:
            self.z = kwargs['redshift']
        else:
            self.z = 1.0
        
        if 'type' in kwargs:
            self.type=kwargs['type']
        else:
            self.type = 'all'
        
        #parameters table 1 all
        self.z_bins_all = np.array([0.2,0.5,0.8,1.1,1.5,2.0,2.5,3.0,3.5,4.5,5.5])
        self.phi1_all = 10**(-3)*np.array([1.187, 1.070, 1.428, 1.069, 0.969, 0.295, 0.228, 0.090, 0.016, 0.003])
        self.x1_all = np.array([10.78, 10.77, 10.56, 10.62, 10.51, 10.60, 10.59, 10.83, 11.10, 11.30])
        self.alpha1_all = np.array([-1.38, -1.36, -1.31, -1.28, -1.28, -1.57, -1.67, 1.76, -1.98, -2.11])
        self.phi2_all = 10**(-3)*np.array([1.92, 1.68, 2.19, 1.21, 0.64, 0.45, 0.21, 0,0,0])
        self.x2_all = self.x1_all
        self.alpha2_all = np.array([-0.43,0.03,0.51,0.29,0.82,0.07,-0.08,0,0,0])
        
        #parameters table 2 star-forming
        self.z_bins_sf = np.array([0.2,0.5,0.8,1.1,1.5,2.0,2.5,3.0,3.5,4.0,6.0])
        self.phi1_sf = 10**(-3)*np.array([2.410, 1.661, 1.739, 1.542, 1.156, 0.441, 0.441, 0.086, 0.052, 0.003])
        self.x1_sf = np.array([10.26,10.40,10.35,10.42,10.40,10.45,10.39,10.83,10.77,11.30])
        self.alpha1_sf = np.array([-1.29,-1.32,-1.29,-1.21,-1.24,-1.50,-1.52,-1.78,-1.84,-2.12])
        self.phi2_sf = 10**(-3)*np.array([1.30, 0.86, 0.95, 0.49,0.46,0.38,0.13,0,0,0])
        self.x2_sf = self.x1_all
        self.alpha2_sf = np.array([1.01,0.84,0.81,1.11,0.90,0.59,1.05,0,0,0])
        
        #parameters table 2 quiscent
        self.z_bins_q = np.array([0.2,0.5,0.8,1.1,1.5,2.0,2.5,3.0,3.5,4.0])
        self.phi1_q = 10**(-3)*np.array([0.098,0.012,1.724,0.757,0.251,0.068,0.028,0.010,0.004])
        self.x1_q = np.array([10.83,10.83,10.75,10.56,10.54,10.69,10.24,10.10,10.10])
        self.alpha1_q = np.array([-1.30,-1.46,-0.07,0.53,0.93,0.17,1.15,1.15,1.15])
        self.phi2_q = 10**(-3)*np.array([1.58,1.44,0,0,0,0,0,0,0])
        self.x2_q = self.x1_all
        self.alpha2_q = np.array([-0.39,-0.21,0,0,0,0,0,0,0])
        
        self.s_all = np.empty((10,), dtype=object)
        self.s_sf = np.empty((10,), dtype=object)
        self.s_q = np.empty((9,), dtype=object)
        
        for i in range(0,10):
            #define components of double Schechter function
            s1 = Log_Schechter(phi0=self.phi1_all[i], x0=self.x1_all[i], alpha=self.alpha1_all[i])
            s2 = Log_Schechter(phi0=self.phi2_all[i], x0=self.x2_all[i], alpha=self.alpha2_all[i])
            #create piecewise model
            self.s_all[i] = s1 + s2
        
        for i in range(0,10):
            #define components of double Schechter function
            s1 = Log_Schechter(phi0=self.phi1_sf[i], x0=self.x1_sf[i], alpha=self.alpha1_sf[i])
            s2 = Log_Schechter(phi0=self.phi2_sf[i], x0=self.x2_sf[i], alpha=self.alpha2_sf[i])
            #create piecewise model
            self.s_sf[i] = s1 + s2
        
        for i in range(0,9):
            #define components of double Schechter function
            s1 = Log_Schechter(phi0=self.phi1_q[i], x0=self.x1_q[i], alpha=self.alpha1_q[i])
            s2 = Log_Schechter(phi0=self.phi2_q[i], x0=self.x2_q[i], alpha=self.alpha2_q[i])
            #create piecewise model
            self.s_q[i] = s1 + s2

    def __call__(self, mstar):
        """
        stellar mass function from Davidson et al. 2017, arXiv:1701.02734
        
        Parameters
        ----------
        mstar : array_like
            stellar mass in units Msol/h^2
        
        Returns
        -------
        phi : nunpy.array
            number density in units h^3 Mpc^-3 dex^-1
        """
        
        # convert from h=1 to h=self.little_h
        mstar = mstar / self.littleh**2
        
        # take log of stellar masses
        mstar = np.log10(mstar)

        # convert from h=self.little_h to h=1.0
        if self.type=='all':
            i = np.searchsorted(self.z_bins_all,self.z)
            return self.s_all[i](mstar) / self.littleh**3
        elif self.type=='active':
            i = np.searchsorted(self.z_bins_sf,self.z)
            return self.s_sf[i](mstar) / self.littleh**3
        elif self.type=='passive':
            i = np.searchsorted(self.z_bins_q,self.z)
            return self.s_q[i](mstar) / self.littleh**3
        else:
            print('type not available')
    


class Tomczak_2014_phi(object):
    """
    stellar mass function from Tomczak et al. 2014, arXiv:1309.5972
    """
    
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        redshift : float
            default is 1.0
        
        type : string
            default is 'all'
        """
        
        self.publication = ['arXiv:1309.5972']
        
        self.littleh = 0.7
        
        if 'redshift' in kwargs:
            self.z = kwargs['redshift']
        else:
            self.z = 1.0
        
        if 'type' in kwargs:
            self.type=kwargs['type']
        else:
            self.type = 'all'
        
        #parameters table 2 all
        self.z_bins = np.array([0.2,0.5,0.75,1.0,1.25,1.5,2.0,2.5,2.5,3.0])
        self.phi1_all = 10**np.array([-2.54,-2.55,-2.56,-2.72,-2.78,-3.05,-3.80,-4.54])
        self.x1_all = np.array([10.78,10.70,10.66,10.54,10.61,10.74,10.69,10.74])
        self.alpha1_all = np.array([-0.98,-0.39,-0.37,0.30,-0.12,0.04,1.03,1.62])
        self.phi2_all = 10**np.array([-4.29,-3.15,-3.39,-3.17,-3.43,-3.38,-3.26,-3.69])
        self.x2_all = self.x1_all
        self.alpha2_all = np.array([-1.90,-1.53,-1.61,-1.45,-1.56,-1.49,-1.33,-1.57])
        
        #parameters table 2 star-forming
        self.phi1_sf = 10**np.array([-2.67,-2.97,-2.81,-2.98,-3.04,-3.37,-4.30,-4.95])
        self.x1_sf = np.array([10.59,10.65,10.56,10.44,10.69,10.59,10.58,10.61])
        self.alpha1_sf = np.array([-1.08,-0.97,-0.46,0.53,-0.55,0.75,2.06,2.36])
        self.phi2_sf = 10**np.array([-4.46,-3.34,-3.36,-3.11,-3.59,-3.28,-3.28,-3.71])
        self.x2_sf = self.x1_all
        self.alpha2_sf = np.array([-2.00,-1.58,-1.61,-1.44,-1.62,-1.47,-1.38,-1.67])
        
        #parameters table 2 quiscent
        self.phi1_q = 10**np.array([-2.76,-2.67,-2.81,-3.03,-3.36,-3.41,-3.59,-4.22])
        self.x1_q = np.array([10.75,10.68,10.63,10.63,10.49,10.77,10.69,9.95])
        self.alpha1_q = np.array([0.47,0.10,0.04,0.11,0.85,-0.19,0.37,0.62])
        self.phi2_q = 10**np.array([-5.21,-4.29,-4.40,-4.80,-3.72,-3.91,-6.95,-4.51])
        self.x2_q = self.x1_all
        self.alpha2_q = np.array([-1.97,-1.69,-1.51,-1.57,-0.54,-0.18,-3.07,-2.51])
        
        self.s_all = np.empty((8,), dtype=object)
        self.s_sf = np.empty((8,), dtype=object)
        self.s_q = np.empty((8,), dtype=object)
        
        for i in range(0,8):
            #define components of double Schechter function
            s1 = Log_Schechter(phi0=self.phi1_all[i], x0=self.x1_all[i], alpha=self.alpha1_all[i])
            s2 = Log_Schechter(phi0=self.phi2_all[i], x0=self.x2_all[i], alpha=self.alpha2_all[i])
            #create piecewise model
            self.s_all[i] = s1 + s2
        
        for i in range(0,8):
            #define components of double Schechter function
            s1 = Log_Schechter(phi0=self.phi1_sf[i], x0=self.x1_sf[i], alpha=self.alpha1_sf[i])
            s2 = Log_Schechter(phi0=self.phi2_sf[i], x0=self.x2_sf[i], alpha=self.alpha2_sf[i])
            #create piecewise model
            self.s_sf[i] = s1 + s2
        
        for i in range(0,8):
            #define components of double Schechter function
            s1 = Log_Schechter(phi0=self.phi1_q[i], x0=self.x1_q[i], alpha=self.alpha1_q[i])
            s2 = Log_Schechter(phi0=self.phi2_q[i], x0=self.x2_q[i], alpha=self.alpha2_q[i])
            #create piecewise model
            self.s_q[i] = s1 + s2
    
    def __call__(self, mstar):
        """
        stellar mass function from Tomczak et al. 2014, arXiv:1309.5972
        
        Parameters
        ----------
        mstar : array_like
            stellar mass in units Msol/h^2
        
        Returns
        -------
        phi : nunpy.array
            number density in units h^3 Mpc^-3 dex^-1
        """
        
        # convert from h=1 to h=self.little_h
        mstar = mstar / self.littleh**2
        
        # take log of stellar masses
        mstar = np.log10(mstar)
        
        i = np.searchsorted(self.z_bins,self.z)
        
        # convert from h=little_h to h=1.0
        if self.type=='all':
            return self.s_all[i](mstar) / self.littleh**3
        elif self.type=='star-forming':
            return self.s_sf[i](mstar) / self.littleh**3
        elif self.type=='quiescent':
            return self.s_q[i](mstar) / self.littleh**3
        else:
            print('type not available')


@custom_model
def Log_Schechter(x, phi0=0.001, x0=10.5, alpha=-1.0):
    """
    log schecter x function
    """
    x = np.asarray(x)
    x = x.astype(float)
    norm = np.log(10.0)*phi0
    val = norm*(10.0**((x-x0)*(1.0+alpha)))*np.exp(-10.0**(x-x0))
    return val
