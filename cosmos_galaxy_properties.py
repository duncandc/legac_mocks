"""
define galaxy properties for COSMOS survey
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import os
from astropy.io import ascii
from scipy.stats import norm
from astropy.table import Table, hstack

# load ECO data
data_path = os.path.dirname(os.path.realpath(__file__))+'/data/'

littleh = 0.7

# load catalog 1
fname = 'COSMOS2015_UVista_cat_withzCOSMOS_withChandra.fits'
data1 = Table.read(data_path + fname)

# load catalog 2
fname = 'COSMOS_0.5_2.5_cosmos2015_masses_toDuncan.fits'
data2 = Table.read(data_path + fname)

# combine catalogs
from halotools.utils import crossmatch
id1, id2 = crossmatch(data1['NUMBER'], data2['ID'])

data = hstack([data1[id1], data2[id2]])

# trim data
z_mask =(data['PHOTOZ']>0.75) & (data['PHOTOZ']<1.0)
mstar_mask = (data['MASS_SC'] > 9.4)
chi2_mask = data['CHI2_BEST'] < 2.5
kmag_mask = data['KMAG'] < 23.4
type_mask = data['TYPE_2']==0

selection_mask = z_mask & mstar_mask & chi2_mask & kmag_mask & type_mask 

data = data[selection_mask]

# process values
mstar = (10**data['MASS_SC'])*littleh**2  # stellar mass
sfr = (10**data['LGSFR_SC'])*littleh**2
ssfr = sfr/mstar
MB = data['MB'] - 5.0*np.log10(littleh)
MH = data['MH'] - 5.0*np.log10(littleh)
MI = data['MI'] - 5.0*np.log10(littleh)
MJ = data['MJ'] - 5.0*np.log10(littleh)
MK = data['MK'] - 5.0*np.log10(littleh)
MR = data['MR'] - 5.0*np.log10(littleh)
MU = data['MU'] - 5.0*np.log10(littleh)
MV = data['MV'] - 5.0*np.log10(littleh)
MY = data['MY'] - 5.0*np.log10(littleh)
MZ = data['MZ'] - 5.0*np.log10(littleh)

cosmos_values = np.vstack([mstar, ssfr, MB, MH, MI, MJ, MK, MR, MU, MV, MY, MZ]).T
cosmos_names = ['stellar_mass', 'ssfr',
                'Abs_Mag_B', 'Abs_Mag_H', 'Abs_Mag_I', 'Abs_Mag_J', 'Abs_Mag_K',
                'Abs_Mag_R', 'Abs_Mag_U', 'Abs_Mag_V', 'Abs_Mag_Y', 'Abs_Mag_Z']

cosmos_table = Table(cosmos_values, names=cosmos_names)


