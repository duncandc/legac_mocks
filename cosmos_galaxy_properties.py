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
    
# convert to h=1 scalings
data['MASS_SC'] = (10**data['MASS_SC'])*littleh**2
data['LGSFR_SC'] = (10**data['LGSFR_SC'])*littleh**2

# trim data
z_mask =(data['PHOTOZ']>0.75) & (data['PHOTOZ']<1.0)
mstar_mask = (data['MASS_SC'] > 10**9.4)
chi2_mask = data['CHI2_BEST'] < 2.5
kmag_mask = data['KMAG'] < 23.4
type_mask = data['TYPE_2']==0

selection_mask = z_mask & mstar_mask & chi2_mask & kmag_mask & type_mask 

data = data[selection_mask]

# process values
mstar = (10**(data['MASS_SC'])*littleh**2  # stellar mass
ssfr = (10**data['LGSFR_SC'])*littleh**2

cosmos_values = np.vstack([mstar, ssfr]).T
cosmos_names = ['stellar_mass', 'ssfr']

cosmos_table = Table(cosmos_values, names=cosmos_names)


