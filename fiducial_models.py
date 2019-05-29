"""
Define Fiducial Models for LEGA-C Mocks
"""

import os
import numpy as np

# set default models
from halotools.empirical_models import SubhaloModelFactory
from legac_mocks.sham_model import DeconvolveSHAM, HaloProps, CAMGalProp, TertiaryGalProps
from legac_mocks.galaxy_abundance_functions import Davidzon_2017_phi
from legac_mocks.galaxy_secondary_functions import prim_prop_nn
from legac_mocks.cosmos_galaxy_properties import cosmos_table as default_data

# some parameters to set for all models
Lbox = np.array([250.0,250.0,250.0])  # Vishnu box size
mstar_bins = 10**np.arange(5,12.0,0.1)  # stellar mass bins for CAM
mbary_bins = 10**np.arange(5,12.0,0.1)  # baryonic mass bins for CAM

####################################################################################
#### stellar mass based models
####################################################################################

#define galaxy selection function
def galaxy_selection_func(table, min_mass=10**9.1, max_mass=np.inf, prim_gal_prop='stellar_mass'):
    """
    define which galaxies should appear in the mock galaxy table
    Parameters
    ----------
    table : astropy.table object
        table containing mock galaxies
    """

    mask = (table[prim_gal_prop] >= min_mass) & (table[prim_gal_prop] < max_mass)
    return mask


####################################################################################
props = ('halo_vpeak', 'stellar_mass', 'halo_halfmass_scale_factor', 'ssfr', 1.0)

phi = Davidzon_2017_phi(type='all', redshift=1.0)
sm_model =  DeconvolveSHAM(stellar_mass_function = phi, scatter=0.15, redshift=1.0,
                           prim_galprop='stellar_mass', prim_haloprop='halo_vpeak', Lbox=Lbox)
additional_halo_properties = HaloProps()
ssfr_dist = prim_prop_nn(prim_prop='stellar_mass', sec_prop='ssfr')
ssfr_model = CAMGalProp('stellar_mass', mstar_bins, rho=1.0, redshift=1.0,
                         secondary_haloprop='halo_halfmass_scale_factor',
                         secondary_galprop='ssfr',
                         conditional_rvs=ssfr_dist.rvs,
                         additional_galprop='reference_idx')
props_to_allocate = default_data.keys()
props_to_allocate.remove('stellar_mass')
props_to_allocate.remove('ssfr')
tertiaty_props = TertiaryGalProps(default_data, 'reference_idx', props_to_allocate)
model_1a = SubhaloModelFactory(stellar_mass = sm_model,
                               haloprops = additional_halo_properties,
                               galaxy_ssfr = ssfr_model,
                               more_galaxy_properties = tertiaty_props,
                               galaxy_selection_func = galaxy_selection_func,
                               model_feature_calling_sequence = ('stellar_mass', 'galaxy_ssfr', 'haloprops',
                                                                 'more_galaxy_properties'))


