"""
create and save all fiducial mocks
"""
import os
from legac_mocks.fiducial_models import model_1a
import numpy as np
from astropy.io import ascii
import sys

cwd = os.getcwd()
save_path = cwd + '/mocks/'

def main():

    # load halo catalog
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    from halotools import sim_manager
    simname = 'bolplanck'
    halocat = sim_manager.CachedHaloCatalog(simname = simname, redshift=1.0, dz_tol = 0.1,
                                            version_name='halotools_v0p4', halo_finder='rockstar')

    # populate mock catalogs
    model_1a.param_dict['rho'] = 1.0
    model_1a.populate_mock(halocat)
    mock_1 = model_1a.mock.galaxy_table
    mock_1.write(save_path+'mock_1a_00.hdf5', format='hdf5', path='data', overwrite=True)


if __name__ == '__main__':
    main()
