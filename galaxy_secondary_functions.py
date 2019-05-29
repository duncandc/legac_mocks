"""
objects to model secondary galaxy properties
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from astropy.table import Table
from legac_mocks.utils import nearest_nieghbors_1d
from legac_mocks.legac_galaxy_properties import legac_table as default_data


class prim_prop_nn(object):
    """
    primary galaxy property nearest neighbor model for secondary property.
    """

    def __init__(self, prim_prop='stellar_mass', sec_prop='ssfr', reference_table=None):
        """
        Parameters
        ----------
        prim_prop : string
            primary galaxy property

        sec_prop : string
            secondary galaxy property

        reference_table : astropy.table object
            reference table to search for nearest neighbors.  This table must have keys for
            `prim_prop` and `sec_prop`.
        """

        # set the reference table to search for nearest neighbors and
        # assiging the asscoaited vales for the secondary property
        if reference_table is None:
            self.data = default_data
        else:
            self.data = reference_table

        # make sure the reference table has the requested keys
        if prim_prop not in self.data.keys():
            msg = ('primary galaxy property not found in reference table.')
            raise ValueError(msg)
        else:
            self.prim_prop_key = prim_prop

        if sec_prop not in self.data.keys():
            msg = ('secondary galaxy property not found in reference table.')
            raise ValueError(msg)
        else:
            self.sec_prop_key = sec_prop

    def rvs(self, x, seed=None, return_index=False):
        """
        Return values of secondary galaxy property
        given values of the primary galaxy property.

        Parameters
        ----------
        x : array_like
            array of primary galaxy properties

        Returns
        -------
        y : numpy.array
            array of secondary galaxy properties

        idx : numpy.array
            if return_index is True, return the index from the reference catalog
        """

        idx = nearest_nieghbors_1d(self.data[self.prim_prop_key], x, seed=seed)

        if return_index:
            return self.data[self.sec_prop_key][idx], idx
        else:
            return self.data[self.sec_prop_key][idx]


