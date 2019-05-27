"""
A set of utilites functions for creating mock surveys
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import sys
from halotools.mock_observables.mock_survey import ra_dec_z
from halotools.utils import normalized_vectors
from eco_mocks.survey_utils import inside_survey_area
from astropy.table import Table


__author__ = ['Duncan Campbell']
__all__ = ['nearest_nieghbors_1d', 'random_survey']


def nearest_nieghbors_1d(arr1, arr2, seed=None):
    """
    Find the indices of the nearest neighbors in `arr1` for each element in `arr2`.

    Parameters
    ----------
    arr1 : array_like

    arr2 : array_like

    seed : int, optional
        random seed

    Returns
    -------
    idx : numpy.array
        an array of length len(arr2) with indices into arr1

    Notes
    -----
    If there are repeat values in `arr1`, the elements to match to
    are chosen randomly amongst the repeat values.  Setting the
    `seed` argument will make the results for such cases reproducible.
    """

    np.random.seed(seed)

    arr1 = np.atleast_1d(arr1)
    arr2 = np.atleast_1d(arr2)

    n = len(arr1)

    sort_inds = np.argsort(arr1)

    # find repeat elements in arr1
    unq_x, n_x = np.unique(arr1[sort_inds], return_counts=True)
    n_x = np.repeat(n_x,n_x)  # number of times each element repeats

    # find the nearest ***larger*** value in arr1
    matching_inds_a = np.searchsorted(arr1[sort_inds], arr2)
    mask = (matching_inds_a>=n)
    matching_inds_a[mask] = n-1

    # find the nearest ***smaller*** value in arr1
    matching_inds_b = np.searchsorted(-1.0*arr1[sort_inds][::-1], -1.0*arr2)
    matching_inds_b = n-1-matching_inds_b
    mask = (matching_inds_b<0)
    matching_inds_b[mask] = 0

    # choose the match wehich results in the smallest difference
    da = np.fabs(arr1[matching_inds_a] - arr2)
    db = np.fabs(arr1[matching_inds_b] - arr2)
    matching_inds = np.where(da<=db, matching_inds_a, matching_inds_b)
    #matching_inds = matching_inds_a

    # account for repeats in arr1
    ran_int = np.round(np.random.random(len(matching_inds))*(n_x[matching_inds]-1))
    ran_int = ran_int.astype(int)
    matching_inds = matching_inds+ran_int

    return sort_inds[matching_inds]


def pad_box(coords, Lbox=130):
    """
    Tile a PBC simulation cube into 3x3x3 cube.
    Note the memory requirements to store this will by 27 times
    larger than that required to store the original data!
    """

    N = len(coords)
    new_coords = np.zeros((N*(3**3), 3))

    l = 0
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                dx = Lbox*i
                dy = Lbox*j
                dz = Lbox*k
                new_coords[l*N:(l+1)*N, 0] = coords[:, 0]+dx
                new_coords[l*N:(l+1)*N, 1] = coords[:, 1]+dy
                new_coords[l*N:(l+1)*N, 2] = coords[:, 2]+dz
                l = l+1

    return new_coords, np.tile(np.arange(0, N), 27)


def random_unit_vectors_3d(npts):
    r"""
    """

    ndim = 3
    x = np.random.normal(size=(npts,ndim), scale=1.0)
    r = np.sqrt(np.sum((x)**2, axis=-1))

    return (1.0/r[:,np.newaxis])*x


def random_rotation(coords, vels):
    """
    apply a random rotation to a set of coordinates and velocities
    """

    ran_direction = random_unit_vectors_3d(1)[0]
    ran_angle = np.random.random(size=1)*(np.pi)
    ran_rot = rotation_matrices_from_angles_3d(ran_angle, ran_direction)

    new_coords = rotate_vector_collection(ran_rot, coords)
    new_vels = rotate_vector_collection(ran_rot, vels)

    return new_coords, new_vels


def rotate_vector_collection(rotation_matrices, vectors, optimize=False):
    r""" Given a collection of rotation matrices and a collection of 3d vectors,
    apply each matrix to rotate the corresponding vector.

    Parameters
    ----------
    rotation_matrices : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of rotation matrices.
        If an array of shape (3, 3) is passed, all the vectors
        are rotated using the same rotation matrix.

    vectors : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

    Returns
    -------
    rotated_vectors : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

    Examples
    --------
    In this example, we'll randomly generate two sets of unit-vectors, `v0` and `v1`.
    We'll use the `rotation_matrices_from_vectors` function to generate the
    rotation matrices that rotate each `v0` into the corresponding `v1`.
    Then we'll use the `rotate_vector_collection` function to apply each
    rotation, and verify that we recover each of the `v1`.

    >>> npts = int(1e4)
    >>> v0 = normalized_vectors(np.random.random((npts, 3)))
    >>> v1 = normalized_vectors(np.random.random((npts, 3)))
    >>> rotation_matrices = rotation_matrices_from_vectors(v0, v1)
    >>> v2 = rotate_vector_collection(rotation_matrices, v0)
    >>> assert np.allclose(v1, v2)
    """

    # apply same rotation matrix to all vectors
    if np.shape(rotation_matrices) == (3, 3):
        return np.dot(rotation_matrices, vectors.T).T
    # rotate each vector by associated rotation matrix
    else:
        try:
            return np.einsum('ijk,ik->ij', rotation_matrices, vectors, optimize=optimize)
        except TypeError:
            return np.einsum('ijk,ik->ij', rotation_matrices, vectors)


def random_survey(mock, Lbox=130.0, survey='eco_a', seed=None):
    """
    create a random mock survey within a periodic box
    """

    np.random.seed(seed=seed)

    # put mock coordinates in halotools format
    coords = np.vstack((mock['x'], mock['y'], mock['z'])).T
    vels = np.vstack((mock['vx'], mock['vy'], mock['vz'])).T

    # pad the simulation
    new_coords, inds = pad_box(coords)
    new_vels = np.tile(vels, (27, 1))

    # choose a new coordionate center within the central box
    new_coords[:, 0] = new_coords[:, 0] - (1.5*Lbox) + ((np.random.random(size=1) *2-1) * (0.5*Lbox))
    new_coords[:, 1] = new_coords[:, 1] - (1.5*Lbox) + ((np.random.random(size=1) *2-1) * (0.5*Lbox))
    new_coords[:, 2] = new_coords[:, 2] - (1.5*Lbox) + ((np.random.random(size=1) *2-1) * (0.5*Lbox))

    # randomly rotate coordinate system
    new_coords, new_vels = random_rotation(new_coords, new_vels)

    # put into ra, dec, redshift space
    ra, dec, z = ra_dec_z(new_coords, new_vels)
    ra = np.degrees(ra)
    dec = np.degrees(dec)

    # keep only galaxies within the footprint
    mask = inside_survey_area(ra, dec, z, survey=survey)
    new_coords = new_coords[mask]
    new_vels = new_vels[mask]
    inds = inds[mask]
    ra = ra[mask]
    dec = dec[mask]
    z = z[mask]

    # create new table to store mock
    new_mock = Table([new_coords[:, 0],
                      new_coords[:, 1],
                      new_coords[:, 2],
                      new_vels[:, 0],
                      new_vels[:, 1],
                      new_vels[:, 2]],
                     names=['x', 'y', 'z',
                            'vx', 'vy', 'vz'])

    # add the remaining galaxy properties in the mock
    cols = list(mock.dtype.names)
    cols.remove('x')
    cols.remove('y')
    cols.remove('z')
    cols.remove('vx')
    cols.remove('vy')
    cols.remove('vz')

    for name in cols:
        new_mock[name] = mock[name][inds]

    # add ra, dec, and redshift
    new_mock['ra'] = ra
    new_mock['dec'] = dec
    new_mock['redshift'] = z

    # remove excess
    r = np.sqrt(new_mock['x']**2+new_mock['y']**2+new_mock['z']**2)
    mask = (r < Lbox)

    return new_mock[mask]

