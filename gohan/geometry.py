from gohan.gohan_math import np

"""
The gohan coordinate system nominally assumes a cartesian coordinate system where
- z axis: axis from the observer to the star
- x, y axis: the axes transverse to the star-observer axis

These are converted into

"""


class StellarCoordinates:
    def __init__(self, x0, y0, z0):
        self.position = [x0, y0, z0]


def cart_to_cylindrical(x, y, z):
    """converts cartesian coordinates to cylindrical coordinates

    Parameters
    ----------
    x, y, z: ndarray
        orthogonal cartesian coordinates

    Returns
    -------
    r, theta, z: ndarray
        orthogonal cylindrical coordinates
    """
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta, z


def cylindrical_to_cart(r, theta, z):
    """converts cartesian coordinates to cylindrical coordinates

    Parameters
    ----------
    r, theta, z: ndarray
        orthogonal cylindrical coordinates

    Returns
    -------
    x, y, z: ndarray
        orthogonal cartesian coordinates
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


def spherical_to_cylindrical(r, theta, phi):
    """converts spherical coordinates to cylindrical coordinates

    Parameters
    ----------
    r: ndarray
        radial spherical coordinate
    theta: ndarray
        polar angle spherical coordinate, i.e. angle from the polar axis
        defined from [0, pi]
    phi: ndarray
        azimuthal angle spherical coordinate, defined from [0, 2pi]
        

    Returns
    -------
    r, theta, z: ndarray
        orthogonal cylindrical coordinates
    """

    z = r * np.cos(theta)

    # overwrites r 
    r = r * np.sin(theta)

    # overwrite theta
    theta = phi

    return r, theta, z
