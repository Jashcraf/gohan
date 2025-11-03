from gohan.gohan_math import np, GOLDEN_RATIO
from gohan.geometry import spherical_to_cylindrical
import numpy as truenp


def fibbonacci_lattice(npts):
    """Generates direction cosines that evenly sample a sphere
    along Fibonacci spirals

    Parameters
    ----------
    npts: int
        Number of points evenly distributed on a sphere

    Returns
    -------
    raydirs: ndarray
        N x 3 array of direction cosines in cylindrical coordinates
    """
    indices = np.arange(npts)
    phi = 2 * np.pi * indices / GOLDEN_RATIO

    # Elevation angle (from top to bottom)
    theta = np.arccos(1 - 2 * indices / npts)
    
    r, theta, z = spherical_to_cylindrical(np.ones(npts), theta, phi)

    return r, theta, z


def random_uniform_lattice(npts):
    """Generates direction cosines that randomly sample a sphere

    Parameters
    ----------
    npts: int
        Number of points evenly distributed on a sphere

    Returns
    -------
    raydirs: ndarray
        N x 3 array of direction cosines in cylindrical coordinates
    """
    u = np.random.uniform(0, 1, npts)
    v = np.random.uniform(0, 1, npts)
    
    theta = np.arccos(2*u - 1)  # elevation
    phi = 2 * np.pi * v         # azimuth
    
    r, theta, z = spherical_to_cylindrical(np.ones(npts), theta, phi)

    return r, theta, z


def gen_isotropic_raybundle(NRAYS):
    pass


def prop_rays(ray_x, ray_k, distance):
    """
    Propagate rays a given distance.

    Parameters
    ----------
    ray_x : ndarray
        Array of shape (N, 3) representing the positions of N rays.
    ray_k : ndarray
        Array of shape (N, 3) representing the direction cosines of N rays.
    distance : float, ndarray
        Distance to propagate the rays. Can be float or array of shape (N,).
    """
    if isinstance(distance, (int, float)):
        distance = np.full(ray_x.shape[0], distance)

    ray_x = ray_x + ray_k * distance[:, np.newaxis]

    return ray_x


def _scatter_henyey_greenstein(ray_k, g):

    """
    Scatter ray directions using the Henyey-Greenstein phase function.

    Parameters
    ----------
    ray_k : ndarray
        Array of shape (N, 3) representing the direction cosines of N rays.
    g : float
        Asymmetry parameter (-1 <= g <= 1).

    Returns
    -------
    ray_k : ndarray
        Updated array of shape (N, 3) with new directions.
    """
    N = ray_k.shape[0]
    u = np.random.uniform(0, 1, N)
    v = np.random.uniform(0, 1, N)

    # Scattering is Isotropic
    if g <= 1e-3:
        cos_theta = 1 - 2 * u

    # Apply phase function
    else:
        term = (1 - g**2) / (1 - g + 2 * g * u)
        cos_theta = (1 + g**2 - term**2) / (2 * g)

    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = 2 * np.pi * v

    new_kx = sin_theta * np.cos(phi)
    new_ky = sin_theta * np.sin(phi)
    new_kz = cos_theta

    ray_k[:, 0] = new_kx
    ray_k[:, 1] = new_ky
    ray_k[:, 2] = new_kz

    return ray_k


def scatter_rays(ray_k, method="Henyey-Greenstein", g=None):
    """
    Scatter ray directions. This function is presently a wrapper for
    _scatter_henyey_greenstein, but is constructed this way to allow for other
    scattering phase functions to be added in the future.

    Parameters
    ----------
    ray_k : ndarray
        Array of shape (N, 3) representing the direction cosines of N rays.
    method : str
        Scattering method. Currently only "Henyey-Greenstein" is implemented.
    g : float
        Asymmetry parameter for Henyey-Greenstein phase function (-1 <= g <= 1).

    Returns
    -------
    ray_k : ndarray
        Updated array of shape (N, 3) with new isotropic directions.
    """

    if method == "Henyey-Greenstein":
        assert g is not None, "Asymmetry parameter g must be provided for Henyey-Greenstein scattering."
        ray_k = _scatter_henyey_greenstein(ray_k, g)
    else:
        raise ValueError(f"Scattering method '{method}' not recognized.")

    return ray_k


def project_to_image_plane(ray_x, observer_dir, image_plane_center,
                           x_axis, y_axis, image_extent, nx, ny):
    """
    Projects a 3D position onto a 2D image plane.

    Parameters:
    -----------
    ray_x : ndarray
        Array of shape (N, 3) representing the positions of N rays.
    observer_dir : ndarray
        Array of shape (3,) representing the observer's direction.
    image_plane_center : ndarray
        Array of shape (3,) representing the center of the image plane.
    x_axis : ndarray
        Array of shape (3,) representing the x-axis of the image plane.
    y_axis : ndarray
        Array of shape (3,) representing the y-axis of the image plane.
    image_extent : float
        Physical size of the image plane (assumed square).
    Nx : int
        Number of pixels along the x-axis and y-axis.

    Returns: (ix, iy) pixel indices
    """

    r = ray_x - image_plane_center
    x_proj = np.dot(r, x_axis)
    y_proj = np.dot(r, y_axis)

    # TODO: Add option for spatial integration over a sample
    ix = int((x_proj + image_extent/2) / image_extent * nx)
    iy = int((y_proj + image_extent/2) / image_extent * ny)
    return ix, iy



