import numpy as np
from warnings import warn
from scipy.special import spherical_jn, spherical_yn

# Physical constants
C = 2.998e10  # cm/s
H = 6.626e-27  # erg s
k_B = 1.381e-16  # erg/K
sigma_SB = 5.67e-5  # erg/cm^2/K^4/s
AU = 1.496e13  # cm
L_sun = 3.828e33  # erg/s
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
M_TO_CM = 1e-2
MM_TO_CM = 1e1
UM_TO_CM = 1e-4
NM_TO_CM = 1e-7

class BackendShim:
    """A shim that allows a backend to be swapped at runtime.
    Taken from prysm.mathops with permission from Brandon Dube
    """

    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == "_srcmodule":
            return self._srcmodule

        return getattr(self._srcmodule, key)


_np = np
np = BackendShim(_np)


def set_backend_to_numpy():
    """Convenience method to automatically configure katsu's backend to numpy."""
    import numpy

    np._srcmodule = numpy

    return


def set_backend_to_cupy():
    """Convenience method to automatically configure katsu's backend to cupy."""
    import cupy as cp

    np._srcmodule = cp

    return

def set_backend_to_jax():
    """Convenience method to automatically configure katsu's backend to jax."""
    import jax as jax

    jax.config.update("jax_enable_x64", True)
    np._srcmodule = jax.numpy

    return


def spherical_h1(n, z):
    """Spherical Hankel function of the n-th order and first kind
    """
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def riccati_psi_xi(n, z):
    """Computes the Riccati-Bessel Functions of the first and second kind,
    These are grouped because they tend to be called together
    
    TODO: Consider replacing n with np.arange(n+1) to pre-compute these functions

    Parameters
    ----------
    n:
    z:
    """

    # Spherical bessel function of the first kind
    jn = spherical_jn(n, z)
    jn_prime = spherical_jn(n, z, derivative=True)
    
    yn = spherical_yn(n, z)
    yn_prime = spherical_yn(n, z, derivative=True)
     
    # Eval Riccati-Bessel
    psi = z * jn
    chi = -z * yn
    xi = psi - 1j* chi

    # Derivatives from product rule
    psi_prime = jn + z * jn_prime
    chi_prime = -yn - z * yn_prime
    xi_prime = psi_prime - 1j*chi_prime

    return psi, psi_prime, xi, xi_prime


def _psi_n(n, z):
    """Compute the Riccati-Bessel Function of the first kind
    
    NOTE: This is a 'thin' wrapper over the scipy function to 
    1) maintain notation with the reference for Mie Theory and
    2) to handle cases where the different backends don't support the scipy functions

    Parameters
    ----------
    n: int
        maximum order of the function
    z: float or ndarray
        argument over which the function is evaluated
    
    Returns
    -------
    ndarray, ndarray:
        Riccati-Bessel function evaluated over z, and its derivative
    """
    warn("Function _psi_n is depreciated")
    return riccati_jn(n, z)


def _xi_n(n, z):
    """Compute the Riccati-Bessel Function of the second kind
    
    NOTE: This is a 'thin' wrapper over the scipy function to 
    1) maintain notation with the reference for Mie Theory and
    2) to handle cases where the different backends don't support the scipy functions

    Parameters
    ----------
    n: int
        maximum order of the function
    z: float or ndarray
        argument over which the function is evaluated
    
    Returns
    -------
    ndarray, ndarray:
        Riccati-Bessel function evaluated over z, and its derivative
    """
    warn("Function _xi_n is depreciated")
    return riccati_yn(n, z)
