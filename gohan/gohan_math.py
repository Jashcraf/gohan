import numpy as np
from warnings import warn
from scipy.special import spherical_jn, spherical_yn
from functools import lru_cache

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


def riccati_psi(n, z):
    """Riccati-Bessel function of the first kind
    """
    return z * spherical_jn(n, z)


def riccati_xi(n, z):
    """Riccati-Bessel function of the second kind

    NOTE: there are two expressions of this function depending on the
    chosen sign convention. We chose the one to match scipy and miepython
    """
    return z * spherical_yn(n, z)


def riccati_psi_der(n, z):
    """
    This is the method done by miepython, because gohan's implementation
    may not be working with complex arguments?
    """
    return (n + 1) * spherical_jn(n, z) - z * spherical_jn(n + 1, z)


def spherical_h1(n, z):
    """
    Spherical Hankel function of the first kind. This is the method done by
    miepython
    """
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def riccati_xi_der(n, z):
    """
    This is the method done by miepython, because gohan's implementation
    may not be working with complex arguments?
    """
    return 1/2 * (z * spherical_h1(n - 1, z) + spherical_h1(n, z) - z * spherical_h1(n + 1, z))


def _riccati_psi_der(n, z):
    """
    Old implementation of derivative, which matches scipy for arbitrary values
    """
    jn = spherical_jn(n, z)
    jnp = spherical_jn(n, z, derivative=True)
    return jn + z * jnp


def _riccati_xi_der(n, z):
    """
    Old implementation of derivative, which matches scipy for arbitrary values
    """
    yn = spherical_yn(n, z)
    ynp = spherical_yn(n, z, derivative=True)
    return (yn + z * ynp)


def log_deriv_psi(n, x, tol=1e-15, max_iter=10_000):
    """Uses Lentz's algorithm for computing the derivative of psi. This method
    converges in fewer iterations than Dave method:
    https://web.gps.caltech.edu/~vijay/Papers/Aerosol/MieAlgorithmPaper.pdf

    Parameters
    ----------
    n: int
        order to evaluate derivative
    x: float or ndarray
        argument of the logarithmic derivative

    Returns
    -------
    ndarray
        logarithmic derivative of the Riccati-Bessel function of the 
        nth order and first kind

    """

    REGULARIZE = 1e-30

    xinv = 2. / x
    alpha = (n + 1/2) * xinv
    aj = -(n + 1.5) * xinv
    alpha_j1 = aj + 1/alpha
    alpha_j2 = aj
    ratio = alpha_j1 / alpha_j2
    runratio = alpha * ratio

    while np.abs(np.abs(ratio) - 1) > 1e-12:
        aj = xinv - aj
        alpha_j1 = 1 / alpha_j1 + aj
        alpha_j2 = 1 / alpha_j2 + aj
        ratio = alpha_j1 / alpha_j2
        xinv = xinv * -1
        runratio = ratio * runratio

    return -n / x + runratio

@lru_cache(maxsize=1024)
def log_deriv_psi_cached(n, x):
    return log_deriv_psi(n, x)


def log_derive_psi_downward(n, x):
    """Computes the values of D_n -> D_0 by downward recurrence
    
    Parameters
    ----------
    n: int
        Order to evaluate logarithmic derivative at
    x: float
        Argument to pass to logarithmic derivative

    Returns
    -------
    ndarray
        logarithmic derivative of order n -> 0 evaluated at x
    """

    D_n = log_deriv_psi(n, x)
    D_ns = np.zeros(n)
    D_ns[0] = D_n

    for m in range(n, 0, -1):

        # Update D_n
        D_n = n / z - 1.0 / (D_n + n / z)
        D_ns[n - 1] = D_n

    return D_ns


def riccati_psi_xi(n, z):
    """Computes the Riccati-Bessel Functions of the first and second kind,
    These are grouped because they tend to be called together
    
    TODO: Xi_prime is wrong
    TODO: Consider replacing n with np.arange(n+1) to pre-compute these functions
    Parameters
    ----------
    n:
    z:
    """

    # Eval Riccati-Bessel
    psi = riccati_psi(n, z) 
    xi = riccati_xi(n, z)

    # Eval Derivatives
    psi_prime = riccati_psi_der(n, z) 
    xi_prime = riccati_xi_der(n, z)

    return psi, psi_prime, xi, xi_prime

