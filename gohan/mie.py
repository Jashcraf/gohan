from gohan.gohan_math import np, psi_n, xi_n
from gohan.conf.py import config
from functools import lru_cache

"""
Tools to evaluate Mie scattering assuming spherical dielectric particles, from the Ocean Optics
article reviewing Mie Theory
https://www.oceanopticsbook.info/view/theory-electromagnetism/level-2/mie-theory-overview
"""


def size_parameter(particle_radius, refractive_index, wavelength):
    """
    Parameters
    ----------
    particle_radius: float
        particle radius in units of distance, should be same as wavelength
    refractive_index: float
        refractive index of the dielectric particle
    wavelength: float
        wavelength in units of distance, same as particle_radius

    Returns
    -------
    float
        size parameter of the particle at the specified wavelength
    """
    return 2 * np.pi * particle_radius * refractive_index / wavelength


def relative_index(medium_index, particle_index):
    """
    Parameters
    ----------
    medium_index: real or complex float
        (generally) complex refractive index of the medium. Can be real-valued
    particle_index:
        refractive index of the dielectric particle

    Returns
    -------
    float
        The relative index experienced between the particle and medium
    """
    return medium_index.real / particle_index + 1j * medium_index.imag / particle_index


# apply memorization to avoid re-calculating values of pi
@lru_cache(maxsize=None)
def pi_recursion(n, theta):
    """
    Parameters
    ----------
    n: int
        maximum order to evaluate the pi_n recursion at
    theta: ndarray
        argument over which to evaluate pi

    Returns
    -------
    ndarray
        pi_n
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1:
    else:
        first = (2*n - 1) / (n - 1) * np.cos(theta)
        second = n / (n - 1)
        return first * pi_recursion(n-1, theta) - second * pi_recursion(n-2, theta)


def tau_recursion(n, theta):
    """
    Parameters
    ----------
    n: int
        maximum order to evaluate the tau_n recursion at
    theta: ndarray
        argument over which to evaluate tau

    Returns
    -------
    ndarray
        tau_n
    """
    first = n * np.cos(theta)
    second = (n + 1)
    return first * pi_recursion(n, theta) - second * pi_recursion(n-1, theta)


def mie_coefficients_ab(x, n, m):
    """
    Parameters
    ----------
    x: float
        The size parameter of the particles, see `size_parameter` method
    n: int
        pole order of the Mie coefficient. n=1 is the dipole term, n=2 is the quadrapole term,
        and so on.
    m: complex float
        The relative refractive index between the dielectric sphere and medium. See
        `relative_index` method.


    Returns
    -------
    ndarray, ndarray
        The a_n and b_n coefficients from Mie Theory
    """
    psi_n_eval_mx, psi_n_deprivative_mx = psi_n(n, m*x)
    psi_n_eval_x, psi_n_derivative_x = psi_n(n, x)
    xi_n_eval_x, xi_n_derivative_x = xi_n(n, x)

    numerator = m * psi_n_eval_mx * psi_n_derivative_x - psi_n_eval_x * psi_n_derivative_mx
    denomenator = m * psi_n_eval_mx * xi_n_derivative_x - xi_n_eval_x * psi_n_derivative_mx
    a_n = numerator / denomenator

    numerator = psi_n_eval_mx * psi_n_derivative_x - m * psi_n_eval_x * psi_n_derivative_mx
    denomenator = psi_n_eval_mx * xi_n_derivative_x - m * xi_n_eval_x * psi_n_derivative_x

    b_n = numerator / denomenator

    return a_n, b_n


def mie_summation_terms(n, medium_index, particle_radius, particle_index, wavelength):
    """
    Parameters
    ----------
    n: int
        order to evaluate the summation terms at
    medium_index: float
        (generally) complex index of refraction the particles are immersed in
    particle_radius: float
        particle radius in units of distance, should be same as wavelength
    refractive_index: float
        refractive index of the dielectric particle
    wavelength: float
        wavelength in units of distance, same as particle_radius
    
    Returns
    -------
    ndarray, ndarray
        The S1 and S2 summation terms for the amplitude scattering matrix
    """

    m = get_relative_index(medium_index, particle_index)
    x = size_parameter(particle_radius, particle_index, wavelength)

    a_n, b_n = mie_coefficients(x, n, m)

    front = (2 * n + 1) / (n * (n + 1))
    S1_n = front * (a_n * pi_recursion(n) + b_n * tau_recursion(n))
    S2_n = front * (b_n * pi_recursion(n) + a_n * tau_recursion(n))
    
    return S1_n, S2_n


def amplitude_scattering_matrix(n_terms, medium_index, particle_radius,
                                particle_index, wavelength):
    
    """
    Parameters
    ----------
    n_terms: int
        maximum order to evaluate the summation terms at
    medium_index: float
        (generally) complex index of refraction the particles are immersed in
    particle_radius: float
        particle radius in units of distance, should be same as wavelength
    refractive_index: float
        refractive index of the dielectric particle
    wavelength: float
        wavelength in units of distance, same as particle_radius
    
    Returns
    -------
    ndarray
        The amplitude scattering matrix
    """
    S1 = 0
    S2 = 0

    for n in range(1, n_terms+1):
        S1_n, S2_n = mie_summation_terms(n,
                                         medium_index,
                                         particle_radius,
                                         particle_index,
                                         wavelength)

        S1 = S1 + S1_n
        S2 = S2 + S2_n

    # Construct the matrix
    amp_scatter_matrix = np.array([[S1, 0], [0, S2]], dtype=config.precision_complex)

    return amp_scatter_matrix


def rule_for_nterms(x):
    """From Ocean Optics article on Mie scattering,
    a rough rule for the number of terms needed for accurate
    simulation given a size parameter x

    Parameters
    ----------
    x: float
        size parameter of the particle simulated

    Returns
    -------
    ndarray
        minimum number of terms required for accurate simulation
    """
    return int(x + 4 * x**(1/3) + 2)


