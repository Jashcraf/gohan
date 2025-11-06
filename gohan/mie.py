from gohan.gohan_math import np, riccati_psi_xi
from gohan.config import config
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
    medium_index: real float
        refractive index of the medium. 
    particle_index: complex float
        refractive index of the dielectric particle. Can be real-valued

    Returns
    -------
    float
        The relative index experienced between the particle and medium
    """
    if particle_index.imag == 0: 
        return particle_index.real / medium_index 
    else:
        return particle_index.real / medium_index + 1j * particle_index.imag / medium_index


# apply memorization to avoid re-calculating values of pi
#@lru_cache(maxsize=None)
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
        return np.zeros_like(theta)
    elif n == 1:
        return np.ones_like(theta)
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
        argument over which to evaluate tau, typically the scattering angle

    Returns
    -------
    ndarray
        tau_n
    """
    first = n * np.cos(theta)
    second = (n + 1)
    return (first * pi_recursion(n, theta)) - (second * pi_recursion(n-1, theta))


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
    
    psi_n_eval_mx, psi_n_derivative_mx, _, _ = riccati_psi_xi(n, m*x)
    psi_n_eval_x, psi_n_derivative_x, xi_n_eval_x, xi_n_derivative_x = riccati_psi_xi(n, x)

    numerator = m * psi_n_eval_mx * psi_n_derivative_x - psi_n_eval_x * psi_n_derivative_mx
    denomenator = m * psi_n_eval_mx * xi_n_derivative_x - xi_n_eval_x * psi_n_derivative_mx
    a_n = numerator / denomenator

    numerator = psi_n_eval_mx * psi_n_derivative_x - m * psi_n_eval_x * psi_n_derivative_mx
    denomenator = psi_n_eval_mx * xi_n_derivative_x - m * xi_n_eval_x * psi_n_derivative_mx

    b_n = numerator / denomenator
    
    return a_n, b_n


def mie_summation_terms(n, medium_index, particle_radius, particle_index, wavelength, theta):
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
    theta: float or ndarray
        Scattering angle in radians
    
    Returns
    -------
    ndarray, ndarray
        The S1 and S2 summation terms for the amplitude scattering matrix
    """

    m = relative_index(medium_index, particle_index)
    x = size_parameter(particle_radius, particle_index, wavelength)

    a_n, b_n = mie_coefficients_ab(x, n, m)


    front = (2 * n + 1) / (n * (n + 1))
    S1_n = front * (a_n * pi_recursion(n, theta) + b_n * tau_recursion(n, theta))
    S2_n = front * (b_n * pi_recursion(n, theta) + a_n * tau_recursion(n, theta))
    
    return S1_n, S2_n


def amplitude_scattering_matrix(n_terms, medium_index, particle_radius,
                                particle_index, wavelength, theta):
    
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
    theta: float or ndarray
        Scattering angle in radians

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
                                         wavelength,
                                         theta)

        S1 = S1 + S1_n
        S2 = S2 + S2_n

    # Construct the matrix
    amp_scatter_matrix = np.array([[S1, 0], [0, S2]], dtype=config.precision_complex)

    return amp_scatter_matrix


def scattering_cross_section_terms(n, medium_index, particle_radius, particle_index, wavelength):
    """
    NOTE: I'm not toatally sure on the physical significance of sig_b and sig_c, the ocean optics
    article says that
    - Q_a is the fraction of incident energy that is absorbed
    - Q_b is the fraction of incident energy scattered into all directions
    - Q_c = Q_a + Q_b for total attenuation

    also that 
    sig_b = Q_b * A_s, where A_s is the area of the sphere = pi * rho^2
    sig_c = Q_c * A_s

    It looks like the single-scattering albedo is given by
    omega = sig_b / sig_c

    so I believe these are the scattering cross-sections for energy scattered into all directions,
    and total attenuated energy.

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
        The sigma_b and sigma_c summation terms for the scattering cross sections,
        units of m^2 per particle
    """

    m = relative_index(medium_index, particle_index)
    x = size_parameter(particle_radius, particle_index, wavelength)
    wave_vector = wavelength**2 / 2 / np.pi / medium_index**2
    
    sigma_b = 0
    sigma_c = 0

    for i in range(0, n):
        a_n, b_n = mie_coefficients_ab(x, i, m)
        sigma_b = sigma_b + wave_vector * (2*i + 1) * (np.abs(a_n)**2 + np.abs(b_n)**2)
        sigma_c = sigma_c + wave_vector * (2*i + 1) * np.real(a_n + b_n)

    return sigma_b, sigma_c


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


def compute_opacities(grain_sizes, grain_size_distribution, material_kwargs, grain_size_kwargs=None):
    """kappa, or the scattering / extinction opacity
    
    Parameters
    ----------
    grain_sizes: list or ndarray
        list of grain radii in centimeters to integrate over
    grain_size_distribution: list, ndarray, or callable of grain sizes
        size distribution of the grains, tends to be a power law. User
        can supply own values or a callable of grain sizes
    grain_size_kwargs: dict
        dictionary of keyword arguments for the grain_size_distribution
        callable, defaults to None
    material_kwargs: dict
        Keyword arguments for the material to input in the `scattering_cross_section_terms`
        method. 
    grain_size_kwargs: dict
        kappa_sca = kappa_sca + area * sigma_sca
        Keyword arguments for the grain size distrubution callable. If grain_size_distribution
        not a callable, defaults to None

    Returns
    -------
    float, float
        The scattering and extinction opacities

    """
    
    if isinstance(grain_size_distribution, Callable):
        grain_size_distribution = grain_size_distribution(grain_sizes, **grain_size_kwargs)
    

    kappa_sca = 0
    kappa_ext = 0
    for grain_size, distribution in zip(grain_sizes, grain_size_distribution):

        area = np.pi * grain_size**2 * distribution
        sigma_sca, sigma_ext = scattering_cross_section_terms(**material_kwargs)
        kappa_sca = kappa_sca + area * sigma_sca
        kappa_ext = kappa_ext + area * sigma_ext

    return kappa_sca, kappa_ext


