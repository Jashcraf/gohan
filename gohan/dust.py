from gohan.gohan_math import np
from gohan.config import config


def eccentric_ring_density(x, y, z, r0=50, e=0.1, omega=0, sigma_r=5, H0=0.05):
    """ Simulates an eccentric ring debris disk density using a Henry-Greenstein Phase Function

    Parameters
    ----------
    x, y, z: ndarrays
        3D coordinates

    """
    phi = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    r_center = r0 * (1 - e**2) / (1 + e * np.cos(phi - omega))
    H = H0 * r
    rho_r = np.exp(-0.5 * ((r - r_center)/sigma_r)**2)
    rho_z = np.exp(-0.5 * (z / H)**2)
    return rho_r * rho_z


def reference_radius(theta, semimajor_axis, eccentricity):
    """
    Parameters
    ----------
    theta: ndarray
        Azimuthal coordinate (in cylindrical coordinates) for the debris disk
    semimajor_axis: float
        Semi-major axis of the elliptical debris disk in <TODO UNITS>
    eccentricity: float
        Eccentricity of the elliptical debris disk

    Returns
    -------
    ndarray
        Reference radius evaluated for each azimuthal position on the debris disk
    """

    numerator = semimajor_axis * (1 - eccentricity**2)
    denomenator = 1 + eccentricity * np.cos(theta)
    return numerator / denomenator


def scale_height(r, theta, xi_0, Beta, semimajor_axis, eccentricity, return_reference=False):
    """
    Parameters
    ----------
    r: ndarray
        radial coordinate (in cylindrical coordinates) for the debris disk
    theta: ndarray
        Azimuthal coordinate (in cylindrical coordinates) for the debris disk
    xi_0: float
        Scale height at the reference radius
    Beta: float
        Flaring coefficient
    semimajor_axis: float
        Semi-major axis of the elliptical debris disk in <TODO UNITS>
    eccentricity: float
        Eccentricity of the elliptical debris disk
    return_reference: bool
        Whether to also return the reference radius, defaults to False

    Returns
    -------
    ndarray
       scale height on the debris disk, 
    R_theta: ndarray
        Reference radius computed for the scale height, returns if return_reference=True
    """

    R_theta = reference_radius(theta, semimajor_axis, eccentricity)
    if return_reference:
        return xi_0 * (r / R_theta) ** Beta, R_theta
    else:
        return xi_0 * (r / R_theta) ** Beta


def vertical_profile(r, theta, z, rho_0=1, a_in=0.1, a_out=1, gamma=2, xi_0=1, Beta=1, semimajor_axis=1, eccentricity=0):
    """
    Parameters
    ----------
    r: ndarray
        radial coordinate (in cylindrical coordinates) for the debris disk
    theta: ndarray
        Azimuthal coordinate (in cylindrical coordinates) for the debris disk
    z: ndarray
        height coordinate (in cylindrical coordinates) for the debris disk
    rho_0: float
        reference radius of the debris disk at the midplane
    a_in: float
        inner power law exponent describing dust distribution
    a_out: float
        outer power law exponent describing dust distribution
    gamma: float
        Decay exponent, gamma=2 corresponds to a Gaussian profile
    xi_0: float
        Scale height at the reference radius
    Beta: float
        Flaring coefficient
    semimajor_axis: float
        Semi-major axis of the elliptical debris disk in <TODO UNITS>
    eccentricity: float
        Eccentricity of the elliptical debris disk

    Returns
    -------
    ndarray
       vertical profile of the debris disk
    """

    H, R_theta = scale_height(r,
                              theta,
                              xi_0,
                              Beta,
                              semimajor_axis,
                              eccentricity,
                              return_reference=True)

    inner = (r / R_theta) ** (-2 * a_in)
    outer = (r / R_theta) ** (2 * a_out)
    decay_exponent = -1 * (z / H) ** gamma
    decay = np.exp(decay_exponent)

    return rho_0 * np.sqrt(2 / (inner * outer)) * decay


def size_distribution_power_law(a_min, a_max, power_law_index=-3.5, nsteps=10, spacing="log"):
    """Dohnanyi power law
    
    Parameters
    ----------
    a_min: float
        Grain size in units of distance
    a_max: float
        Grain size in units of distance
    power_law_index: float
        Index of the power law that parameterizes the size distribution,
        defaults to -3.5
    nsteps: int
        number of steps between the minimum and maximum grain size
    spacing: str
        the spacing of values between a_min and a_max, accepts "linear"
        or "log"

    Returns
    -------
    ndarray
        number density of particles evaluated at each particle size
    """
    spacing = spacing.lower()
    assert spacing in ["linear", "log"]
    
    if spacing == "linear":
        a = np.linspace(a_min, a_max, nsteps)
    elif spacing == "log":
        a = np.logspace(a_min, a_max, nsteps)

    return a ** power_law_index





class SkyGrid:
    def __init__(self, origin, n_samples):

        self.origin = origin
        self.n_samples = n_samples
        self.spacings = [
            config.radial_spacing,
            config.azimuthal_spacing,
            config.height_spacing
        ]

    def make_coordinates(self):
        
        # NOTE: Does this work if n_samples is an int?
        n_r, n_th, n_z = self.n_samples

        coordinates = [n_r, n_th, n_z]

        for coord, spacing in zip(coordinates, self.spacings):

            if spacing == "linear":
                space = np.linspace(0, coord)
            elif spacing == "logarithmic":
                space = np.logspace(0, coord)






