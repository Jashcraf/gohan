from gohan.gohan_math import np
from gohan.dust import vertical_profile
from gohan.raytrace import prop_rays
from gohan.mie import compute_opacities 

def henyey_greenstein_phase_function(theta, g=0):
    """
    Parameters
    ----------
    theta: ndarray
        array of scattering angles, radians
    g: float
        asymmetry parameter, must be -1 <= g <= 1. Defaults to 0, which is 
        isotropic scattering

    Returns
    -------
    """

    assert np.abs(g) <= 1, f"assymetry parameter g={g} invalid, must be |g| <= 1"

    numerator = 1 - g ** 2
    denomenator = (1 + g**2 - 2*g*np.cos(theta)) ** (3/2)
    return 1 / (4 * np.pi) * numerator / denomenator


def calculate_step_size(ray_x):
    """Determines the appropriate step size at each position
    For now, we can treat this as a single step of 0.1 AU

    Parameters
    ----------
    ray_x: ndarray
        N x 3 array of ray positions in cylindrical coordinates

    Returns
    -------
    ndarray
        N x 1 array of distances to propagate the rays to next

    """
    NRAYS = ray_x.shape[0]
    distances = 0.1 * np.ones(NRAYS)
    return distances


def propagate_until_scatter(ray_x, ray_k, kappa, density_function, max_steps, disk_kwargs, grid_kwargs, dust_kwargs):

    """

    Parameters
    ----------
    ray_x: ndarray
        N x 3 array of ray starting positions
    ray_k: ndarray
        N x 3 array of ray starting direction cosines
    kappa: complex float
        mass opacity of dust grains - from Mie / Rayleigh theory
    density_function: callable
        The dust density distribution function which is a function of
        r, theta, z. 
    max_steps: int
        The maximum number of ray tracing steps that the computation permits
    disk_kwargs: dictionary
        inputs for the following positional arguments to density_function:
        - rho_0
        - a_in
        - a_out
        - gamma
        - xi_0
        - Beta
        - semimajor_axis
        - eccentricity
    grid_kwargs: dictionary
        Limits for the computational grid:
        - rlim
        - thetalim
        - zlim
    dust_kwargs: dictionary
        inputs for generating the scattering opacities
        - grain_sizes
        - grain_size_distribution
        - material_kwargs
        - grain_size_kwargs

    Returns
    -------
    ndarray
        M x 3 array of rays that scattered, does not return rays that escape
        the computation grid
    """
    NRAYS = ray_x.shape[0]
    tau_target = -np.log(np.random.random(nrays))
    tau_accumulated = np.zeros(NRAYS)
    active = np.ones_like(tau_accumulated, dtype=bool)


    for _ in range(max_steps):
        if not np.any(active):
            break

        # Get density function
        rho_local = vertical_profile(ray_x, **disk_kwargs)

        # Fixed step size
        ds = calculate_step_size(ray_x[active])

        # Calculate scattering opacity
        kappa = compute_opacities(**dust_kwargs)

        # Accumulate optical depth
        dtau = kappa * rho_local * ds
        tau_accumulated[active] = tau_accumulated[active] + dtau

        # Check for scattering
        scattered = active & (tau_accumulated >= tau_target)

        # Check for escape
        escaped = check_grid_boundaries(ray_x[active], **grid_kwargs)
        # Terminate rays that escape or scatter
        active[active] = ~escaped
        active[active] = ~scattered

        # Propagate rays that haven't escaped or scattered
        ray_x[active] = prop_rays(ray_x, ray_k, ds)

    return ray_x[scattered]


