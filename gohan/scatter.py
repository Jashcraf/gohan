from gohan.gohan_math import np

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
