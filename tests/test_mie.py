from gohan.gohan_math import np
from gohan.mie import (
    size_parameter,
    pi_recursion,
    tau_recursion,
    mie_summation_terms
)


def test_size_parameter():
    pass

def test_relative_index():
    pass

def test_pi_recursion():

    """
    Test cases for special angles
    """

    # Hand evaluated
    pi_truth = [
        1, 1, 1, 1, # costheta = 1
        -1, 1, -1, 1, # costheta = -1
        1, 1/(2 * np.sqrt(2)), (1 - 3 * np.sqrt(2))/8, (np.sqrt(2) - 15)/16 # costheta = 1/sqrt(2)
    ]
    test_values = []
    ns = np.arange(1, 5, 1)
    
    theta = 0
    for n in ns:
        test = pi_recursion(n=n, theta=np.radians(theta))
        test_values.append(test)
    
    theta = 180 
    for n in ns:
        test = pi_recursion(n=n, theta=np.radians(theta))
        test_values.append(test)
    
    theta = 45 
    for n in ns:
        test = pi_recursion(n=n, theta=np.radians(theta))
        test_values.append(test)

    np.testing.assert_allclose(test_values, pi_truth)

    
def test_tau_recursion():

    """
    Test cases for special angles
    """

    # Hand evaluated
    tau_truth = [
        1, 3, 6, 10, # costheta = 1
        -1, 3, -6, 10, # costheta = -1
        np.sqrt(2), 3*np.sqrt(2)/2, 2 * np.sqrt(2), 5*np.sqrt(2)/2 # costheta = 1/sqrt(2)
    ]
    test_values = []
    ns = np.arange(1, 5, 1)
    
    theta = 0
    for n in ns:
        test = tau_recursion(n=n, theta=np.radians(theta))
        test_values.append(test)
    
    theta = 180 
    for n in ns:
        test = tau_recursion(n=n, theta=np.radians(theta))
        test_values.append(test)
    
    theta = 45 
    for n in ns:
        test = tau_recursion(n=n, theta=np.radians(theta))
        test_values.append(test)

    np.testing.assert_allclose(test_values, tau_truth)


def test_mie_coefficients_ab():
    pass


def mie_summation_terms():
    """
    miepython demo
    https://miepython.readthedocs.io/en/latest/01_basics.html
    """
    particle_index = 1.507 - 0.002j
    sphere_size_parameter = 0.7086
    mu = -1.0
    
    # I just randomly chose a wavelength here oops
    # distances are now in nanometers
    particle_radius = sphere_size_parameter * 500 / 2 / np.pi # nanometers?
    theta = np.arccos(mu)

    S1, S2 = mie_summation_terms(2, 1, particle_radius, particle_index, 500, theta)

    true_s1 = 0.02452301+0.29539154j, 
    true_s2 = -0.02452301-0.29539154j

    np.testing.assert_allclose([S1, S2], [true_s1, true_s2])
    # miepython also uses norm=albedo and n_pole=0, which returns all terms
    # We have to chose a size parameter that matches
