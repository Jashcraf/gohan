from gohan.gohan_math import np
from gohan.mie import (
    size_parameter,
    relative_index,
    pi_recursion,
    tau_recursion,
    mie_summation_terms
)
import pytest

"""
based largely on Appendix A of
https://scatterlib.wdfiles.com/local--files/codes/app2_bohren.pdf
"""


REFMED = 1
REFRE = 1.55
SPHERE_RADIUS = 0.525
WAVELENGTH = 0.6328

pi_truth_values = [
    1, 3, 6, # mu = 1
    1, -3, 6, # mu = -1
    1, 3/np.sqrt(2), 9/4 # mu = 2 / sqrt(2)
]

# uses same mu values
tau_truth_values = [
    1, 3, 6,
    -1, 3, -6,
    1/np.sqrt(2), 0, -21/(4 * np.sqrt(2))
]

def test_size_parameter():
    TRUE_SIZE_PARAMETER = 5.213
    test_size_parameter = size_parameter(SPHERE_RADIUS, REFMED, WAVELENGTH)
    np.testing.assert_allclose(test_size_parameter, TRUE_SIZE_PARAMETER, rtol=1e-3)

def test_relative_index():
    true_relative_index = REFRE / REFMED
    test_relative_index = relative_index(REFMED, REFRE)
    np.testing.assert_allclose(test_relative_index, true_relative_index)

def test_pi_recursion():

    """
    Test cases for special angles
    """

    test_values = []
    ns = np.arange(1, 4, 1)
    
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
    
    np.testing.assert_allclose(test_values, pi_truth_values, rtol=1e-10, atol=1e-10)

    
def test_tau_recursion():

    """
    Test cases for special angles
    """

    test_values = []
    ns = np.arange(1, 4, 1)
    
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

    np.testing.assert_allclose(test_values, tau_truth_values, rtol=1e-10, atol=1e-10)

# Don't presently have a good test for this one
def test_mie_coefficients_ab():
    pass


def test_mie_summation_terms():
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
    true_s1 = 0.02452301+1j*0.29539154 
    true_s2 = -0.02452301-1j*0.29539154

    test = np.array([S1, S2])
    truth = np.array([true_s1, true_s2])    
    # miepython also uses norm=albedo and n_pole=0, which returns all terms
    # We have to chose a size parameter that matches
    np.testing.assert_allclose(test, truth)


if __name__ == "__main__":
    test_mie_summation_terms()
