from gohan.gohan_math import np
from gohan.mie import (
    size_parameter,
    relative_index,
    pi_recursion,
    tau_recursion,
    mie_coefficients_ab,
    mie_summation_terms,
    amplitude_scattering_matrix
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
    
    """
    From Table 4.1 in A&SBAS
    """
    x = 3
    medium_index = 1.
    particle_index = 1.33 - 1j * 1e-8
    m = relative_index(medium_index, particle_index)

    ns = [1, 2, 3]
    ans = [5.1631e-1 - 1j * 4.9973e-1,
           3.4192e-1 - 1j * 4.7435e-1,
           4.8467e-2 - 1j * 2.1475e-1]
    bns = [7.3767e-1 - 1j * 4.3990e-1,
           4.0079e-1 - 1j * 4.9006e-1,
           9.3553e-3 - 1j * 9.6269e-2 ]
    
    ans_test = []
    bns_test = []
    for n in ns:

        # Now compute the values with gohan
        an, bn = mie_coefficients_ab(x, n=n, m=m)
        ans_test.append(an)
        bns_test.append(bn)
    
    test = ans_test + bns_test
    truth = ans + bns
    
    # Let's try the example from miepython instead
    m = 4/3
    x = 50
    a_1 = 0.531105889295 - 0.499031485631 * 1j
    b_1 = 0.791924475935 - 0.405931152229 * 1j
    
    a1test, b1test = mie_coefficients_ab(x, n=1, m=m)
    print(a1test)
    print(b1test)

    truth = [a_1, b_1]
    test = [a1test, b1test]

    np.testing.assert_allclose(test, truth)
    


@pytest.mark.skip
def test_mie_summation_terms():
    """
    From a miepython demo where they calculate a_n b_n coefficients
    via upward and downward recurrence
    """

    
    for theta in thetas:

        S1, S2 = mie_summation_terms(NTERM,
                                     REFMED,
                                     SPHERE_RADIUS,
                                     REFRE,
                                     0.6328,
                                     np.radians(theta))
        
        test_S1.append(S1)
    

    np.testing.assert_allclose(test_S1, true_S1)


@pytest.mark.skip
def test_amplitude_scattering_matrix():
    
    thetas = [0, 18, 36]
    NTERM = 25

    true_S1 = [1., 0.356857, 0.0355355]
    test_S1 = []
    for theta in thetas:

        ASM = amplitude_scattering_matrix(NTERM,
                                         REFMED,
                                         SPHERE_RADIUS,
                                         REFRE,
                                         0.6328,
                                         np.radians(theta))

        print(ASM.shape)
        S1 = ASM[0, 0]
        S2 = ASM[1, 1]
        
        test_S1.append(S1)
    

    np.testing.assert_allclose(test_S1, true_S1)

if __name__ == "__main__":
    test_mie_coefficients_ab()
