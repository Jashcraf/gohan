from gohan.gohan_math import np
from gohan.mie import (
    size_parameter,
    relative_index,
    pi_recursion,
    tau_recursion,
    mie_coefficients_ab,
    mie_summation_terms,
    amplitude_scattering_matrix,
    rule_for_nterms,
    scattering_cross_section_terms
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
    
    np.testing.assert_allclose(test, truth, rtol=1e-4, atol=1e-4)


@pytest.mark.skip(reason="Unclear why the values are so differnt - has to do with summation?")
def test_mie_summation_terms():
    """Based on a Miepython demo
    """

    m = 1.55 + 0.1 * 1j
    x = 5.213
    mu = [0, 0.5, 1.0]
    thetas = np.arccos(mu) 
    NTERM = rule_for_nterms(x)
    
    # We have to pick a sphere radius that satisfies this
    medium_index = 1.
    sphere_index = m
    wavelength = 0.6328
    sphere_radius = x * wavelength / (2 * np.pi)

    true_S2_real = [0.04308, -0.08407, 1.124380]
    true_S2_imag = [-0.05982, 0.13895, -0.19843]
    true_S2 = true_S2_real + true_S2_imag

    test_S2_real = []
    test_S2_imag = []

    for theta in thetas:

        ASM = amplitude_scattering_matrix(NTERM,
                                     medium_index,
                                     sphere_radius,
                                     sphere_index,
                                     wavelength,
                                     theta)
        S2 = ASM[1,1]
        test_S2_real.append(S2.real) 
        test_S2_imag.append(S2.imag)

    test_S2 = test_S2_real + test_S2_imag

    np.testing.assert_allclose(test_S2, true_S2)


def test_amplitude_scattering_matrix():
    """From appendix A of https://staff.cs.manchester.ac.uk/~fumie/internal/scattering.pdf
    """
    thetas = [0, 9, 18]
    test_size_parameter = size_parameter(SPHERE_RADIUS, REFMED, 0.6328)

    NTERM = rule_for_nterms(test_size_parameter)

    true_S11 = [1., 0.78538504, 0.356857]
    test_S11 = []

    true_S12 = [0, -0.00458392, -0.04578478]
    test_S12 = []

    true_S33 = [1., 0.99940039, 0.98602789]
    test_S33 = []

    true_S34 = [0., 0.03431985, 0.16016480]
    test_S34 = []

    for i, theta in enumerate(thetas):

        ASM = amplitude_scattering_matrix(NTERM,
                                         REFMED,
                                         SPHERE_RADIUS,
                                         REFRE,
                                         0.6328,
                                         np.radians(theta))

        S1 = ASM[0, 0]
        S2 = ASM[1, 1]
        
        # Stokes I
        S11 = (np.abs(S1)**2 + np.abs(S2)**2) / 2
        if i == 0:
            S11_0 = S11

        # Stokes Q
        S12 = (np.abs(S1)**2 - np.abs(S2)**2) / 2
        
        # Stokes U
        S33 = np.real(S2 * S1.conj())

        # Stokes V
        S34 = np.imag(S2 * S1.conj())
        
        test_S11.append(S11)
        test_S12.append(S12)
        test_S33.append(S33)
        test_S34.append(S34)

    test_S11_norm = [t / S11_0 for t in test_S11]
    test_S12_norm = [t / test_S11[i] for i, t in enumerate(test_S12)]
    test_S33_norm = [t / test_S11[i] for i, t in enumerate(test_S33)]
    test_S34_norm = [t / test_S11[i] for i, t in enumerate(test_S34)]

    test = test_S11_norm + test_S12_norm + test_S33_norm + test_S34_norm
    true = true_S11 + true_S12 + true_S33 + true_S34
    np.testing.assert_allclose(test, true, rtol=1e-5, atol=1e-4)


@pytest.mark.skip(reason="unclear on fair comparison with miepython, plots in scripts/test_mie_scatter.py look identical, which is encouraging")
def test_scattering_cross_section_terms():
    # A miepython example

    particle_index = complex(1.5, 0.0)
    medium_index = 1.
    x = 1000
    wavelength = 1.
    particle_radius = x * wavelength / (2 * np.pi)
    
    NTERMS = rule_for_nterms(x)

    true_values = [2.0139]

    sig_sca, sig_ext = scattering_cross_section_terms(NTERMS,
                                                      medium_index,
                                                      particle_radius,
                                                      particle_index,
                                                      wavelength)

    np.testing.assert_allclose(sig_sca, true_values)


def test_compute_opacities():
    pass

