from gohan.gohan_math import np
from gohan.mie import (
    pi_recursion,
    tau_recursion
)


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
