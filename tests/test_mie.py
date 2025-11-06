from gohan.gohan_math import np
from gohan.mie import (
    pi_recursion,
    tau_recursion
)

def test_pi_recursion():

    """
    Test cases for special angles
    """

    def pi_truth(n, costheta):
        return (costheta)**(n+1) * n * (n + 1) /2

    def pi_2_truth(costheta):
        return 3 * costheta

    def pi_3_truth(costheta):
        return (5 * costheta**2 + 1) / 2

    truth_values = []
    test_values = []
    ns = np.arange(1, 4, 1)
    
    for n in ns:
        truth = pi_truth(n=1, costheta=-1)
        truth_values.append(truth)

        test = pi_recursion(n=1, theta=np.radians(180))
        test_values.append(test)

    # For specific n's
    mus = np.linspace(-1, 1, 5)
    for mu in mus:
        truth = pi_2_truth(costheta=mu)
        truth_values.append(truth)

        test = pi_recursion(n=2, theta=np.arccos(mu))
        test_values.append(test)
        
    for mu in mus:
        truth = pi_3_truth(costheta=mu)
        truth_values.append(truth)

        test = pi_recursion(n=3, theta=np.arccos(mu))
        test_values.append(test)

    np.testing.assert_allclose(truth_values, test_values)



def test_tau_recursion():

    """
    Test cases for special angles
    """

    def tau_truth(n, costheta):
        return (costheta)**(n) * n * (n + 1) / 2
    
    def tau_2_truth(costheta):
        return 3 * costheta**2 - 1

    def tau_3_truth(costheta):
        return 5 * costheta**3 - 3 * costheta

    truth_values = []
    test_values = []
    ns = np.arange(1, 8, 1)
    
    for n in ns:
        truth = tau_truth(n=1, costheta=-1)
        truth_values.append(truth)

        test = tau_recursion(n=1, theta=np.radians(180))
        test_values.append(test)
    
    # For specific n's
    mus = np.linspace(-1, 1, 5)
    for mu in mus:
        truth = tau_2_truth(costheta=mu)
        truth_values.append(truth)

        test = tau_recursion(n=2, theta=np.arccos(mu))
        test_values.append(test)
        
    for mu in mus:
        truth = tau_3_truth(costheta=mu)
        truth_values.append(truth)

        test = tau_recursion(n=3, theta=np.arccos(mu))
        test_values.append(test)

    np.testing.assert_allclose(truth_values, test_values)
