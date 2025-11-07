import numpy as np
import pytest
from scipy.special import riccati_jn, riccati_yn

from gohan.gohan_math import (
    riccati_psi,
    riccati_xi,
    riccati_psi_der,
    riccati_xi_der,
    riccati_psi_xi
)

def test_riccati_psi():

    ns = [0, 1, 2]
    zs = [0.1, 0.5, 1., 2.]
    
    psi_test = []
    psi_truth = []
    for n in ns:
        for z in zs:
            scipy_eval, _ = riccati_jn(n, z)
            psi_truth.append(scipy_eval[-1])
            psi_test.append(riccati_psi(n, z))

    np.testing.assert_allclose(psi_test, psi_truth, rtol=1e-7)


def test_riccati_xi():

    ns = [0, 1, 2]
    zs = [0.1, 0.5, 1., 2.]
    
    psi_test = []
    psi_truth = []
    for n in ns:
        for z in zs:
            scipy_eval, _ = riccati_yn(n, z)
            psi_truth.append(scipy_eval[-1])
            psi_test.append(riccati_xi(n, z))

    np.testing.assert_allclose(psi_test, psi_truth, rtol=1e-7)


def test_riccati_psi_der():

    ns = [0, 1, 2]
    zs = [0.1, 0.5, 1., 2.]
    
    psi_test = []
    psi_truth = []
    for n in ns:
        for z in zs:
            _, scipy_eval = riccati_jn(n, z)
            psi_truth.append(scipy_eval[-1])
            psi_test.append(riccati_psi_der(n, z))

    np.testing.assert_allclose(psi_test, psi_truth, rtol=1e-7)


def test_riccati_xi_der():

    ns = [0, 1, 2]
    zs = [0.1, 0.5, 1., 2.]
    
    psi_test = []
    psi_truth = []
    for n in ns:
        for z in zs:
            _, scipy_eval = riccati_yn(n, z)
            psi_truth.append(scipy_eval[-1])
            psi_test.append(riccati_xi_der(n, z))

    np.testing.assert_allclose(psi_test, psi_truth, rtol=1e-7)


def test_riccati_psi_xi():

    ns = [0, 1, 2]
    zs = [0.1, 0.5, 1., 2.]
    truth_values = []
    test_values = []

    for n in ns:
        for z in zs:
            
            # These return for all orders up to n
            truth_psi, truth_psi_der = riccati_jn(n, z)
            truth_xi, truth_xi_der = riccati_yn(n, z)

            truth_psi = truth_psi[-1]
            truth_psi_der = truth_psi_der[-1]
            truth_xi = truth_xi[-1]
            truth_xi_der = truth_xi_der[-1]

            test_psi, test_psi_der, test_xi, test_xi_der = riccati_psi_xi(n, z)
            
            truth_values += [truth_psi, truth_psi_der, truth_xi, truth_xi_der]
            test_values += [test_psi, test_psi_der, test_xi, test_xi_der]
    np.testing.assert_allclose(test_values, truth_values)

if __name__ == "__main__":
    test_riccati_psi_xi()
