from gohan.gohan_math import np


def eccentric_ring_density(x, y, z, r0=50, e=0.1, omega=0, sigma_r=5, H0=0.05):
    phi = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    r_center = r0 * (1 - e**2) / (1 + e * np.cos(phi - omega))
    H = H0 * r
    rho_r = np.exp(-0.5 * ((r - r_center)/sigma_r)**2)
    rho_z = np.exp(-0.5 * (z / H)**2)
    return rho_r * rho_z
