from gohan.gohan_math import np
from gohan.mie import pi_recursion, tau_recursion
import matplotlib.pyplot as plt

"""
Aim to re-create figure 4.3 from this book:
"""

NPTS = 10000
ns = np.arange(1, 6, 1)
angles = np.linspace(-180, 180, NPTS)
angles_rad = np.radians(angles)

fig, axs = plt.subplots(ncols=2, nrows=5, figsize=(5, 10), subplot_kw={'projection': 'polar'},
                        layout='constrained')

for i in range(5):
    # The tau plot
    r = np.abs(tau_recursion(int(ns[i]), angles_rad))
    theta = angles_rad + (r < 0) * np.pi
    axs[i, 0].plot(theta, r)
    
    # The pi plot
    r = np.abs(pi_recursion(int(ns[i]), angles_rad))
    theta = angles_rad + (r < 0) * np.pi
    axs[i, 1].plot(theta, r, label=f"n={ns[i]}")

plt.show()
