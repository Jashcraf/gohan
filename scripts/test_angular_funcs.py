from gohan.gohan_math import np
from gohan.mie import pi_recursion, tau_recursion
import matplotlib.pyplot as plt

"""
Aim to re-create figure 3 from this book:
https://www.oceanopticsbook.info/view/theory-electromagnetism/level-2/mie-theory-examples
"""

NPTS = 10000
ns = np.arange(1, 9, 1)
angles = np.linspace(0, 180, NPTS)
angles_rad = np.radians(angles)
colors = ["red", "limegreen", "blue", "tab:brown", "m", "orange", "cyan", "g"]

plt.figure()
plt.subplot(121)
plt.title("Pi Recursion")
for n, color in zip(ns, colors):
    pi_eval = pi_recursion(int(n), angles_rad)
    plt.plot(angles, pi_eval, label=f"n={n}", color=color)
plt.legend()
plt.xlim(0, 180)
plt.ylim(-40, 40)

plt.subplot(122)
plt.title("Tau Recursion")
for n, color in zip(ns, colors):
    tau_eval = tau_recursion(int(n), angles_rad)
    plt.plot(angles, tau_eval, label=f"n={n}", color=color)
plt.legend()
plt.xlim(0, 180)
plt.ylim(-100, 100)
plt.show()
