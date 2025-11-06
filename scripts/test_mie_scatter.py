"""
Try to recreate this Figure from Wikipedia
https://en.wikipedia.org/wiki/Mie_scattering#/media/File:N4wiki.svg
"""

import matplotlib.pyplot as plt
from gohan.mie import scattering_cross_section_terms
from gohan.gohan_math import np
from time import perf_counter
import importlib.resources as resources

MEDIUM_INDEX = 1.
PARTICLE_RADIUS = 0.3 # microns
SUMMATION_ORDER = 2

# Import silver sphere data
# ag = np.genfromtxt('https://refractiveindex.info/tmp/database/data/main/Ag/nk/Johnson.txt', delimiter='\t')
nname = "refractiveindex.info/tmp/database/data/main/Ag/nk/Johnson.txt"
ref = resources.files("gohan").joinpath(nname)
ag = np.genfromtxt(ref, delimiter="\t")

# data is stacked so need to rearrange
N = len(ag) // 2
ag_lam = ag[1:N, 0]
ag_mre = ag[1:N, 1]
ag_mim = ag[N + 1 :, 1]

plt.figure(figsize=(8, 4.5))
plt.scatter(ag_lam * 1000, ag_mre, s=2, color="blue")
plt.scatter(ag_lam * 1000, ag_mim, s=2, color="red")
plt.xlim(300, 800)
plt.ylim(0, 5)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Refractive Index")
plt.text(350, 0.5, "$m_{re}$", color="blue", fontsize=14)
plt.text(350, 2.2, "$m_{im}$", color="red", fontsize=14)

plt.title("Complex Refractive Index of Silver")

wavelengths = ag_lam
particle_index = ag_mre - 1j * ag_mim

absorb_cross = []
scatter_cross = []
extinct_cross = []

t1 = perf_counter()
for wvl, index in zip(wavelengths, particle_index):

    sigma_sca, sigma_ext = scattering_cross_section_terms(SUMMATION_ORDER,
                                                          MEDIUM_INDEX,
                                                          PARTICLE_RADIUS,
                                                          index,
                                                          wvl)
    
    sigma_abs = (sigma_ext - sigma_sca)
    scatter_cross.append(sigma_sca)
    extinct_cross.append(sigma_ext)
    absorb_cross.append(sigma_abs)

runtime = perf_counter() - t1

plt.figure()
plt.title(f"Time to compute {len(wavelengths)} wavelengths = {runtime:.2e}s")
plt.plot(wavelengths * 1e3, scatter_cross, label=r"$\sigma_{sca}$")
plt.plot(wavelengths * 1e3, extinct_cross, label=r"$\sigma_{ext}$")
plt.plot(wavelengths * 1e3, absorb_cross, label=r"$\sigma_{abs}$")
plt.xlabel("Wavelength, nm")
plt.ylabel("Scattering Cross-section, microns^2")
plt.legend()
plt.xlim(300, 800)
plt.show()
