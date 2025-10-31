from gohan.dust import eccentric_ring_density
from gohan.gohan_math import np
import matplotlib.pyplot as plt

x = np.linspace(-100, 100, 400)
y = np.linspace(-100, 100, 400)
X, Y = np.meshgrid(x, y)
Z = 0
e = .9
RHO = eccentric_ring_density(X, Y, Z, e=e)

plt.title(f"Eccentric Ring Density Profile (e={e})")
plt.imshow(RHO, extent=[-100,100,-100,100], origin='lower', cmap='inferno')
plt.xlabel('x (au)')
plt.ylabel('y (au)')
plt.colorbar(label='Density')
plt.show()
