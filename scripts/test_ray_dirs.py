import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from gohan.raytrace import (
    fibbonacci_lattice,
    random_uniform_lattice,
    prop_rays
)
from gohan.gohan_math import np
from gohan.geometry import cylindrical_to_cart

NPTS = 1000

if __name__ == "__main__":

    for method in (fibbonacci_lattice, random_uniform_lattice):

        origin = np.array([0, 0, 0])
        origin = np.broadcast_to(origin, [1, *origin.shape])

        r, th, z = method(NPTS)
        x, y, z = cylindrical_to_cart(r, th, z)
        ray_k = np.array([x, y, z])
        ray_k = np.moveaxis(ray_k, 0, -1)

        ray_pos_p = prop_rays(origin, ray_k, 1)
        
        origin = origin[0]
        segments = np.array([[origin, endpoint] for endpoint in ray_pos_p])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        lc = Line3DCollection(segments, linewidths=5, alpha=0.6, colors="r")
        ax.add_collection(lc)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect([1, 1, 1])
        plt.show()

