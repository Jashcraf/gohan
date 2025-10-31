from gohan.gohan_math import np

"""
The gohan coordinate system nominally assumes a cartesian coordinate system where
- z axis: axis from the observer to the star
- x, y axis: the axes transverse to the star-observer axis

These are converted into

"""


class StellarCoordinates:

    def __init__(self, x0, y0, z0):



def cart_to_cylindrical(x, y, z):
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta, z


def cylindrical_to_cart(r, theta, z):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z



