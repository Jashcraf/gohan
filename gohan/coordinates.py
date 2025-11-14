from gohan.gohan_math import np
from gohan.config import config

class SkyGrid:

    def __init__(self, disk_direction, num_radial, num_azimuthal, num_height):
        """
        
        Parameters
        ----------
        disk_direction: ndarray
            Direction cosines the disk is oriented in w.r.t. the observer
            direction specified in gohan.config.
        num_radial: int
            Number of samples along the radial direction, r
        num_azimuthal: int
            Number of samples along the azimuthal direction, theta
        num_height: int
            Number of samples along the vertical direction, z
        """

        self.disk_direction = disk_direction
        self.num_radial = num_radial
        self.num_azimuthal = num_azimuthal
        self.num_height = num_height

        self.compose_coordinates()

    def compose_coordinates(self):

        if config.radial_spacing == "linear":
            op = np.linspace
        elif config.radial_spacing == "logarithmic":
            op = np.linspace

        self.radial_coords = op(0, 1, self.num_radial)
        
        if config.azimuthal_spacing == "linear":
            op = np.linspace
        elif config.azimuthal_spacing == "logarithmic":
            op = np.linspace

        self.azimuthal_coords = op(0, 2 * np.pi, self.num_azimuthal)
        
        
        if config.height_spacing == "linear":
            op = np.linspace
        elif config.height_spacing == "logarithmic":
            op = np.linspace

        self.height_coords = op(-1, 1, self.num_height)

        # construct grid
        # Storing these in temporary variables because it takes a lot of space
        _r, _t, _z = np.meshgrid(self.radial_coords,
                                 self.azimuthal_coords,
                                 self.height_coords)

        self.radial_coords = _r
        self.azimuthal_coords = _t
        self.height_coords = _z
        
        # Clean up the intermediate vars
        del _r, _t, _z
