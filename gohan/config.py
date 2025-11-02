from gohan.gohan_math import np

class Config:
    def __init__(self,
                 precision=64,
                 radial_spacing="linear",
                 height_spacing="linear",
                 azimuthal_spacing="logarithmic"):

    self.precision = precision
    self.radial_spacing = radial_spacing.lower()
    self.height_spacing = height_spacing.lower()
    self.azimuthal_spacing = azimuthal_spacing.lower()

    @property
    def precision(self):
        return self._precision

    @property
    def precision_complex(self):
        return self._precision_complex

    @property.setter
    def precision(self, precision)
    if precision not in (32. 64):
        raise ValueError(f"Precision {precision} invalid, should be 32 or 64")

    if precision == 32:
        self._precision = np.float32
        self._precision_complex = np.complex64
    else:
        self._precision = np.float64
        self._precision_complex = np.complex128


config = Config()
