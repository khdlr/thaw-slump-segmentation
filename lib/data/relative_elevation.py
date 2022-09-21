import ee
from .base import EETileSource


class RelativeElevation(EETileSource):
    def __init__(self, kernel_size=100, offset=30, factor=300):
        self.kernel_size = kernel_size
        self.offset = offset
        self.factor = factor

    def get_ee_image(self):
        dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")
        conv = dem.convolve(ee.Kernel.circle(self.kernel_size, 'meters'))
        diff = (dem
                .subtract(conv)
                .add(ee.Image.constant(self.offset))
                .multiply(ee.Image.constant(self.factor))
                .toInt16())
        return diff

    def get_dtype(self):
        return 'int16'

    def __repr__(self):
        return f'RelativeElevation(kernel_size={self.kernel_size}, offset={self.offset}, factor={self.factor})'
