import ee
from .base import EETileSource


class AbsoluteElevation(EETileSource):
    def get_ee_image(self):
        return ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")

    def get_dtype(self):
        return 'float32'

    def __repr__(self):
        return f'AbsoluteElevation()'
