import ee
from .base import EETileSource


class Slope(EETileSource):
    def get_ee_image(self):
        dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")
        return ee.Terrain.slope(dem)

    def __repr__(self):
        return f'Slope()'
