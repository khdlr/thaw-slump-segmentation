import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio.features import rasterize
from einops import rearrange

from .base import TileSource


class Mask(TileSource):
    def __init__(self, geometries: gpd.GeoSeries, bounds: gpd.GeoSeries=None):
        self.geometries = geometries
        self.bounds = bounds

    def get_raster_data(self, scene) -> xr.DataArray:
        mask = rasterize(self.geometries.to_crs(scene.crs),
                out_shape=scene.size, transform=scene.transform)
        if self.bounds is not None:
          is_valid = rasterize(self.bounds.to_crs(scene.crs),
                  out_shape=scene.size, transform=scene.transform)

          # Set unlabelled regions to mask=255
          mask = np.where(is_valid, mask, 255)

        mask = rearrange(mask, 'H W -> 1 H W')

        mask = xr.DataArray(mask,
            coords={
              'mask_band': [1],
              **scene.get_coords()
            },
            attrs={
              'long_name': 'mask',
            }
        )
        mask = mask.rio.write_transform(scene.transform)
        mask = mask.rio.write_crs(scene.crs)
        return mask

    def __repr__(self):
        area = int(self.geometries.to_crs('EPSG:9834').area.sum())
        return f'Mask({len(self.geometries)} targets. Total Area: {area}m²)'
