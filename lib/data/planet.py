from typing import Union
from pathlib import Path
import xarray as xr
import rasterio as rio
import rioxarray
from pyproj import Proj

from .base import TileSource, Scene, cache_path


class PlanetScope(TileSource):
    def __init__(self, tile_path: Union[str, Path]):
        self.tile_path = Path(tile_path)

    def get_raster_data(self, scene: Scene) -> xr.DataArray:
        data = rioxarray.open_rasterio(self.tile_path)

        # Usually, PlanetScope will be the master imagery.
        # But if not, we'll need to reproject it to match whatever master we're using.
        same_size = scene.size == data.shape[-2:]
        same_crs  = Proj(data.rio.crs) == Proj(scene.crs)
        same_transform = data.rio.transform() == scene.transform
        if not (same_size and same_crs and same_transform):
            print('Reprojecting PlanetScope data. This has never been tested, so please double-check!')
            data = data.reproject(
                dst_crs=scene.crs,
                shape=scene.size,
                transform=scene.transform,
                resampling=rio.enums.Resampling.average,
            )

        # We could transfer the band names as coordinate labels here
        # But other tools don't seem to be compatible with that (i.e. QGIS)
        # data = data.assign_coords({'band': list(data.long_name[:13])})
        data = data.rename(band='PlanetScope_band')
        return data

    @staticmethod
    def build_scene(tile_path):
        tile_path = Path(tile_path)
        ds = rioxarray.open_rasterio(tile_path, decode_coords='all')
        scene = Scene(
            id=tile_path.stem,
            crs=ds.rio.crs,
            transform=ds.rio.transform(),
            size=ds.shape[-2:],
            layers=[PlanetScope(tile_path)])
        return scene

    def __repr__(self):
        return f'PlanetScope({self.tile_path.stem})'
