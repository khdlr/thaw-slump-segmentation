from pathlib import Path
import json
from rasterio.features import rasterize
import math
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
import geopandas as gpd
import ee
import geedim as gd
from einops import rearrange
from joblib import Parallel, delayed

from .earthengine import get_ArcticDEM_rel_el, get_ArcticDEM_slope


SHAPEFILE_ROOT = Path('data/ML_training_labels/retrogressive_thaw_slumps/')
CACHEFOLDER = Path('data/S2Cache')
OUTFOLDER = Path('data/S2Scenes')
TILESIZE = 192

BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

lookback  = pd.Timedelta(days=5)
lookahead = pd.Timedelta(days=5)

def burn_mask(gdf, master):
  mask = rasterize(gdf.geometry.to_crs(master.rio.crs),
      out_shape=master.shape[-2:], transform=master.rio.transform())
  mask = rearrange(mask, 'H W -> 1 H W')
  mask = xr.DataArray(mask,
      coords={
        'band': [1],
        'y': master.coords['y'],
        'x': master.coords['x'],
      },
      attrs={
        'long_name': 'mask',
      }
  )
  mask.coords['spatial_ref'] = master.coords['spatial_ref']

  return mask


def tile_data(datasets):
  tiles = {d: [] for d in datasets}
  H, W = list(datasets.values())[0].shape[1:]
  y_vals = np.linspace(0, H-TILESIZE, int(math.ceil(H / TILESIZE)))
  x_vals = np.linspace(0, W-TILESIZE, int(math.ceil(W / TILESIZE)))

  for y0 in y_vals:
    for x0 in x_vals:
      y0, x0 = int(y0), int(x0)
      for d in datasets:
        tiles[d].append(datasets[d][:, y0:y0+TILESIZE, x0:x0+TILESIZE])

  stacks = {}
  for d in tiles:
    stacks[d] = np.stack(tiles[d])
  return stacks


def extract_and_rename_bands(dataarray, bands, band_name):
  indices = [dataarray.long_name.index(b) for b in bands]
  subset = dataarray.isel(band=indices)
  subset.attrs['long_name'] = tuple(dataarray.long_name[i] for i in indices)
  return subset.rename(band=band_name)


def get_s2_scene(data):
  image_id = data['image_id']
  image_date = data['image_date']
  geometry = data['geometry']

  gd.Initialize()
  s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").select(BANDS)
  s2 = gd.MaskedCollection(s2)

  minx, miny, maxx, maxy = geometry.bounds
  centroid_x = 0.5 * (minx + maxx)
  UTMZone = int(math.ceil((180 + centroid_x) / 6))
  CRS = f'EPSG:326{UTMZone:02d}'
  geom = gpd.GeoSeries(geometry, crs='EPSG:4326').to_crs(CRS)

  # Define ROI as a ee Geometry
  bounds = ee.Geometry.Polygon(list(geom[0].exterior.coords), proj=CRS, evenOdd=False)

  imgs = s2.search(
    start_date=image_date - lookback,  # Select starting `lookback` days before PSOrthoTile
    end_date=image_date + lookahead,   # Select until `lookahead` days after PSOrthoTile
    region=bounds,
    fill_portion=90,             # Ensure at least 90% of pixels are valid
    custom_filter='CLOUDY_PIXEL_PERCENTAGE < 20'  # Filter Scenes with too many clouds
  )

  out_dir = CACHEFOLDER / image_id
  out_dir.mkdir(parents=True, exist_ok=True)
  print(f'Found {len(imgs.properties)} scenes for {image_id}')

  def get_aux_data(name, crs, ee_image):
    img = gd.MaskedImage(ee_image)
    out_path = out_dir / f'{name}_epsg{crs.to_epsg()}.tif'
    if not out_path.exists():
      img.download(out_path,
          region=bounds,
          crs=str(crs),
          scale=10,
      )

    data = rioxarray.open_rasterio(out_path)
    return data

  for img in imgs.properties:
    s2_id = img.split('/')[-1]
    s2path = out_dir / f'{s2_id}.tif'
    full_scene_path = OUTFOLDER / f'{image_id}_{s2_id}.nc'

    if not s2path.exists():
      # By default, geedim downloads scenes at native resolution and CRS
      gd.MaskedImage.from_id(img).download(s2path, region=bounds)

      # Save S2 image metadata, we might need that later
      with s2path.with_suffix('.json').open('w') as f:
        json.dump(imgs.properties[img], f)

    # TODO: Why does casting to uint16 destroy the geo-referencing?
    s2 = rioxarray.open_rasterio(s2path)# .astype(np.uint16)
    s2 = extract_and_rename_bands(s2, ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'], 's2_band')
    TCVIS = ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic()

    tcvis = get_aux_data('TCVIS', s2.rio.crs, TCVIS)
    tcvis = extract_and_rename_bands(tcvis, ['TCB_slope', 'TCG_slope', 'TCW_slope'], 'tcvis_band')

    relative_elevation = get_aux_data('RelativeElevation', s2.rio.crs, get_ArcticDEM_rel_el())
    relative_elevation = extract_and_rename_bands(relative_elevation, ['elevation'], 'relative_elevation_band')

    slope = get_aux_data('Slope', s2.rio.crs, get_ArcticDEM_slope())
    slope = extract_and_rename_bands(slope, ['slope'], 'slope_band')

    mask = burn_mask(labels[labels.image_id == image_id], s2).rename(band='mask_band')
    valid_label = burn_mask(geom, s2).rename(band='valid_label_band')

    dataset = xr.Dataset({
      'Sentinel2': s2,
      'TCVIS': tcvis,
      'RelativeElevation': relative_elevation,
      'Slope': slope,
      'Mask': mask,
      'ValidLabel': valid_label,
    })

    # dataset = dataset.chunk({'y': 192, 'x': 192})
    full_scene_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(full_scene_path, format='NETCDF4', engine='h5netcdf')


if __name__ == '__main__':
  # Build JoinTable to get date from image_id
  labels = map(gpd.read_file, SHAPEFILE_ROOT.glob('*/TrainingLabel*.shp'))
  labels = pd.concat(labels).reset_index(drop=True)
  labels['image_date'] = pd.to_datetime(labels['image_date'])
  id2date = dict(labels[['image_id', 'image_date']].values)

  scenes = map(gpd.read_file, SHAPEFILE_ROOT.glob('*/ImageFootprints*.shp'))
  scenes = pd.concat(scenes).reset_index(drop=True)
  scenes = scenes[scenes.image_id.isin(labels.image_id)]
  scenes['image_date'] = scenes['image_id'].apply(id2date.get)

  Parallel(n_jobs=4)(delayed(get_s2_scene)(row) for _, row in scenes.iterrows())
