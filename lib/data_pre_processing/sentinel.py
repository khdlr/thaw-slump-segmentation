from pathlib import Path
import json
import rasterio as rio
from rasterio.features import rasterize
import math
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import ee
import geedim as gd
from einops import rearrange
from joblib import Parallel, delayed

from .earthengine import get_ArcticDEM_rel_el, get_ArcticDEM_slope


SHAPEFILE_ROOT = Path('data/ML_training_labels/retrogressive_thaw_slumps/')
OUTFOLDER = Path('data/Sentinel2') 
TILESIZE = 192

BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

lookback  = pd.Timedelta(days=5)
lookahead = pd.Timedelta(days=5)

def burn_mask(gdf, shape, profile):
  mask = rasterize(gdf.geometry.to_crs(profile['crs']),
        out_shape=shape, transform=profile['transform'])
  return rearrange(mask, 'H W -> 1 H W')

def tile_data(datasets, master='Sentinel2'):
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
  geom = gpd.GeoSeries(geometry, crs='EPSG:4326').to_crs(CRS)[0]

  # Define ROI as a ee Geometry
  bounds = ee.Geometry.Polygon(list(geom.exterior.coords), proj=CRS, evenOdd=False)

  imgs = s2.search(
    start_date=image_date - lookback,  # Select starting `lookback` days before PSOrthoTile
    end_date=image_date + lookahead,   # Select until `lookahead` days after PSOrthoTile
    region=bounds,
    fill_portion=90,             # Ensure at least 90% of pixels are valid
    custom_filter='CLOUDY_PIXEL_PERCENTAGE < 20'  # Filter Scenes with too many clouds
  )

  out_dir = Path(f'data/Sentinel2/{image_id}')
  out_dir.mkdir(parents=True, exist_ok=True)
  tiles_path = Path(f'data/S2Tiles/{image_id}.nc')
  tiles_path.parent.mkdir(parents=True, exist_ok=True)
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

    with rio.open(out_path) as raster:
      data = raster.read()
      if raster.descriptions[-1] == 'FILL_MASK':
        data = data[:-1]
    return data

  stacks = []
  for img in imgs.properties:
    s2_id = img.split('/')[-1]
    outpath = out_dir / f'{s2_id}.tif'

    if not outpath.exists():
      # By default, geedim downloads scenes at native resolution and CRS
      gd.MaskedImage.from_id(img).download(outpath, region=bounds)

      # Save S2 image metadata, we might need that later
      with outpath.with_suffix('.json').open('w') as f:
        json.dump(imgs.properties[img], f)

    with rio.open(outpath) as s2raster:
      profile = s2raster.profile
      datasets = {
        'Sentinel2': s2raster.read()[:13].astype(np.uint16),
      }
    with rio.open(outpath) as raster:
      crs = raster.crs

    datasets['TCVIS'] = get_aux_data('TCVIS',
        crs, ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic())
    datasets['RelativeElevation'] = get_aux_data('RelativeElevation', crs, get_ArcticDEM_rel_el())
    datasets['Slope'] = get_aux_data('Slope', crs, get_ArcticDEM_slope())
    datasets['Mask'] = burn_mask(labels[labels.image_id == image_id], s2raster.shape, profile)

    stacks.append(tile_data(datasets))

  data = {}
  for stack in stacks:
    for var in stack:
      if var in data:
        data[var] = np.concatenate([data[var], stack[var]], axis=0)
      else:
        data[var] = stack[var]

  dataset = xr.Dataset({d: xr.DataArray(data[d], dims=['sample', f'{d}_band', 'y', 'x']) for d in data})
  dataset.to_netcdf(tiles_path, engine='h5netcdf')

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

  # get_s2_scene(scenes.iloc[0])
  Parallel(n_jobs=4)(delayed(get_s2_scene)(row) for _, row in scenes.iterrows())
