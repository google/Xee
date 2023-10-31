# Xee: Xarray + Google Earth Engine

![Xee Logo](https://raw.githubusercontent.com/google/Xee/main/docs/xee-logo.png)

_An Xarray extension for Google Earth Engine._

[![image](https://img.shields.io/pypi/v/xee.svg)](https://pypi.python.org/pypi/xee)
[![image](https://static.pepy.tech/badge/xee)](https://pepy.tech/project/xee)
[![Conda Recipe](https://img.shields.io/badge/recipe-xee-green.svg)](https://github.com/conda-forge/xee-feedstock)
[![image](https://img.shields.io/conda/vn/conda-forge/xee.svg)](https://anaconda.org/conda-forge/xee)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/xee.svg)](https://anaconda.org/conda-forge/xee)

## How to use

Install with pip:

```shell
pip install --upgrade xee
```

Install with conda:

```shell
conda install -c conda-forge xee
```

Then, authenticate Earth Engine:

```shell
earthengine authenticate --quiet
```

Now, in your Python environment, make the following imports:

```python
import ee
import xarray
```

Next, initialize the EE client with the high volume API:

```python
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
```

Open any Earth Engine ImageCollection by specifying the Xarray engine as `'ee'`:

```python
ds = xarray.open_dataset('ee://ECMWF/ERA5_LAND/HOURLY', engine='ee')
```

Open all bands in a specific projection (not the Xee default):

```python
ds = xarray.open_dataset('ee://ECMWF/ERA5_LAND/HOURLY', engine='ee',
                         crs='EPSG:4326', scale=0.25)
```

Open an ImageCollection (maybe, with EE-side filtering or processing):

```python
ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate('1992-10-05', '1993-03-31')
ds = xarray.open_dataset(ic, engine='ee', crs='EPSG:4326', scale=0.25)
```

Open an ImageCollection with a specific EE projection or geometry:

```python
ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate('1992-10-05', '1993-03-31')
leg1 = ee.Geometry.Rectangle(113.33, -43.63, 153.56, -10.66)
ds = xarray.open_dataset(
    ic,
    engine='ee',
    projection=ic.first().select(0).projection(),
    geometry=leg1
)
```

Open multiple ImageCollections into one `xarray.Dataset`, all with the same projection:

```python
ds = xarray.open_mfdataset(['ee://ECMWF/ERA5_LAND/HOURLY', 'ee://NASA/GDDP-CMIP6'],
                           engine='ee', crs='EPSG:4326', scale=0.25)
```

Open a single Image by passing it to an ImageCollection:

```python
i = ee.ImageCollection(ee.Image("LANDSAT/LC08/C02/T1_TOA/LC08_044034_20140318"))
ds = xarray.open_dataset(i, engine='ee')
```

See [examples](examples/) or [docs](docs/) for more uses and integrations.

## License

This is not an official Google product.

```
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
