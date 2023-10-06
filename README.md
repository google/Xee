# Xee: Xarray + Google Earth Engine

![Xee Logo](docs/xee-logo.png)

_An Xarray extension for Google Earth Engine._

## How to use

Install with pip (distributions on PyPi will come soon):

```shell
pip install git+https://github.com/google/xee.git
```

Then, authenticate Earth Engine:

```shell
earthengine authenticate --quiet
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
ic = ee.ImageCollection('ee://ECMWF/ERA5_LAND/HOURLY').filterDate('1992-10-05', '1993-03-31')
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

See [examples/](examples/) or [docs](docs/) for more uses and integrations.

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