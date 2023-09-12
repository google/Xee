
# Xee: Xarray + Google Earth Engine

![Xee Logo](g3docs/xee-logo.png)

_A Google Earth Engine backend for Xarray | An Xarray Client for Google Earth Engine._

## How to use

Install with pip (distributions on PyPi will come soon):

```shell
pip install git+https://github.com/googlestaging/xee.git
```

Then, authenticate Earth Engine: 

```shell
earthengine authenticate --quiet
```

Open any Earth Engine ImageCollection by specifying the Xarray engine as `'ee'`:

```python
ee.Initialize()
ds = xarray.open_dataset('ECMWF/ERA5_LAND/HOURLY', engine='ee')
```

Open all bands in a specific projection (not the Xee default):

```python
ee.Initialize()
ds = xarray.open_dataset('ECMWF/ERA5_LAND/HOURLY', engine='ee',
                         crs='EPSG:4326', scale=0.25)
```

Open an ImageCollection (maybe, with EE-side filtering or processing):

```python
ee.Initialize()
ic = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate('1992-10-05', '1993-03-31')
ds = xarray.open_dataset(ic, engine='ee', crs='EPSG:4326', scale=0.25)
```

Open multiple ImageCollections into one `xarray.Dataset`, all with the same projection:

```python
ee.Initialize()
ds = xarray.open_mfdataset(['ECMWF/ERA5_LAND/HOURLY', 'NASA/GDDP-CMIP6'],
                           engine='ee', crs='EPSG:4326', scale=0.25)
```

See [examples/](examples/) for more uses and integrations.

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