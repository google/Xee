# Xee Examples

Xee integrations & use cases.

## Export Earth Engine ImageCollections to Zarr with [Xarray-Beam](https://github.com/google/xarray-beam)

The following demonstrates how to export ~20 TiBs of NASA IMERG data to Zarr
(in about ~25 LOC) with Xarray-Beam and GCP's Dataflow Runner. At the time of
writing, we were able to export the data in about ~3 hours (using Google's
internal Beam runner).

```shell
python3 ee_to_zarr.py \
  --input NASA/GPM_L3/IMERG_V06 \
  --output $BUCKET/imerg_006.zarr \
  --target_chunks "time=6" \
  --runner DataflowRunner \
  -- \
  --project $PROJECT \
  --region $REGION \
  --temp_location $BUCKET/tmp/ \
  --no_use_public_ips  \
  --network $NETWORK \
  --subnetwork regions/$REGION/subnetworks/$SUBNET \
  --requirements ee_to_zarr_reqs.txt \
  --job_name imerg-to-zarr
```