# Xee Examples

Xee integrations & use cases.

## Export Earth Engine ImageCollections to Zarr with [Xarray-Beam](https://github.com/google/xarray-beam)

The following demonstrates how to export ~20 TiBs of NASA IMERG data to Zarr
(in about ~25 LOC) with Xarray-Beam and GCP's Dataflow Runner. At the time of
writing,  we were about to export this dataset in about ~3 hours (using Google's
internal Beam runner).

```shell
python3 ee_to_zarr.py \
  --input NASA/GPM_L3/IMERG_V06 \
  --output $BUCKET/imerg_006.zarr \
  --target_chunks "index=6" \
  --runner DataflowRunner \
  -- \
  --project $PROJECT \
  --region $REGION \
  --disk_size_gb 50 \
  --machine_type n2-highmem-2 \
  --no_use_public_ips  \
  --network $NETWORK \
  --subnetwork regions/$REGION/subnetworks/$SUBNET \
  --job_name imerg-to-zarr
```