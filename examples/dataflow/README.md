# Xee Dataflow Example

This example illustrates how to run an Xee Beam process using Dataflow on Google Cloud Platform.

The example requires a Google Cloud account and will incur charges!

## Cloud setup

To begin, there is a fair amount of setup of Cloud resources to execute the workflow on a Cloud Project.

This example assumes you have the [Google Cloud SDK installed](https://cloud.google.com/sdk/docs/install) and an [Earth Engine project setup with your Cloud Project](https://developers.google.com/earth-engine/cloud/earthengine_cloud_project_setup).


### Set environment variables

These environment variables are used throughout the example to make life easier when working across different Cloud environments. These get Cloud project info as well as set naming information for infrastructure setup in example.

```shell
PROJECT=$(gcloud config get-value project)

REGION=us-central1

REPO=xee-dataflow
CONTAINER=beam-runner

SA_NAME=xee-dataflow-controller
SERVICE_ACCOUNT=${SA_NAME}@${PROJECT}.iam.gserviceaccount.com
```

### Create custom Docker Container with dependencies

One of the suggested ways to handle external dependencies within a Beam pipeline is to [use a custom Docker Container](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/#custom-containers) with the pipeline. This is useful because each remote worker will need to install dependencies when it spins up and having a pre-built container makes that much quicker.

To do this with Google Cloud, one must first create an [Artifact Registry](https://cloud.google.com/artifact-registry/docs/overview) repository where the Docker Container can be stored and then build/push the container to the registry repository.

To create an Artifact Registry repository run the following command:

```shell
gcloud artifacts repositories create $REPO \
  --location=$REGION \
  --repository-format=docker \
  --description="Repository for hosting the Docker images to test xee with Dataflow" \
  --async
```

The next step is to build the Docker Container and push to the repository just created. This is done using [Cloud Build](https://cloud.google.com/build/docs/overview) with a configuration file. The config file defines how the image is built and where it is stored.

The `cloudbuild.yaml` file has general variables that need to be replaced with information with your Cloud environment. Open the file in your favorite text editor and replace "REGION" with the Cloud Region you selected, "YOUR_PROJECT" with your Cloud Project ID, "REPO" with the Artifact Registry repository name, and "CONTAINER" with the container name.

Alternatively, you can replace them with the following command:

```shell
sed -i 's/REGION/'"$REGION"'/g; s/YOUR_PROJECT/'"$PROJECT"'/g; s/REPO/'"$REPO"'/g; s/CONTAINER/'"$CONTAINER"'/g' cloudbuild.yaml
```

Run the following command to build the container to use with Dataflow:

```shell
gcloud builds submit --config cloudbuild.yaml
```

### Create custom Docker Container with dependencies

This example will output data to a Cloud Storage bucket so one needs to be created
for the pipeline. To do so run the following command:

```shell
gsutil mb -l $REGION gs://xee-out-${PROJECT}
```

Cloud bucket names need to be globally unique so this uses the Cloud Project Number (also globally unique) in the name.

### Create a Service Account

Service Accounts (SA) are used for authorization of remote workers to make calls to different services. It is good practice to create a SA for a specific process and this is to limit the roles assigned to one individual SA required for the process.

To create a SA run the following code:

```shell
gcloud iam service-accounts create ${SA_NAME} \
  --description="Controller service account for services used with Dataflow" \
  --display-name="Xee Dataflow Controller SA"
```

next assign the required roles to the Service Account to properly manage workers and read/write data.

```shell
roles=("roles/earthengine.writer" "roles/serviceusage.serviceUsageConsumer" "roles/storage.objectAdmin" "roles/artifactregistry.reader" "roles/dataflow.worker")

for role in ${roles[@]}
do
   gcloud projects add-iam-policy-binding ${PROJECT} \
    --member=serviceAccount:${SERVICE_ACCOUNT} \
    --role=${role}
done
```

Now that all of the Cloud infrastructure is setup, it is time to run the pipeline!

## Run the pipeline

This example is focused on pulling data from Earth Engine, transforming the data into Zarr formats and storing the results. There is the script `ee_to_zarr_dataflow.py` script that defines the pipeline and passing command line arguments define how it is executed with Dataflow.

```shell
python ee_to_zarr_dataflow.py \
  --input NASA/GPM_L3/IMERG_V06 \
  --output gs://xee-out-${PROJECT} \
  --target_chunks='time=6' \
  --runner DataflowRunner \
  --project $PROJECT \
  --region $REGION \
  --temp_location gs://xee-out-${PROJECT}/tmp/ \
  --service_account_email $SERVICE_ACCOUNT \
  --sdk_location=container \
  --sdk_container_image=${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${CONTAINER} \
  --job_name imerg-dataflow-$(date '+%Y%m%d%H%M%S')
```
