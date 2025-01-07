# Installation

Install Xee and its dependencies using `pip` or conda-like package managers. To
help minimize system disruption and package conflicts, it's recommended to use
virtual environments like Python's
[`venv`](https://docs.python.org/3/library/venv.html) with `pip` or [conda's
integrated environment management
system](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

Install with `pip`:

```shell
pip install --upgrade xee
```

Install with conda:

```shell
conda install -c conda-forge xee
```

## Earth Engine setup

Xee makes requests to [Google Earth
Engine](https://developers.google.com/earth-engine/guides) for data. To use
Earth Engine, you'll need to create and register a Google Cloud project,
authenticate with Google, and initialize the service.

If you already have a Cloud project registered for Earth Engine and are familiar
with Earth Engine authentication and initialization, you can skip this section.

**Note**: the authentication and initialization steps described in the following
sections cover the majority of common system configurations and access methods,
if you're having trouble, refer to the Earth Engine [Authentication and
Initialization guide](https://developers.google.com/earth-engine/guides/auth).

### Create and register a Cloud project

Follow instructions in the [Earth Engine Access
guide](https://developers.google.com/earth-engine/guides/access#get_access_to_earth_engine
) to create and register a Google Cloud project.

### Authentication

Google needs to know who is accessing Earth Engine to determine what services
are available and what permissions are granted. The goal of authentication is to
establish credentials that can be used during initialization. There are several
ways to verify your identity and create credentials, depending on your working
environment:

#### Persistent environment

If you're working from a system with a persistent environment, such as a local
computer or on-premises server, you can authenticate using the [Earth Engine
command line
utility](https://developers.google.com/earth-engine/guides/command_line#authenticate):

```shell
earthengine authenticate
```

This command opens a browser window for authentication. Once authenticated, the
credentials are stored locally (`~/.config/earthengine/credentials`), allowing
them to be used in subsequent initialization to the Earth Engine service. This
is typically a one-time step.

#### Temporary environment

If you're working from a system like [Google Colab](https://colab.google/) that
provides a temporary environment recycled after use, you'll need to authenticate
every session. In this case, you can use the `earthengine-api` library
(installed with Xee) to authenticate interactively:

```python
ee.Authenticate()
```

This method selects the most appropriate [authentication
mode](https://developers.google.com/earth-engine/guides/auth#authentication_details)
and guides you through steps to generate authentication credentials. Be sure to
rerun the authentication process each time the environment is reset.

### Initialization

Initialization checks user authentication credentials, sets the Cloud project to
use for requests, and connects the client to Earth Engine's services. At the
top of your script, include one of the following expressions with the `project`
argument modified to match the Google Cloud project ID enabled and registered
for Earth Engine use.

#### High-volume endpoint

If you are requesting stored data (supplying a collection ID or passing an
unmodified `ee.ImageCollection()` object to `xarray.open_dataset`), connect to
the [high-volume
endpoint](https://developers.google.com/earth-engine/guides/processing_environments#high-volume_endpoint).

```python
ee.Initialize(
    project='your-project-id',
    opt_url='https://earthengine-highvolume.googleapis.com'
)
```

#### Standard endpoint

If you are requesting computed data (applying expressions to the data), consider
connecting to the [standard
endpoint](https://developers.google.com/earth-engine/guides/processing_environments#standard_endpoint).
It utilizes caching, so it can be more efficient if you need to rerun or adjust
something about the request.

```python
 ee.Initialize(project='your-project-id')
```
