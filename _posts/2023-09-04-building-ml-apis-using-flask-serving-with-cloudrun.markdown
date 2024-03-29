---
layout: post
title:  "Building ML APIs using Flask and Serving with Cloud Run"
date:   2023-09-04 06:30:00 +0000
categories: gcp vertex-ai gcloud workbench bigquery bq bqml cloud-run flask web-apis
author: Ali Cabukel
---

*Content was generated by Bard*

### Introduction
Cloud Run is a serverless compute platform that allows you to run stateless containers that are invocable via HTTP requests. It is based on Knative, an open-source platform for building, deploying, and managing modern serverless workloads.

Flask is a popular Python microframework that makes it easy to create web applications. It is known for its simplicity, flexibility, and scalability.

In this blog post, we will show you how to deploy a Flask API to Cloud Run. We will also provide some tips on how to optimize your API for performance and scalability.

### Prerequisites
To follow along with this tutorial, you will need the following:

A Google Cloud Platform (GCP) account
The Cloud Run API enabled
The Cloud SDK installed and configured
A Python 3 environment
The Flask framework installed
Creating a Flask API
Let's start by creating a simple Flask API. Create a file called `app.py` and add the following code:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
return jsonify({'message': 'Hello, World!'})

if __name__ == '__main__':
app.run(debug=True)
```
This API simply returns a JSON response with the message "Hello, World!".

### Deploying the API to Cloud Run
Now that we have created our API, we can deploy it to Cloud Run. To do this, we will use the `gcloud` command-line tool.

First, we need to create a Cloud Run service. We can do this with the following command:

```
gcloud run services create my-api
```
This will create a new Cloud Run service called "my-api".

Next, we need to deploy our API code to Cloud Run. We can do this with the following command:

```
gcloud run deploy my-api --image gcr.io/PROJECT_ID/my-api
```
This will build a container image for our API and deploy it to Cloud Run.

Testing the API
Once our API is deployed, we can test it by making a request to the `/hello` endpoint. We can do this with the following command:

```
curl -X GET http://localhost:8080/hello
```
This should return a JSON response with the message "Hello, World!".

### Optimizing the API for Performance and Scalability
There are a few things we can do to optimize our API for performance and scalability.

First, we can use a CDN to cache static assets. This will reduce the load on our Cloud Run service and improve performance.

Second, we can use a load balancer to distribute traffic across multiple instances of our API. This will improve scalability and ensure that our API is always available.

Finally, we can use autoscaling to automatically scale our API up or down based on demand. This will help us save costs and ensure that our API is always available.

### Conclusion
In this blog post, we showed you how to deploy a Flask API to Cloud Run. We also provided some tips on how to optimize your API for performance and scalability.

Cloud Run is a powerful platform for running stateless containers. It is easy to use and can be used to deploy a wide variety of applications.

Flask is a popular Python microframework that is well-suited for creating web APIs. It is simple to use and can be used to create powerful and scalable APIs.

By combining Cloud Run and Flask, you can create and deploy powerful and scalable web APIs.

### Deploying ML APIs using Cloud Run

1. Pre-requisites
2. Creating a Workbench user-managed notebook
3. Getting the data
4. Creating the directory structure
5. Developing API methods `main.py`
6. Creating `Dockerfile`
7. OPTIONAL: Creating the dockerhub on Artifact Registry
8. Building docker image
9. Deploying API into Cloud Run
10. Call training API method
11. Call prediction API method

#### Pre-requisites

- Acccess to GCP with a proper billing account along with VertexAI, Cloud Storage, BigQuery, Cloud Run and Artifact Registry
- Service Accounts and Network settings must have been configured properly
- Google Cloud SDK (`gcloud` and `gsutil`) must have been installed and configured
- The following environment variables must have been set

```shell
BUCKET_NAME=<Bucket Name>
REGION=<Region>
ZONE=<Zone>
PROJECT_ID=<GCP Project ID>
SERVICE_ACCOUNT=<Service Account>
SUBNETWORK=<Subnetwork Name>
LOCATION=<BigQuery Dataset Location>
```

#### Creating a Workbench user-managed notebook

The following script will create a workbench user-managed notebook `telco-churn-tf-notebook` along with the following options:
- `n1-standard-4` machine type
- `tf2-ent-2-6-cu110-notebooks-v20211202-debian-10` image name from `deeplearning-platform-release` image project

```shell
gcloud notebooks instances create telco-churn-tf-notebook \
        --project $PROJECT_ID \
        --location $ZONE \
        --vm-image-project=deeplearning-platform-release \
        --vm-image-name=tf2-ent-2-6-cu110-notebooks-v20211202-debian-10 \
        --machine-type=n1-standard-4 \
        --service-account=$SERVICE_ACCOUNT \
        --subnet=$SUBNETWORK \
        --subnet-region=$REGION \
        --metadata=report-system-health=TRUE 
```

#### Getting the data

- The following command can be executed on workbench notebook in order to upload data into a bucket.

```shell
gsutil cp train.csv gs://$BUCKET_NAME/telco_churn/data/train.csv
gsutil cp valid.csv gs://$BUCKET_NAME/telco_churn/data/valid.csv
gsutil cp test.csv gs://$BUCKET_NAME/telco_churn/data/test.csv
```

- `bq` tool enables us to create a new dataset `telco_churn_db` on BigQuery

```shell
bq --location=$LOCATION --project_id=$PROJECT_ID mk -d telco_churn_db
```

- `schema.json` defines the BigQuery table's structure.

```json
[
  {
    "description": "MonthlyCharges",
    "mode": "NULLABLE",
    "name": "MonthlyCharges",
    "type": "FLOAT"
  },
  {
    "description": "TotalCharges",
    "mode": "NULLABLE",
    "name": "TotalCharges",
    "type": "FLOAT"
  },
  {
    "description": "tenure",
    "mode": "NULLABLE",
    "name": "tenure",
    "type": "INTEGER"
  },
  {
    "description": "gender",
    "mode": "NULLABLE",
    "name": "gender",
    "type": "STRING"
  },
  {
    "description": "SeniorCitizen",
    "mode": "NULLABLE",
    "name": "SeniorCitizen",
    "type": "STRING"
  },
  {
    "description": "Partner",
    "mode": "NULLABLE",
    "name": "Partner",
    "type": "STRING"
  },
  {
    "description": "Dependents",
    "mode": "NULLABLE",
    "name": "Dependents",
    "type": "STRING"
  },
  {
    "description": "PhoneService",
    "mode": "NULLABLE",
    "name": "PhoneService",
    "type": "STRING"
  },
  {
    "description": "MultipleLines",
    "mode": "NULLABLE",
    "name": "MultipleLines",
    "type": "STRING"
  },
  {
    "description": "InternetService",
    "mode": "NULLABLE",
    "name": "InternetService",
    "type": "STRING"
  },
  {
    "description": "OnlineSecurity",
    "mode": "NULLABLE",
    "name": "OnlineSecurity",
    "type": "STRING"
  },
  {
    "description": "OnlineBackup",
    "mode": "NULLABLE",
    "name": "OnlineBackup",
    "type": "STRING"
  },
  {
    "description": "DeviceProtection",
    "mode": "NULLABLE",
    "name": "DeviceProtection",
    "type": "STRING"
  },
  {
    "description": "TechSupport",
    "mode": "NULLABLE",
    "name": "TechSupport",
    "type": "STRING"
  },
  {
    "description": "StreamingTV",
    "mode": "NULLABLE",
    "name": "StreamingTV",
    "type": "STRING"
  },
  {
    "description": "StreamingMovies",
    "mode": "NULLABLE",
    "name": "StreamingMovies",
    "type": "STRING"
  },
  {
    "description": "Contract",
    "mode": "NULLABLE",
    "name": "Contract",
    "type": "STRING"
  },
  {
    "description": "PaperlessBilling",
    "mode": "NULLABLE",
    "name": "PaperlessBilling",
    "type": "STRING"
  },
  {
    "description": "PaymentMethod",
    "mode": "NULLABLE",
    "name": "PaymentMethod",
    "type": "STRING"
  },
  {
    "description": "Churn",
    "mode": "NULLABLE",
    "name": "Churn",
    "type": "INTEGER"
  }
]
```

- Creating `train` table under `telco_churn_db` dataset with the given `schema.json` schema and `train.csv` data source

```shell
bq --location=$LOCATION --project_id=$PROJECT_ID load \
    telco_churn_db.train \
    gs://$BUCKET_NAME/telco_churn/data/train.csv \
    schema.json
```

- Creating `valid` table under `telco_churn_db` dataset with the given `schema.json` schema and `valid.csv` data source

```shell
bq --location=$LOCATION --project_id=$PROJECT_ID load \
    telco_churn_db.valid \
    gs://$BUCKET_NAME/telco_churn/data/valid.csv \
    schema.json
```

- Creating `test` table under `telco_churn_db` dataset with the given `schema.json` schema and `test.csv` data source

```shell
bq --location=$LOCATION --project_id=$PROJECT_ID load \
    telco_churn_db.test \
    gs://$BUCKET_NAME/telco_churn/data/test.csv \
    schema.json
```

#### Creating the directory structure

```shell
mkdir -p telco_churn_run
```

#### Developing API methods `main.py`

- Create `main.py` under `telco_churn_run` directory

```python
import logging
import os
import json
from flask import Flask, request
import google.cloud.logging
from google.cloud import bigquery

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World"

@app.route("/train", methods=["POST"])
def train():
    if request.method == "POST":
        project_id = os.environ.get("PROJECT_ID")
        location = os.environ.get("LOCATION")
        dataset_name = os.environ.get("DATASET_NAME")
        log_client = google.cloud.logging.Client(project=project_id)
        log_client.setup_logging()
        bq_client = bigquery.Client(project=project_id, location=location)
        query = f"""
        CREATE OR REPLACE MODEL `{project_id}.{dataset_name}.telco_churn_xgb_baseline_v3` 
        OPTIONS( 
            model_type='BOOSTED_TREE_CLASSIFIER', 
            input_label_cols=['Churn'], 
            enable_global_explain=TRUE 
        ) AS 
        SELECT  t.* FROM `{project_id}.{dataset_name}.train` t 
        """
        query_job = bq_client.query(query)
        result = query_job.result()
        logging.info(f"BQ model has been created successfully")
        return {"message": "success"}


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        project_id = os.environ.get("PROJECT_ID")
        location = os.environ.get("LOCATION")
        dataset_name = os.environ.get("DATASET_NAME")
        log_client = google.cloud.logging.Client(project=project_id)
        log_client.setup_logging()
        bq_client = bigquery.Client(project=project_id, location=location)
        query = f"""
        SELECT  predicted_Churn,
                predicted_Churn_probs,
                Churn 
        FROM ML.PREDICT (
            MODEL `{project_id}.{dataset_name}.telco_churn_xgb_baseline_v3`, 
            TABLE `{project_id}.{dataset_name}.test`
        )
        """
        job_config = bigquery.QueryJobConfig( 
            destination=f'{project_id}.{dataset_name}.telco_churn_xgb_baseline_pred',
            write_disposition="WRITE_TRUNCATE"
        )
        query_job = bq_client.query(query, job_config=job_config)
        result = query_job.result()
        logging.info(f"BQ table has been executed successfully")
        return {"message": "success"}
```

#### Creating `Dockerfile`

- Create the `Dockerfile` under `telco_churn_run` directory

```shell
FROM python:3.10

RUN pip install Flask==2.3.2 
RUN pip install gunicorn==21.2.0 
RUN pip install google.cloud.bigquery==3.10.0 
RUN pip install google-cloud-bigquery-storage==2.19.1 
RUN pip install google-cloud-logging==3.5.0

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . .

CMD exec gunicorn --bind :8080 --workers 8 --threads 8 --timeout 120 --log-level=debug main:app
```

#### OPTIONAL: Creating the dockerhub on Artifact Registry

- Create `telco-churn-hub` on `Artifact Registry`:

```shell
pip install keyrings.google-artifactregistry-auth
gcloud config set artifacts/repository telco-churn-hub
gcloud config set artifacts/location europe-west1
gcloud auth application-default login
gcloud auth configure-docker europe-docker.pkg.dev
```

```shell
gcloud artifacts repositories create telco-churn-hub \
    --project=$PROJECT_ID \
    --repository-format=docker \
    --location=$LOCATION \
    --description="Telco-churn Docker repository"
```

#### Building docker image

```shell
cd telco_churn_run
gcloud builds submit \
  --tag europe-west1-docker.pkg.dev/$PROJECT_ID/telco-churn-hub/telco-churn-serving:latest .
```

#### Deploying API into Cloud Run

- Deploying API into Cloud Run

```shell
gcloud run deploy telco-churn-api \
    --image europe-west1-docker.pkg.dev/$PROJECT_ID/telco-churn-hub/telco-churn-serving:latest \
    --region $LOCATION \
    --port 8080 \
    --service-account $SERVICE_ACCOUNT \
    --no-allow-unauthenticated \
    --set-env-vars "PROJECT_ID=$PROJECT_ID" \
    --set-env-vars "LOCATION=$LOCATION" \
    --set-env-vars "DATASET_NAME=telco_churn_db" \
    --platform=managed \
    --memory 2Gi \
    --cpu 2 \
    --timeout=1800
```

- Getting API address (endpoint)

```shell
gcloud run services describe telco-churn-api --region $LOCATION --format='value(status.url)'
```

- Testing API method. *(Do not forget to replace API_ADDRESS)*

```shell
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" <API_ADDRESS>
```

#### Call training API method

- Testing training API method. *(Do not forget to replace API_ADDRESS)*

```shell
curl --request POST \
     --header "Authorization: Bearer $(gcloud auth print-identity-token)" \
     --header 'Content-Type: application/json'\
     <API_ADDRESS>
```

#### Call prediction API method

- Testing prediction API method. *(Do not forget to replace API_ADDRESS)*

```shell
curl --request POST \
     --header "Authorization: Bearer $(gcloud auth print-identity-token)" \
     --header 'Content-Type: application/json'\
     <API_ADDRESS>
```