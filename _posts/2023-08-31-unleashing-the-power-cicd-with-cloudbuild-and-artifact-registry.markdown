---
layout: post
title:  "Unleashing the Power of CI/CD with Cloud Build and Artifact Registry"
date:   2023-08-30 06:30:00 +0000
categories: gcp vertex-ai gcloud workbench bigquery bq bqml kfp pipeline component compile trigger payload jinja2 pathlib kubernetes containers docker artifact-registry cloud-build mlops ci/cd github
author: Ali Cabukel
---

*Content was generated by Bard*

### Introduction:
Continuous Integration and Continuous Delivery (CI/CD) are essential practices in modern software development. They enable teams to automate the building, testing, and deployment of code changes, resulting in faster and more reliable software delivery. In this blog post, we will explore how Cloud Build and Artifact Registry, two powerful tools offered by Google Cloud Platform (GCP), can help you implement a robust CI/CD pipeline.

### Cloud Build:
Cloud Build is a fully managed build service that allows you to automate the building and testing of your code. It supports a wide range of programming languages and frameworks, including Java, Python, Node.js, and Go. With Cloud Build, you can easily create and manage build triggers, such as code changes or scheduled builds. It also provides built-in support for popular tools like Maven, Gradle, and npm.

### Artifact Registry:
Artifact Registry is a managed repository service that allows you to store and manage your build artifacts, such as compiled code, packages, and container images. It provides a secure and scalable way to share artifacts across different projects and environments. Artifact Registry integrates seamlessly with Cloud Build, enabling you to publish and consume artifacts as part of your CI/CD pipeline.

### Setting Up CI/CD with Cloud Build and Artifact Registry:
To set up CI/CD with Cloud Build and Artifact Registry, follow these steps:

### Create a Cloud Build trigger: 
Go to the Cloud Build console and click on "Create Trigger". Select the source code repository (e.g., GitHub or Cloud Source Repositories) and specify the branch or tag that should trigger the build.

### Configure the build steps: 
Define the build steps that need to be executed, such as compiling code, running tests, and generating artifacts. You can use the built-in steps provided by Cloud Build or create custom steps using Dockerfiles or other scripts.

### Specify the artifact location: 
In the build configuration, specify the Artifact Registry repository where the build artifacts should be published.

### Enable the trigger: 
Once you have configured the build steps and artifact location, enable the trigger to start the build process.

### Building and Testing:
When a code change is detected, Cloud Build will trigger the build process based on the configured trigger. It will then execute the build steps, such as compiling the code and running unit tests. If the build is successful, Cloud Build will publish the build artifacts to the specified Artifact Registry repository.

### Deploying to Production:
Once the build artifacts are available in Artifact Registry, you can deploy them to production using various methods. For example, you can use Cloud Deploy to deploy container images to Kubernetes clusters or use Cloud Run to deploy serverless applications.

### Benefits of Using Cloud Build and Artifact Registry:
Using Cloud Build and Artifact Registry for CI/CD offers several benefits:

### Increased Efficiency: 
Automating the build, test, and deployment processes with Cloud Build and Artifact Registry significantly improves efficiency and reduces manual effort.

### Improved Reliability: 
By automating the CI/CD pipeline, you can ensure that code changes are built, tested, and deployed consistently and reliably.

### Scalability: 
Cloud Build and Artifact Registry are highly scalable services that can handle large-scale builds and manage a vast number of artifacts.

### Security: 
Both Cloud Build and Artifact Registry provide robust security features to protect your code and artifacts from unauthorized access.

### Cost-Effectiveness: 
Cloud Build and Artifact Registry are cost-effective solutions that eliminate the need for maintaining and managing your own build and artifact management infrastructure.

### Conclusion:
Cloud Build and Artifact Registry are powerful tools that can help you implement a robust and efficient CI/CD pipeline. By automating the build, test, and deployment processes, you can accelerate software delivery, improve reliability, and focus on delivering value to your customers. Embrace the power of CI/CD with Cloud Build and Artifact Registry, and unlock the full potential of your software development process.

### Deploying ML Pipelines using KFP through Cloud Build

1. Pre-requisites
2. Creating a Workbench user-managed notebook
3. Getting the data
4. Creating a repo on Github and cloning into local
5. Creating the directory structure on a new feature branch
6. Creating the service account json file for Google Credentials
7. Developing the components
8. Creating the create model query template
9. Creating the evaluate model query template
10. Creating the training payload
11. Developing the training pipeline
12. Creating the prediction query templates
13. Creating the prediction payload
14. Developing the prediction pipeline
15. Developing the trigger
16. Creating docker file to create a docker image
17. Creating cloud build yaml files
18. Seting-up Github Triggers
19. OPTIONAL: Creating the dockerhub on Artifact Registry
20. Triggering Cloud Build with Github Events

#### Pre-requisites

- Acccess to GCP with a proper billing account along with VertexAI, Cloud Storage, BigQuery, Cloud Build and Artifact Registry services
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

#### Creating a repo on Github and cloning into local

- Create a new repo on an enterprise or public github namely telco_churn_kfp
- Configure `Collaborators and teams` under the `Settings` section
- Configure `Branches` under the `Settings` section by providing a `Branch protection rules`
    + Branch name pattern: `main`, check `Require a pull-request before merging`, uncheck `Require approvals` and hit `Create` button
- Create a `Personal access token` by navigating `Settings` > `Developer settings` > `Personal access tokens`
- Clone the repo using the following command. *(Do not forget to replace GITHUB_USER, GITHUB_TOKEN and REPO_URL)*

```shell
git clone https://<GITHUB_USER>:<GITHUB_TOKEN>@<REPO_URL>/telco_churn_kfp.git
cd telco_churn_kfp
```

- Create a new branch namely `feature-develop` and push to origin

```shell
git checkout -b feature-develop
git push -u origin feature-develop
```

#### Creating the directory structure on a new feature branch

```shell
mkdir -p telco_churn_kfp/payloads
mkdir -p telco_churn_kfp/queries
touch telco_churn_kfp/__init__.py
```

#### Creating the service account json file for Google Credentials

- Create `sa-key.json` under `telco_churn_kfp` directory

#### Developing the components

- Create the `components.py` under `telco_churn_kfp/telco_churn_kfp` directory. *(Do not forget to replace the variables (PROJECT_ID, LOCATION, SERVICE_ACCOUNT, BUCKET_NAME)*

```python
from kfp.dsl import component

PYTHON = "python:3.10"
BIGQUERY = "google-cloud-bigquery==3.12.0"

PROJECT_ID = "<PROJECT_ID>"
LOCATION = "<LOCATION>"
SERVICE_ACCOUNT = "<SERVICE_ACCOUNT>"
PIPELINE_ROOT = "<BUCKET_NAME>"


@component(
    base_image=PYTHON,
    packages_to_install=[BIGQUERY]
)
def bq_run_query(
    project_id: str,
    location: str,
    query: str
) -> None:
    from google.cloud import bigquery
    import logging
    logging.getLogger().setLevel(logging.INFO)
    bq_client = bigquery.client.Client(project=project_id, location=location)
    query_job = bq_client.query(query)
    result = query_job.result()
    logging.info(f"BQ query has been executed successfully")
    
@component(
    base_image=PYTHON,
    packages_to_install=[BIGQUERY]
)
def bq_create_table(
    project_id: str,
    location: str,
    query: str,
    dataset_id: str,
    table_id: str
) -> None:
    from google.cloud import bigquery
    import logging
    logging.getLogger().setLevel(logging.INFO)
    bq_client = bigquery.client.Client(project=project_id, location=location)
    job_config = bigquery.QueryJobConfig( 
        destination=f'{project_id}.{dataset_id}.{table_id}',
        write_disposition="WRITE_TRUNCATE"
    )
    query_job = bq_client.query(query, job_config=job_config)
    result = query_job.result()
    logging.info(f"BQ query has been executed successfully")
```

#### Creating the create model query template

- Create the `create_model.sql` under `telco_churn_kfp/telco_churn_kfp/queries` directory

```sql
{% raw %}
CREATE OR REPLACE MODEL `{{ project_id }}.{{ dataset_id }}.{{ model_id }}` 
OPTIONS( 
    model_type='BOOSTED_TREE_CLASSIFIER', 
    input_label_cols=['{{ label }}'], 
    enable_global_explain=TRUE 
) AS 
SELECT  t.* FROM `{{ project_id }}.{{ dataset_id }}.{{ table_id }}` t 
{% endraw %}
```

#### Creating the evaluate model query template

- Create the `evaluate_model.sql` under `telco_churn_kfp/telco_churn_kfp/queries` directory

```sql
{% raw %}
SELECT * 
FROM ML.EVALUATE(MODEL `{{ project_id }}.{{ dataset_id }}.{{ model_id }}`, 
                TABLE `{{ project_id }}.{{ dataset_id }}.{{ table_id }}`
                )
{% endraw %}
```

#### Creating the training payload

- Create the `training.json` under `telco_churn_kfp/telco_churn_kfp/payloads` directory. *(Do not forget to replace the variables (PROJECT_ID, LOCATION)*

```json
{
    "attributes": {
        "display_name": "telco-churn-training-pipeline",
        "template_path": "training.json"
    },
    "data": {
        "project_id": "<PROJECT_ID>",
        "dataset_id": "telco_churn_db",
        "location": "<LOCATION>",
        "model_id": "telco_churn_xgb_baseline_v1",
        "train_table_id": "train",
        "valid_table_id": "valid",
        "label_name": "Churn"
    }
}
```

#### Developing the training pipeline

- Create the `training.py` under `telco_churn_kfp/telco_churn_kfp` directory

```python
import json
import pathlib
from jinja2 import Template

from kfp import compiler, dsl
from telco_churn_kfp.components import (
    bq_run_query,
    bq_create_table
)


@dsl.pipeline(name="training-pipeline")
def training_pipeline(
    project_id: str,
    dataset_id: str,
    location: str,
    model_id: str,
    train_table_id: str,
    valid_table_id: str,
    label_name: str
):
    model_query_template_file = pathlib.Path(__file__).parent / "queries" / "create_model.sql"
    with open(model_query_template_file) as f:
        model_query_template = f.read()
    model_query = (
        Template(model_query_template)
        .render(
            project_id=project_id, 
            dataset_id=dataset_id, 
            table_id=train_table_id,
            model_id=model_id,
            label=label_name
        )
    )
    create_model = (
        bq_run_query(
            project_id=project_id,
            location=location,
            query=model_query
        )
        .set_display_name('BQ Create XGB model')
    )
    
    eval_query_template_file = pathlib.Path(__file__).parent / "queries" / "evaluate_model.sql"
    with open(eval_query_template_file) as f:
        eval_query_template = f.read()
    eval_query = (
        Template(eval_query_template)
        .render(
            project_id=project_id, 
            dataset_id=dataset_id, 
            table_id=valid_table_id,
            model_id=model_id
        )
    )
    eval_table = (
        bq_create_table(
            project_id=project_id,
            location=location,
            query=eval_query,
            dataset_id=dataset_id,
            table_id='telco_churn_xgb_baseline_eval'
        )
        .set_display_name('BQ Evaluate XGB model against validation data')
        .after(create_model)
    )
    

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="training.json",
        type_check=False,
    )
```

#### Creating the prediction query template

- Create the `predict_model.sql` under `telco_churn_kfp/telco_churn_kfp/queries` directory

```sql
{% raw %}
SELECT  predicted_{{ label }},
        predicted_{{ label }}_probs,
        {{ label }}
FROM ML.PREDICT (
    MODEL `{{ project_id }}.{{ dataset_id }}.{{ model_id }}`, 
    TABLE `{{ project_id }}.{{ dataset_id }}.{{ table_id }}`
)
{% endraw %}
```

#### Creating the prediction payload

- Create the `prediction.json` under `telco_churn_kfp/telco_churn_kfp/payloads` directory. *(Do not forget to replace the variables (PROJECT_ID, LOCATION)*

```json
{
    "attributes": {
        "display_name": "telco-churn-prediction-pipeline",
        "template_path": "prediction.json"
    },
    "data": {
        "project_id": "<PROJECT_ID>",
        "dataset_id": "telco_churn_db",
        "location": "<LOCATION>",
        "model_id": "telco_churn_xgb_baseline_v1",
        "score_table_id": "train",
        "label_name": "Churn"
    }
}
```

#### Developing the prediction pipeline

- Create the `prediction.py` under `telco_churn_kfp/telco_churn_kfp` directory

```python
import json
import pathlib
from jinja2 import Template

from kfp import compiler, dsl
from telco_churn_kfp.components import bq_create_table


@dsl.pipeline(name="prediction-pipeline")
def prediction_pipeline(
    project_id: str,
    dataset_id: str,
    location: str,
    model_id: str,
    score_table_id: str,
    label_name: str
):
    predict_query_template_file = pathlib.Path(__file__).parent / "queries" / "predict_model.sql"
    with open(predict_query_template_file) as f:
        predict_query_template = f.read()
    predict_model_query = (
        Template(predict_query_template)
        .render(
            project_id=project_id, 
            dataset_id=dataset_id, 
            table_id=score_table_id,
            model_id=model_id,
            label=label_name
        )
    )
    predict_model = (
        bq_create_table(
            project_id=project_id,
            location=location,
            query=predict_model_query,
            dataset_id=dataset_id,
            table_id='telco_churn_xgb_baseline_pred'
        )
        .set_display_name('BQ Predict XGB model against scoring data')
    )
    

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=prediction_pipeline,
        package_path="prediction.json",
        type_check=False,
    )
```

#### Developing the trigger

- Create the `main.py` under `telco_churn_kfp/telco_churn_kfp` directory

```python
from google.cloud import aiplatform
from telco_churn_kfp.components import PROJECT_ID, LOCATION, SERVICE_ACCOUNT, PIPELINE_ROOT
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", help="Payload json file location", type=str)
    args = parser.parse_args()

    with open(args.payload) as f:
        param_values = json.load(f)
        
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    pl = aiplatform.PipelineJob(
        display_name=param_values['attributes']['display_name'],
        enable_caching=False,
        template_path=param_values['attributes']['template_path'],
        parameter_values=param_values['data'],
        pipeline_root=PIPELINE_ROOT
    )

    pl.submit(
        service_account=SERVICE_ACCOUNT,
    )
    
    pl.wait()
```

#### Creating docker file to create a docker image

- Create the `Dockerfile` under `telco_churn_kfp` directory

```shell
FROM python:3.10

RUN pip install kfp==2.3.0
RUN pip install google-cloud-aiplatform==1.7.1
RUN pip install Jinja2==3.1.2
```

#### Creating cloud build yaml files

- Create the `image.yaml` under `telco_churn_kfp` directory. *(Do not forget to replace PROJECT_ID)*

```yaml
---
steps:
  - name: gcr.io/cloud-builders/docker
    entrypoint: /bin/bash
    args:
      - -c
      - |
        docker build -t europe-west1-docker.pkg.dev/<PROJECT_ID>/telco-churn-hub/python310-cicd:latest .
        docker push europe-west1-docker.pkg.dev/<PROJECT_ID>/telco-churn-hub/python310-cicd:latest
```

- Create the `deploy.yaml` under `telco_churn_kfp` directory. *(Do not forget to replace PROJECT_ID)*

```yaml
---
steps:
  - name: europe-west1-docker.pkg.dev/<PROJECT_ID>/telco-churn-hub/python310-cicd:latest
    entrypoint: /bin/bash
    args:
      - -c
      - |
        python -m telco_churn_kfp.training && \
        python -m telco_churn_kfp.prediction
  
  - name: europe-west1-docker.pkg.dev/<PROJECT_ID>/telco-churn-hub/python310-cicd:latest
    entrypoint: /bin/bash
    args:
      - -c
      - |
        python -m telco_churn_kfp.main --payload telco_churn_kfp/payloads/training.json && \
        python -m telco_churn_kfp.main --payload telco_churn_kfp/payloads/prediction.json
    env:
      - GOOGLE_APPLICATION_CREDENTIALS=sa-key.json
```
#### Seting-up Github Triggers

Navigate the Github Triggers under `Cloud Build`:

- Connect to the Github Host by navigating the wizard, install the required app, Connecti to the Github repository
- Create a trigger under `Triggers` section
    + Trigger name: `telco-churn-image`
    + Region: `<REGION>`
    + Event: `pull-request`
    + Source: `1st gen` and Select Repository
    + Base branch: `^main$`
    + `Required except for owners and collaborators`
    + `Cloud Build configuration file (YAML or JSON)`
    + Cloud Build configuration file location: `image.yaml`
    + Service account email: `<SERVICE ACCOUNT>`
- Create a trigger under `Triggers` section
    + Trigger name: `telco-churn-deploy`
    + Region: `<REGION>`
    + Event: `pull-request`
    + Source: `1st gen` and Select Repository
    + Base branch: `^main$`
    + `Required except for owners and collaborators`
    + `Cloud Build configuration file (YAML or JSON)`
    + Cloud Build configuration file location: `deploy.yaml`
    + Service account email: `<SERVICE ACCOUNT>`

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

#### Triggering Cloud Build with Github Events

- Check the changes

```shell
cd telco_churn_kfp
git status
```

- Commit the changes

```shell
git add .
git commit -m "development changes"
git push
```

- Create a Pull Request from `feature-develop` to `main`. This will start `telco-churn-image` trigger.

- Once the Pull Request has been finished successfully, merge the pull request under `main` branch. This will start `telco-churn-deploy` trigger.
