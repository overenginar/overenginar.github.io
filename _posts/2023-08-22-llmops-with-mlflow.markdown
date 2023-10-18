---
layout: post
title:  "LLMOps with MLflow"
date:   2023-08-22 08:30:00 +0000
categories: llmops mlflow
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/blob/published/LLM%2006%20-%20LLMOps/LLM%2006%20-%20LLMOps.py*

### Examples

> Install packages:

```shell
pip install mlflow==2.6.0
pip install transformers==4.31.0 --user
pip install datasets==2.14.4 --user
```

> Import libraries:

```py
from datasets import load_dataset
from transformers import pipeline
```

> Load dataset:

```py
xsum_dataset = load_dataset(
    "xsum", version="1.2.0"
)
xsum_sample = xsum_dataset["train"].select(range(10))
xsum_sample.to_pandas()
```

> Summarizer pipeline:

```py
from transformers import pipeline
hf_model_name = "t5-small"
min_length = 20
max_length = 40
truncation = True
do_sample = True

summarizer = pipeline(
    task="summarization",
    model=hf_model_name,
    min_length=min_length,
    max_length=max_length,
    truncation=truncation,
    do_sample=do_sample,
)
```

> Sample data:

```py
doc0 = xsum_sample["document"][0]
print(f"Summary: {summarizer(doc0)[0]['summary_text']}")
print("===============================================")
print(f"Original Document: {doc0}")
```

> Predictions:

```py
import pandas as pd

results = summarizer(xsum_sample["document"])
pd.DataFrame(results, columns=["summary_text"])
```

> Experiment:

```py
import mlflow

mlflow.set_experiment("mlflow_experiments")

with mlflow.start_run():
    mlflow.log_params(
        {
            "hf_model_name": hf_model_name,
            "min_length": min_length,
            "max_length": max_length,
            "truncation": truncation,
            "do_sample": do_sample,
        }
    )

    results_list = [r["summary_text"] for r in results]

    mlflow.llm.log_predictions(
        inputs=xsum_sample["document"],
        outputs=results_list,
        prompts=["" for _ in results_list],
    )

    signature = mlflow.models.infer_signature(
        xsum_sample["document"][0],
        mlflow.transformers.generate_signature_output(
            summarizer, xsum_sample["document"][0]
        ),
    )
    print(f"Signature:\n{signature}\n")

    inference_config = {
        "min_length": min_length,
        "max_length": max_length,
        "truncation": truncation,
        "do_sample": do_sample,
    }

    model_info = mlflow.transformers.log_model(
        transformers_model=summarizer,
        artifact_path="summarizer",
        task="summarization",
        inference_config=inference_config,
        signature=signature,
        input_example="This is an example of a long news article which this pipeline can summarize for you.",
    )
```

> Access to the MLFlow server:

```shell
mlflow server --backend-store-uri file:////home/jupyter/mlruns --no-serve-artifacts
```

> Predictions:

```py
loaded_summarizer = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
loaded_summarizer.predict(xsum_sample["document"][0])
```

> Prediction with pandas:

```py
results = loaded_summarizer.predict(xsum_sample.to_pandas()["document"])
pd.DataFrame(results, columns=["generated_summary"])
```

> Clean model name:

```py
model_name = "summarizer"
model_name = model_name.replace("/", "_").replace(".", "_").replace(":", "_")
print(model_name)
```

> Register model:

```py
mlflow.register_model(model_uri=model_info.model_uri, name=model_name)  
```

> Search the registered model:

```py
from mlflow import MlflowClient

client = MlflowClient()

client.search_registered_models(filter_string=f"name = '{model_name}'")
```

> load registered model:

```py
model_version = 1
dev_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
dev_model
```

> get model to staging:

```py
client.transition_model_version_stage(model_name, model_version, "staging")
```

> dev to staging:

```py
staging_model = dev_model
```

> predictions with staging model:

```py
results = staging_model.predict(xsum_sample.to_pandas()["document"])
pd.DataFrame(results, columns=["generated_summary"])
```

> get model to production:

```py
client.transition_model_version_stage(model_name, model_version, "production")
```

> Create scoring data:

```py
xsum_dataset["test"].to_pandas().drop('summary', axis=1).to_csv('scoring_data.csv', index=None)
```

```py
from datasets import load_dataset
ds = load_dataset('csv', data_files={'test': 'scoring_data.csv'})

# import pandas as pd
# scoring_data = pd.read_csv('scoring_data.csv')

loaded_model = mlflow.pyfunc.load_model(
     model_uri=f"models:/{model_name}/Production",
)

loaded_model.predict(ds['test'].select(range(10)).to_pandas().astype(str))
# loaded_model.predict(scoring_data.head(10)['document'].astype(str))
```
