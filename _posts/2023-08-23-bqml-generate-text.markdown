---
layout: post
title:  "Generating Text with BQML"
date:   2023-08-23 06:30:00 +0000
categories: bqml generative-ai text-generation bigquery llm
author: Ali Cabukel
---

*Reference: https://cloud.google.com/bigquery/docs/generate-text-tutorial*

BigQuery ML offers several impressive text generation capabilities, albeit currently confined to specific regions. This blog post will focus on utilizing the `US multi-region` setting as we embark on the installation process.

### Installation and Configuration

1) `bqml_tutorial` dataset was created from the console by selecting `US` multi-region as location type
2) Creating external connection


```sh
PROJECT_ID=<>
bq mk --connection --location=US --project_id=$PROJECT_ID \
    --connection_type=CLOUD_RESOURCE connection_test
```

```sh
bq show --connection $PROJECT_ID.us.connection_test
```

3) Granting connection service account to `Vertex AI > Vertex AI User` role on IAM

4) Running the following query from the console in order to create a connection with a remote `LLM` model.

```sql
CREATE OR REPLACE MODEL bqml_tutorial.llm_model
  REMOTE WITH CONNECTION `us.connection_test`
  OPTIONS (remote_service_type = 'CLOUD_AI_LARGE_LANGUAGE_MODEL_V1');
```

### Extracting the keywords from idmb reviews

> Query:

```sql
SELECT
  ml_generate_text_result['predictions'][0]['content'] AS generated_text,
  ml_generate_text_result['predictions'][0]['safetyAttributes']
    AS safety_attributes,
  * EXCEPT (ml_generate_text_result)
FROM
  ML.GENERATE_TEXT(
    MODEL `bqml_tutorial.llm_model`,
    (
      SELECT
        CONCAT('Extract the key words from the text below: ', review) AS prompt,
        *
      FROM
        `bigquery-public-data.imdb.reviews`
      LIMIT 5
    ),
    STRUCT(
      0.2 AS temperature,
      100 AS max_output_tokens));
```
 
> Example Prompt:

```
Extract the key words from the text below: I had to see this on the British Airways plane. It was terribly bad acting and a dumb story. Not even a kid would enjoy this. Something to switch off if possible.
```

> Output (generated_text):

```
The key words in the text are:

act
enjoy
plane
story
switch
```

> Output (generated_text):

```
{"blocked":false,"categories":["Insult","Toxic"],"scores":[0.4,0.4]}
```

> Output columns:

- **generated_text:** the generated text.
- **safety_attributes:** the safety attributes, along with information about whether the content is blocked due to one of the blocking categories. For more information about the safety attributes, see Vertex PaLM API.
- **ml_generate_text_status:** the API response status for the corresponding row. If the operation was successful, this value is empty.
- **prompt:** the prompt that is used for the sentiment analysis.

### Analysing keyword extraction

> Query: with `flatten_json_output` and `LIMIT 100`

```sql
CREATE OR REPLACE TABLE bqml_tutorial.extract_kw_imdb_reviews_100
AS
SELECT
  *
FROM
  ML.GENERATE_TEXT(
    MODEL `bqml_tutorial.llm_model`,
    (
      SELECT
        CONCAT('Extract the key words from the text below: ', review) AS prompt,
        *
      FROM
        `bigquery-public-data.imdb.reviews`
      LIMIT 100
    ),
    STRUCT(
      0.2 AS temperature,
      100 AS max_output_tokens,
      TRUE AS flatten_json_output));
```

> Output columns:

- **ml_generate_text_llm_result:** the generated text.
- **ml_generate_text_rai_result:** the safety attributes, along with information about whether the content is blocked due to one of the blocking categories. For more information about the safety attributes, see Vertex PaLM API.
- **ml_generate_text_status:** the API response status for the corresponding row. If the operation was successful, this value is empty.
- **prompt:** the prompt that is used for the keyword extraction.

> Query: parsing struct and array fields into tabular format

```sql
WITH rai_result AS(
SELECT *, 
      BOOL(ml_generate_text_rai_result.blocked) AS blocked
FROM bqml_tutorial.extract_kw_imdb_reviews_100
)
SELECT  prompt, ml_generate_text_llm_result, category, score
FROM rai_result
CROSS JOIN UNNEST(JSON_EXTRACT_ARRAY(ml_generate_text_rai_result.categories)) category WITH OFFSET cid
LEFT JOIN UNNEST(JSON_EXTRACT_ARRAY(ml_generate_text_rai_result.scores)) score WITH OFFSET sid
ON cid = sid
WHERE LENGTH(ml_generate_text_status) = 0
AND blocked is false
LIMIT 10
```

### Sentiment Analysis

> Query:

```sql
CREATE OR REPLACE TABLE bqml_tutorial.sentiment_analysis_imdb_reviews_100
AS
SELECT
  ml_generate_text_result['predictions'][0]['content'] AS generated_text,
  ml_generate_text_result['predictions'][0]['safetyAttributes']
    AS safety_attributes,
  * EXCEPT (ml_generate_text_result)
FROM
  ML.GENERATE_TEXT(
    MODEL `bqml_tutorial.llm_model`,
    (
      SELECT
        CONCAT(
          'perform sentiment analysis on the following text, return one the following categories: positive, negative: ',
          review) AS prompt,
        *
      FROM
        `bigquery-public-data.imdb.reviews`
      LIMIT 100
    ),
    STRUCT(
      0.2 AS temperature,
      100 AS max_output_tokens));
```

> Query: blocked results

```sql
select * 
from bqml_tutorial.sentiment_analysis_imdb_reviews_100
where BOOL(safety_attributes.blocked) is true
limit 10
```

> Query: aggregating sentiments

```sql
select string(generated_text), count(1)
from bqml_tutorial.sentiment_analysis_imdb_reviews_100
where BOOL(safety_attributes.blocked) is false
group by string(generated_text)
```
