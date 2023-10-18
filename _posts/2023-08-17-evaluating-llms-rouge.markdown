---
layout: post
title:  "Evaluating Large Language Models - Rouge"
date:   2023-08-17 08:30:00 +0000
categories: llm evaluation rouge
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/blob/published/LLM%2004%20-%20Fine-tuning%20and%20Evaluating%20LLMs/LLM%2004b%20-%20Evaluating%20LLMs.py*

### Examples

> Install packages:

```shell
pip install transformers==4.31.0 --user
pip install rouge_score==0.1.2 --user
pip install datasets==2.14.4 --user
```

> Load data and sample:

```py
import torch
from datasets import load_dataset

full_dataset = load_dataset(
    "cnn_dailymail", version="3.0.0"
) 
sample_size = 100
sample = (
    full_dataset["train"]
    .filter(lambda r: "CNN" in r["article"][:25])
    .shuffle(seed=42)
    .select(range(sample_size))
)
sample
```

> Sample as pandas df:

```py
sample.to_pandas()
```

> Example article and summary:

```py
example_article = sample["article"][0]
example_summary = sample["highlights"][0]
print(f"Article:\n{example_article}\n")
print(f"Summary:\n{example_summary}")
```

> Summarize with t5 function:

```py
import pandas as pd
import torch
import gc
from transformers import AutoTokenizer, T5ForConditionalGeneration

def batch_generator(data: list, batch_size: int):
    """
    Creates batches of size `batch_size` from a list.
    """
    s = 0
    e = s + batch_size
    while s < len(data):
        yield data[s:e]
        s = e
        e = min(s + batch_size, len(data))


def summarize_with_t5(
    model_checkpoint: str, articles: list, batch_size: int = 8
) -> list:
    """
    Compute summaries using a T5 model.
    This is similar to a `pipeline` for a T5 model but does tokenization manually.

    :param model_checkpoint: Name for a model checkpoint in Hugging Face, such as "t5-small" or "t5-base"
    :param articles: List of strings, where each string represents one article.
    :return: List of strings, where each string represents one article's generated summary
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = T5ForConditionalGeneration.from_pretrained(
        model_checkpoint
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, model_max_length=1024
    )

    def perform_inference(batch: list) -> list:
        inputs = tokenizer(
            batch, max_length=1024, return_tensors="pt", padding=True, truncation=True
        )

        summary_ids = model.generate(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            num_beams=2,
            min_length=0,
            max_length=40,
        )
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    res = []

    summary_articles = list(map(lambda article: "summarize: " + article, articles))
    for batch in batch_generator(summary_articles, batch_size=batch_size):
        res += perform_inference(batch)

        torch.cuda.empty_cache()
        gc.collect()

    del tokenizer
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return res
```

> summarize the sample articles:

```py
t5_small_summaries = summarize_with_t5("t5-small", sample["article"])
```

>  ref summaries:

```py
reference_summaries = sample["highlights"]
```

> Generated and reference texts:

```py
(
    pd.DataFrame.from_dict(
        {
            "generated": t5_small_summaries,
            "reference": reference_summaries,
        }
    )
)
```

> Accuracy:

```py
accuracy = 0.0
for i in range(len(reference_summaries)):
    generated_summary = t5_small_summaries[i]
    if generated_summary == reference_summaries[i]:
        accuracy += 1.0
accuracy = accuracy / len(reference_summaries)

print(f"Achieved accuracy {accuracy}!")
```

> Rouge score:

```py
import evaluate
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

rouge_score = evaluate.load("rouge")
```

> Compute rouge score function:

```py
def compute_rouge_score(generated: list, reference: list) -> dict:
    """
    Compute ROUGE scores on a batch of articles.

    This is a convenience function wrapping Hugging Face `rouge_score`,
    which expects sentences to be separated by newlines.

    :param generated: Summaries (list of strings) produced by the model
    :param reference: Ground-truth summaries (list of strings) for comparison
    """
    generated_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in generated]
    reference_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in reference]
    return rouge_score.compute(
        predictions=generated_with_newlines,
        references=reference_with_newlines,
        use_stemmer=True,
    )
```

> Rouge scores:

```py
compute_rouge_score(t5_small_summaries, reference_summaries)
```

```py
compute_rouge_score(reference_summaries, reference_summaries)
```

```py
compute_rouge_score(
    generated=["" for _ in range(len(reference_summaries))],
    reference=reference_summaries,
)
```

> Compare models:

```py
rouge_score.compute(
    predictions=["Large language models beat world record"],
    references=["Large language models beating world records"],
    use_stemmer=False,
)
```

```py
rouge_score.compute(
    predictions=["Large language models beat world record"],
    references=["Large language models beating world records"],
    use_stemmer=True,
)
```

```py
rouge_score.compute(
    predictions=["Large language models beat world record"],
    references=["Large"],
    use_stemmer=True,
)
```

```py
rouge_score.compute(
    predictions=["Large"],
    references=["Large language models beat world record"],
    use_stemmer=True,
)
```

```py
rouge_score.compute(
    predictions=["Large language"],
    references=["Large language models beat world record"],
    use_stemmer=True,
)
```

```py
rouge_score.compute(
    predictions=["Models beat large language world record"],
    references=["Large language models beat world record"],
    use_stemmer=True,
)
```

> Compute rouge per row:

```py
def compute_rouge_per_row(
    generated_summaries: list, reference_summaries: list
) -> pd.DataFrame:
    """
    Generates a dataframe to compare rogue score metrics.
    """
    generated_with_newlines = [
        "\n".join(sent_tokenize(s.strip())) for s in generated_summaries
    ]
    reference_with_newlines = [
        "\n".join(sent_tokenize(s.strip())) for s in reference_summaries
    ]
    scores = rouge_score.compute(
        predictions=generated_with_newlines,
        references=reference_with_newlines,
        use_stemmer=True,
        use_aggregator=False,
    )
    scores["generated"] = generated_summaries
    scores["reference"] = reference_summaries
    return pd.DataFrame.from_dict(scores)
```


```py
compute_rouge_score(t5_small_summaries, reference_summaries)
```

```py
t5_small_results = compute_rouge_per_row(
    generated_summaries=t5_small_summaries, reference_summaries=reference_summaries
)
t5_small_results
```

```py
t5_base_summaries = summarize_with_t5(
    model_checkpoint="t5-base", articles=sample["article"]
)
compute_rouge_score(t5_base_summaries, reference_summaries)
```

```py
t5_base_results = compute_rouge_per_row(
    generated_summaries=t5_base_summaries, reference_summaries=reference_summaries
)
t5_base_results
```

> Summarize with gpt2:

```py
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def summarize_with_gpt2(
    model_checkpoint: str, articles: list, batch_size: int = 8
) -> list:
    """
    Convenience function for summarization with GPT2 to handle these complications:
    - Append "TL;DR" to the end of the input to get GPT2 to generate a summary.
    https://huggingface.co/course/chapter7/5?fw=pt
    - Truncate input to handle long articles.
    - GPT2 uses a max token length of 1024.  We use a shorter 512 limit here.

    :param model_checkpoint: reference to checkpointed model
    :param articles: list of strings
    :return: generated summaries, with the input and "TL;DR" removed
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained(
        model_checkpoint, padding_side="left"
    )
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model = GPT2LMHeadModel.from_pretrained(
        model_checkpoint,
        pad_token_id=tokenizer.eos_token_id,
    ).to(device)

    def perform_inference(batch: list) -> list:
        tmp_inputs = tokenizer(
            batch, max_length=500, return_tensors="pt", padding=True, truncation=True
        )
        tmp_inputs_decoded = tokenizer.batch_decode(
            tmp_inputs.input_ids, skip_special_tokens=True
        )
        inputs = tokenizer(
            [article + " TL;DR:" for article in tmp_inputs_decoded],
            max_length=512,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        summary_ids = model.generate(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            num_beams=2,
            min_length=0,
            max_length=512 + 32,
        )
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    decoded_summaries = []
    for batch in batch_generator(articles, batch_size=batch_size):
        decoded_summaries += perform_inference(batch)

        # batch clean up
        torch.cuda.empty_cache()
        gc.collect()

    # post-process decoded summaries
    summaries = [
        summary[summary.find("TL;DR:") + len("TL;DR: ") :]
        for summary in decoded_summaries
    ]

    # cleanup
    del tokenizer
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return summaries
```

```py
gpt2_summaries = summarize_with_gpt2(
    model_checkpoint="gpt2", articles=sample["article"]
)
compute_rouge_score(gpt2_summaries, reference_summaries)
```

```py
gpt2_results = compute_rouge_per_row(
    generated_summaries=gpt2_summaries, reference_summaries=reference_summaries
)
gpt2_results
```

> Compare models:

```py
def compare_models(models_results: dict) -> pd.DataFrame:
    """
    :param models_results: dict of "model name" string mapped to pd.DataFrame of results computed by `compute_rouge_per_row`
    :return: pd.DataFrame with 1 row per model, with columns: model, rouge1, rouge2, rougeL, rougeLsum
    where metrics are averages over input results for each model
    """
    agg_results = []
    for r in models_results:
        model_results = models_results[r].drop(
            labels=["generated", "reference"], axis=1
        )
        agg_metrics = [r]
        agg_metrics[1:] = model_results.mean(axis=0)
        agg_results.append(agg_metrics)
    return pd.DataFrame(
        agg_results, columns=["model", "rouge1", "rouge2", "rougeL", "rougeLsum"]
    )
```

```py
(
    compare_models(
        {
            "t5-small": t5_small_results,
            "t5-base": t5_base_results,
            "gpt2": gpt2_results,
        }
    )
)
```

> Compare models summaries:

```py
def compare_models_summaries(models_summaries: dict) -> pd.DataFrame:
    """
    Aggregates results from `models_summaries` and returns a dataframe.
    """
    comparison_df = None
    for model_name in models_summaries:
        summaries_df = models_summaries[model_name]
        if comparison_df is None:
            comparison_df = summaries_df[["generated"]].rename(
                {"generated": model_name}, axis=1
            )
        else:
            comparison_df = comparison_df.join(
                summaries_df[["generated"]].rename({"generated": model_name}, axis=1)
            )
    return comparison_df
```

```py
(
    compare_models_summaries(
        {
            "t5_small": t5_small_results,
            "t5_base": t5_base_results,
            "gpt2": gpt2_results,
        }
    )
)
```
