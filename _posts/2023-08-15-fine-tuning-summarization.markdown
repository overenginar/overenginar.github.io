---
layout: post
title:  "Fine Tuning Summarization"
date:   2023-08-15 12:30:00 +0000
categories: fine-tuning llm tokenizers transformers datasets rouge summarization
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/tree/published/LLM%2004%20-%20Fine-tuning%20and%20Evaluating%20LLMs*

### Examples

> Installing Libraries:

```shell
pip install sqlalchemy==1.4.49 --user
pip install transformers[torch]==4.31.0 --user
pip install datasets==2.14.4 --user
pip install evaluate==0.4.0 --user
pip install nltk==3.8.1 --user
pip install rouge_score==0.1.2 --user
pip install tensorboard==2.14.0 --user
```

> Create a temp directory:

```py
import tempfile

tmpdir = tempfile.TemporaryDirectory()
local_training_root = tmpdir.name
```

> Importing Libraries:

```py
import os
import pandas as pd
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

import evaluate
import nltk
from nltk.tokenize import sent_tokenize
```

> Loading dataset:

```py
ds = load_dataset('databricks/databricks-dolly-15k')
```

> Model checkpoint:

```py
model_checkpoint = 'EleutherAI/pythia-70m-deduped'
```

> Tokenizer:

```py
tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["### End", "### Instruction:", "### Response:\n"]}
)
```

> tokenize function:

```py
remove_columns = ["instruction", "response", "context", "category"]

def tokenize(x: dict, max_length: int = 1024) -> dict:
    """
    For a dictionary example of instruction, response, and context a dictionary of input_id and attention mask is returned
    """
    instr = x["instruction"]
    resp = x["response"]
    context = x["context"]

    instr_part = f"### Instruction:\n{instr}"
    context_part = ""
    if context:
        context_part = f"\nInput:\n{context}\n"
    resp_part = f"### Response:\n{resp}"

    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

{instr_part}
{context_part}
{resp_part}

### End
"""
    return tokenizer(text, max_length=max_length, truncation=True)
```

> tokenize data:

```py
tokenized_dataset = ds.map(
    tokenize, batched=False, remove_columns=remove_columns
)
```


> Training arguments:

```py
checkpoint_name = "test-trainer-lab"
local_checkpoint_path = os.path.join(local_training_root, checkpoint_name)
training_args = TrainingArguments(
    local_checkpoint_path,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    optim='adamw_torch',
    report_to=["tensorboard"],
)
```

> model:

```py
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint
)
```

> split data and trainer:

```py
TRAINING_SIZE=6000
SEED=42
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
)
split_dataset = tokenized_dataset['train'].train_test_split(train_size=TRAINING_SIZE, seed=SEED)
trainer = Trainer(
    model,
    training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

> Tensorboard display dir:

```py
tensorboard_display_dir = f"{local_checkpoint_path}/runs"
```

> Tensorboard monitoring

```
%load_ext tensorboard
%tensorboard --logdir '{tensorboard_display_dir}'
```

> Training:

```py
trainer.train()
```

> Save model to local checkpoint:

```py
trainer.save_model()
trainer.save_state()
```

> Save model to final checkpoint:

```py
final_model_path = f"fine_tuning_examples/{checkpoint_name}"
trainer.save_model(output_dir=final_model_path)
```

> Garbage collection:

```py
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

> Import Fine-tuned model:

```py
fine_tuned_model = AutoModelForCausalLM.from_pretrained(final_model_path)
```

> Utility functions:

```py
def to_prompt(instr: str, max_length: int = 1024) -> dict:
    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instr}

### Response:
"""
    return tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)


def to_response(prediction):
    decoded = tokenizer.decode(prediction)
    m = re.search(r"#+\s*Response:\s*(.+?)#+\s*End", decoded, flags=re.DOTALL)
    res = "Failed to find response"
    if m:
        res = m.group(1).strip()
    else:
        m = re.search(r"#+\s*Response:\s*(.+)", decoded, flags=re.DOTALL)
        if m:
            res = m.group(1).strip()
    return res
```

> Predictions:

```py
import re
res = []
for i in range(100):
    instr = ds["train"][i]["instruction"]
    resp = ds["train"][i]["response"]
    inputs = to_prompt(instr)
    pred = fine_tuned_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=128,
    )
    res.append((instr, resp, to_response(pred[0])))
```

> Pandas data frame:

```py
pdf = pd.DataFrame(res, columns=["instruction", "response", "generated"])
pdf
```

> Rouge score function:

```py
nltk.download("punkt")

rouge_score = evaluate.load("rouge")

def compute_rouge_score(generated, reference):
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

> Evaluation:

```py
rouge_scores = compute_rouge_score(pdf['generated'], pdf['response'])
rouge_scores
```

> Clear the temp dir:

```py
tmpdir.cleanup()
```