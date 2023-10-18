---
layout: post
title:  "Fine Tuning with Deepspeed"
date:   2023-08-14 12:30:00 +0000
categories: fine-tuning llm tokenizers transformers datasets deepspeed
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/tree/published/LLM%2004%20-%20Fine-tuning%20and%20Evaluating%20LLMs*

### Examples

> Installing Libraries:

```shell
pip install sqlalchemy==1.4.49 --user
pip install transformers==4.31.0 --user
pip install datasets==2.14.4 --user
pip install torch==2.0.0+cu118 -i https://download.pytorch.org/whl/cu118 --user
pip install py-cpuinfo==9.0.0 --user
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
import transformers as tr
from datasets import load_dataset
```

> Deepspeed env variables:

```py
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
```

> Deepspeed zero config:

```py
zero_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
            "torch_adam": True,
        },
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

> to_tokens function:

```py
def to_tokens(
    tokenizer: tr.models.t5.tokenization_t5_fast.T5TokenizerFast, label_map: dict
) -> callable:
    """
    Given a `tokenizer` this closure will iterate through `x` and return the result of `apply()`.
    This function is mapped to a dataset and returned with ids and attention mask.
    """

    def apply(x) -> tr.tokenization_utils_base.BatchEncoding:
        """From a formatted dataset `x` a batch encoding `token_res` is created."""
        target_labels = [label_map[y] for y in x["label"]]
        token_res = tokenizer(
            x["text"],
            text_target=target_labels,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        return token_res

    return apply
```

> Load data, tokenize, transform and load model:

```py
model_checkpoint = "t5-base"
imdb_label_lookup = {0: "negative", 1: "positive", -1: "unknown"}

imdb_ds = load_dataset("imdb")

tokenizer = tr.AutoTokenizer.from_pretrained(
    model_checkpoint
)

imdb_to_tokens = to_tokens(tokenizer, imdb_label_lookup)
tokenized_dataset = imdb_ds.map(
    imdb_to_tokens, batched=True, remove_columns=["text", "label"]
)

model = tr.AutoModelForSeq2SeqLM.from_pretrained(
    model_checkpoint
)
```

> Training arguments:

```py
checkpoint_name = "test-trainer-deepspeed"
checkpoint_location = os.path.join(local_training_root, checkpoint_name)
training_args = tr.TrainingArguments(
    checkpoint_location,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    deepspeed=zero_config,
    report_to=["tensorboard"],
)
```

> Trainer:

```py
data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)
trainer = tr.Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

> Tensorboard dir:

```py
tensorboard_display_dir = f"{checkpoint_location}/runs"
```

> Monitor tensorboard on notebooks:

```
%load_ext tensorboard
%tensorboard --logdir '{tensorboard_display_dir}'
```

> Training:

```py
trainer.train()
```

> Save model and state:

```py
trainer.save_model()
trainer.save_state()
```

> Save the final model:

```py
final_model_path = f"fine_tuning_examples/{checkpoint_name}"
trainer.save_model(output_dir=final_model_path)
```

> Import pre-trained model:

```py
fine_tuned_model = tr.AutoModelForSeq2SeqLM.from_pretrained(final_model_path)
```

> Predictions:

```py
review = [
    """
           I'm not sure if The Matrix and the two sequels were meant to have a tight consistency but I don't think they quite fit together. They seem to have a reasonably solid arc but the features from the first aren't in the second and third as much, instead the second and third focus more on CGI battles and more visuals. I like them but for different reasons, so if I'm supposed to rate the trilogy I'm not sure what to say."""
]
inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)

pred = fine_tuned_model.generate(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
)
```

> Decode the predictions:

```py
pdf = pd.DataFrame(
    zip(review, tokenizer.batch_decode(pred, skip_special_tokens=True)),
    columns=["review", "classification"],
)
pdf
```

> Clear temp dir:

```py
tmpdir.cleanup()
```