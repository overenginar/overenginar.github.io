---
layout: post
title:  "Fine Tuning Examples"
date:   2023-08-11 12:30:00 +0000
categories: fine-tuning llm tokenizers transformers datasets
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/tree/published/LLM%2004%20-%20Fine-tuning%20and%20Evaluating%20LLMs*

### Examples

> Installing Libraries:

```shell
pip install transformers==4.31.0 --user
pip install datasets==2.14.4 --user
pip install tensorboard==2.14.0 --user
```

> Create a temp directory

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

> Load dataset:

```py
imdb_ds = load_dataset("imdb")
```

> Model checkpoint name:

```py
model_checkpoint = "t5-small"
```

> Tokenizer:

```py
tokenizer = tr.AutoTokenizer.from_pretrained(
    model_checkpoint
)
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

> Data transformation:

```py
imdb_label_lookup = {0: "negative", 1: "positive", -1: "unknown"}
imdb_to_tokens = to_tokens(tokenizer, imdb_label_lookup)
tokenized_dataset = imdb_ds.map(
    imdb_to_tokens, batched=True, remove_columns=["text", "label"]
)
```

> Training arguments:

```py
checkpoint_name = "test-trainer"
local_checkpoint_path = os.path.join(local_training_root, checkpoint_name)
training_args = tr.TrainingArguments(
    local_checkpoint_path,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    optim="adamw_torch",
    report_to=["tensorboard"],
)
```

> Loading model:

```py
model = tr.AutoModelForSeq2SeqLM.from_pretrained(
    model_checkpoint
)
```

> Building trainer:

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

> Tensorboard display directory:

```py
tensorboard_display_dir = f"{local_checkpoint_path}/runs"
```

> You can monitor tensorboard if you use in notebooks:

```
%load_ext tensorboard
%tensorboard --logdir '{tensorboard_display_dir}'
```

> Training model:

```py
trainer.train()
```

> Saving model:

```py
trainer.save_model()
trainer.save_state()
```

> Saving the final model:

```py
final_model_path = f"fine_tuning_examples/{checkpoint_name}"
trainer.save_model(output_dir=final_model_path)
```

> Loading fine tuned model:

```py
fine_tuned_model = tr.AutoModelForSeq2SeqLM.from_pretrained(final_model_path)
```

> Predictions along with transformations:

```py
reviews = [
    """
'Despicable Me' is a cute and funny movie, but the plot is predictable and the characters are not very well-developed. Overall, it's a good movie for kids, but adults might find it a bit boring.""",
    """ 'The Batman' is a dark and gritty take on the Caped Crusader, starring Robert Pattinson as Bruce Wayne. The film is a well-made crime thriller with strong performances and visuals, but it may be too slow-paced and violent for some viewers.
""",
    """
The Phantom Menace is a visually stunning film with some great action sequences, but the plot is slow-paced and the dialogue is often wooden. It is a mixed bag that will appeal to some fans of the Star Wars franchise, but may disappoint others.
""",
    """
I'm not sure if The Matrix and the two sequels were meant to have a tigh consistency but I don't think they quite fit together. They seem to have a reasonably solid arc but the features from the first aren't in the second and third as much, instead the second and third focus more on CGI battles and more visuals. I like them but for different reasons, so if I'm supposed to rate the trilogy I'm not sure what to say.
""",
]
inputs = tokenizer(reviews, return_tensors="pt", truncation=True, padding=True)
pred = fine_tuned_model.generate(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
)
```

> Decode the predictions:

```py
pdf = pd.DataFrame(
    zip(reviews, tokenizer.batch_decode(pred, skip_special_tokens=True)),
    columns=["review", "classification"],
)
pdf
```

> Clear the temp dir:

```py
tmpdir.cleanup()
```
