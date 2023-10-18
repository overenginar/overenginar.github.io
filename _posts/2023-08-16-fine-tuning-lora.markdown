---
layout: post
title:  "Fine Tuning falcon-7b with Lora"
date:   2023-08-16 12:30:00 +0000
categories: fine-tuning llm tokenizers transformers datasets lora chatbot falcon
author: Ali Cabukel
---

*Reference: https://github.com/curiousily/Get-Things-Done-with-Prompt-Engineering-and-LangChain/blob/master/07.falcon-qlora-fine-tuning.ipynb*
*Reference: https://github.com/facebookresearch/llama-recipes/blob/main/quickstart.ipynb*


### Examples

> Installing Libraries:

```shell
pip install bitsandbytes==0.39.0 --user --force-reinstall
pip install peft==0.4.0 --user --force-reinstall
pip install transformers[torch]==4.31.0 --user --force-reinstall
pip install torch==2.0.1+cu118 -i https://download.pytorch.org/whl/cu118 --user --force-reinstalls
pip install accelerate==0.21.0 --user --force-reinstall
pip install datasets==2.14.4 --user --force-reinstall
pip install evaluate==0.4.0 --user --force-reinstall
pip install loralib==0.1.1 --user --force-reinstall
pip install einops==0.6.1 --user --force-reinstall
pip install tensorboard==2.14.0 --user --force-reinstall
```

> Importing Libraries

```py
import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

> Load model and tokenizer:

```py
MODEL_NAME = "tiiuae/falcon-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
```

> Trainable parameters:

```py
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
```

> model config:

```py
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```

> Lora config:

```py
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)
```

> Format the prompt:

```py
prompt = f"""
: How can I create an account?
:
""".strip()
print(prompt)
```

> Model config:

```py
generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
```

> Predictions with base model:

```py
device = "cuda:0"

encoding = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> load dataset:

```py
ds = load_dataset('Andyrasika/Ecommerce_FAQ')
```

> Generate and tokenize prompt:

```py
def generate_prompt(data_point):
    return f"""
: {data_point["question"]}
: {data_point["answer"]}
""".strip()


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
    return tokenized_full_prompt
```

> Prepare training data:

```py
ds = ds["train"].shuffle().map(generate_and_tokenize_prompt)
```

> Output directory:

```py
OUTPUT_DIR = "experiments"
```

> Tensboard:

```
%load_ext tensorboard
%tensorboard --logdir experiments/runs
```

> Training Arguments, Trainer and train the model:

```py
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir=OUTPUT_DIR,
    max_steps=80,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="tensorboard",
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=ds,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()
```

> Save tuned model:

```py
model.save_pretrained("trained-model")
```

> Push to hugging face:

```py
model.push_to_hub(
    "overenginar/falcon-7b-qlora-chat-support-bot-faq", use_auth_token=True
)
```

> Load tuned model:

```py
PEFT_MODEL = "overenginar/falcon-7b-qlora-chat-support-bot-faq"

config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, PEFT_MODEL)
```

> Model config:

```py
generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
```

> GPU device:

```py
DEVICE = "cuda:0"
```

> Inference:

```py
prompt = f"""
: How can I create an account?
:
""".strip()

encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
with torch.inference_mode():
    outputs = model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

```py
def generate_response(question: str) -> str:
    prompt = f"""
: {question}
:
""".strip()
    encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    assistant_start = ":"
    response_start = response.find(assistant_start)
    return response[response_start + len(assistant_start) :].strip()
```

```py
prompt = "Can I return a product if it was a clearance or final sale item?"
print(generate_response(prompt))
```

```py
prompt = "What happens when I return a clearance item?"
print(generate_response(prompt))
```

```py
prompt = "How do I know when I'll receive my order?"
print(generate_response(prompt))
```
