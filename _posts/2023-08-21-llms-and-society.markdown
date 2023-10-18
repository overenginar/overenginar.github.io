---
layout: post
title:  "LLMs and Society"
date:   2023-08-21 08:30:00 +0000
categories: llmops mlflow
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/tree/published/LLM%2005%20-%20Society%20and%20LLMs*

### Examples

> Install packages:

```shell
pip install transformers==4.31.0 --user
pip install datasets==2.14.4 --user

pip install disaggregators==0.1.2 https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl
pip install shap
pip install nlptest=1.4.0

```

> Disaggregator:

```py
from disaggregators import Disaggregator

disaggregator = Disaggregator("pronoun", column="target_text")
# disaggregator = Disaggregator("gender", column="target_text")
# disaggregator = Disaggregator("continent", column="target_text")
# disaggregator = Disaggregator("religion", column="target_text")
# disaggregator = Disaggregator("age", column="target_text")
```

> Load dataset:

```py
from datasets import load_dataset

wiki_data = load_dataset(
    "wiki_bio", split="test"
)
ds = wiki_data.map(disaggregator)
pdf = ds.to_pandas()
pdf
```

```py
import json

print(pdf.iloc[[19], :].to_json(indent=4))
```

> Split arrays based on pronoun:

```py
import numpy as np
she_array = np.where(pdf["pronoun.she_her"] == True)
print(f"she_her: {len(she_array[0])} rows")
he_array = np.where(pdf["pronoun.he_him"] == True)
print(f"he_him: {len(he_array[0])} rows")
```

> Unmasker pipeline:

```py
from transformers import pipeline

unmasker = pipeline(
    "fill-mask",
    model="bert-base-uncased",
)
```

> Fill the mask:

```py
result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])
```

> Calculate toxicity scores:

```py
import evaluate

toxicity = evaluate.load("toxicity", module_type="measurement")
```

```py
candidates = [
    "their kid loves reading books",
    "she curses and makes fun of people",
    "he is a wimp and pathetic loser",
]
toxicity.compute(predictions=candidates)
```

> GPT2 model

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import shap

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2", use_fast=True
)
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

> Model comfig:

```py
model.config.is_decoder = True
```

```py
model.config.task_specific_params["text-generation"] = {
    "do_sample": True,
    "max_length": 50,
    "temperature": 0,
    "top_k": 50,
    "no_repeat_ngram_size": 2,
}
```

> Explain the predictions:

```py
input_sentence = ["Sunny days are the best days to go to the beach. So"]
```

```py
explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(input_sentence)
```

```py
shap.plots.text(shap_values)
```

```py
shap.plots.bar(shap_values[0, :, "looking"])
```

```py
input_sentence2 = ["I know many people who prefer beaches to the mountains"]
shap_values2 = explainer(input_sentence2)
shap.plots.text(shap_values2)
```

```py
shap.plots.bar(shap_values2[0, :, "not"])
```

```py
input_sentence3 = ["Can you stop the dog from"]
shap_values3 = explainer(input_sentence3)
shap.plots.text(shap_values3)
```

```py
from transformers import GPT2Tokenizer, GPT2LMHeadModel
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2",
)
gpt2_model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
)
input_tokens = gpt2_tokenizer(input_sentence3[0])["input_ids"]
attention_ids = gpt2_tokenizer(input_sentence3[0])["attention_mask"]
```

```shell
git clone https://github.com/kayoyin/interpret-lm.git
```

```py
import lm_saliency
from lm_saliency import *
target = "barking"
foil = "crying"
explanation = "erasure"
CORRECT_ID = gpt2_tokenizer(" " + target)["input_ids"][0]
FOIL_ID = gpt2_tokenizer(" " + foil)["input_ids"][0]
base_explanation = erasure_scores(gpt2_model, input_tokens, attention_ids, normalize=True)
contra_explanation = erasure_scores(gpt2_model, input_tokens, attention_ids, correct=CORRECT_ID, foil=FOIL_ID, normalize=True)
visualize(np.array(base_explanation), gpt2_tokenizer, [input_tokens], print_text=True, title=f"Why did the model predict {target}?")
visualize(np.array(contra_explanation), gpt2_tokenizer, [input_tokens], print_text=True, title=f"Why did the model predict {target} instead of {foil}?")
```


```py
from datasets import load_dataset

bold = load_dataset(
    "AlexaAI/bold", split="train"
) 

from random import sample

def generate_samples(category_name: str, n: int) -> list:
    """
    Given a category, returns `n` samples
    """
    bold_samples = sample([p for p in bold if p["category"] == category_name], n)
    return bold_samples


science_bold = generate_samples("scientific_occupations", 10)
dance_bold = generate_samples("dance_occupations", 10)

print("Science example: ", science_bold[0])
print("-" * 60)
print("Dance example: ", dance_bold[0])

import numpy as np

np.unique(bold["category"])

group1_bold = generate_samples("American_actors", 10)
group2_bold = generate_samples("American_actresses", 10)

science_prompts = [p["prompts"][0] for p in science_bold]
dance_prompts = [p["prompts"][0] for p in dance_bold]
print("Science prompt example: ", science_prompts[0])
print("Dance prompt example: ", dance_prompts[0])

group1_prompts = [p["prompts"][0] for p in group1_bold]
group2_prompts = [p["prompts"][0] for p in group2_bold]

from transformers import pipeline, AutoTokenizer

text_generation = pipeline(
    "text-generation", model="gpt2"
)

def complete_sentence(text_generation_pipeline: pipeline, prompts: list) -> list:
    """
    Via a list of prompts a prompt list is appended to by the generated `text_generation_pipeline`.
    """
    prompt_continuations = []
    for prompt in prompts:
        generation = text_generation_pipeline(
            prompt, max_length=30, do_sample=False, pad_token_id=50256
        )
        continuation = generation[0]["generated_text"].replace(prompt, "")
        prompt_continuations.append(continuation)
    return prompt_continuations

dance_continuation = complete_sentence(text_generation, dance_prompts)

science_continuation = complete_sentence(text_generation, science_prompts)

group1_continuation = complete_sentence(text_generation, group1_prompts)
group2_continuation = complete_sentence(text_generation, group2_prompts)

import evaluate

regard = evaluate.load("regard", "compare")

regard.compute(data=science_continuation, references=dance_continuation)

regard.compute(data=group1_continuation, references=group2_continuation)

from nlptest import Harness

h = Harness(task="ner", model="dslim/bert-base-NER", hub="huggingface")

h.generate().run().report()
 
```