---
layout: post
title:  "Jekyll and Hyde Langchain with OpenAI"
date:   2023-08-04 12:30:00 +0000
categories: langchain openai jekyll hyde prompt-template llm-chain sequential-chain profanity
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/blob/published/LLM%2003%20-%20Multi-stage%20Reasoning*

### Examples

```shell
pip install better-profanity==0.7.0
pip install langchain==0.0.229
pip install openai==0.27.8
```

> API Keys:

```py
import os
os.environ["OPENAI_API_KEY"] = "<FILL IN>"
```

- **Jekyll and Hyde**

> Import Libraries:

```py
from langchain import PromptTemplate
import numpy as np
```

> Jekyll Prompt Template:

```py
jekyll_template = """
You are a social media post commenter, you will respond to the following post with a {sentiment} response. 
Post:" {social_post}"
Comment: 
"""
jekyll_prompt_template = PromptTemplate(
    input_variables=["sentiment", "social_post"],
    template=jekyll_template,
)
```

> Random sentiment and social post:

```py
random_sentiment = "nice"
if np.random.rand() < 0.3:
    random_sentiment = "mean"
social_post = "I can't believe I'm learning about LangChain in this MOOC, there is so much to learn and so far the instructors have been so helpful. I'm having a lot of fun learning! #AI #Databricks"
```

> Generate prompt from the jekyll template using the random sentiment and social post:

```py
jekyll_prompt = jekyll_prompt_template.format(
    sentiment=random_sentiment, social_post=social_post
)
print(f"Jekyll prompt:{jekyll_prompt}")
```

> Output:

```
Jekyll prompt:
You are a social media post commenter, you will respond to the following post with a nice response. 
Post:" I can't believe I'm learning about LangChain in this MOOC, there is so much to learn and so far the instructors have been so helpful. I'm having a lot of fun learning! #AI #Databricks"
Comment: 
```

> OpenAI LLM:

```py
from langchain.llms import OpenAI
jekyll_llm = OpenAI(model="text-babbage-001")
```

> Jekyll Chain:

```py
from langchain.chains import LLMChain
jekyll_chain = LLMChain(
    llm=jekyll_llm,
    prompt=jekyll_prompt_template,
    output_key="jekyll_said",
    verbose=False,
)
```

> Run Jekyll Chain:

```py
jekyll_said = jekyll_chain.run(
    {"sentiment": random_sentiment, "social_post": social_post}
)
```

> Clean the text:

```py
from better_profanity import profanity
cleaned_jekyll_said = profanity.censor(jekyll_said)
print(f"Jekyll said:{cleaned_jekyll_said}")
```

> Output:

```
Jekyll said:I'm so excited for the LangChain MOOC! I'm learning a lot about blockchain technology and it's so fascinating! The instructors have been so helpful so far and I'm looking forward to learning more.
```

> Hyde Prompt Template:

```py
hyde_template = """
You are Hyde, the moderator of an online forum, you are strict and will not tolerate any negative comments. You will look at this next comment from a user and, if it is at all negative, you will replace it with symbols and post that, but if it seems nice, you will let it remain as is and repeat it word for word.
Original comment: {jekyll_said}
Edited comment:
"""

hyde_prompt_template = PromptTemplate(
    input_variables=["jekyll_said"],
    template=hyde_template,
)
```

> Hyde LLM:

```py
# hyde_llm = jekyll_llm
hyde_llm = OpenAI(model="text-davinci-003")
```

> Hyde Chain:

```py
hyde_chain = LLMChain(
    llm=hyde_llm, prompt=hyde_prompt_template, verbose=False
)
```

> Run Hyde Chain:

```py
hyde_says = hyde_chain.run({"jekyll_said": jekyll_said})
print(f"Hyde says: {hyde_says}")
```

> Output:

```
Hyde says: I'm so excited for the LangChain MOOC! I'm learning a lot about blockchain technology and it's so fascinating! The instructors have been so helpful so far and I'm looking forward to learning more.
```

> Sequential Chain:

```py
from langchain.chains import SequentialChain

jekyllhyde_chain = SequentialChain(
    chains=[jekyll_chain, hyde_chain],
    input_variables=["sentiment", "social_post"],
    verbose=True,
)
jekyllhyde_chain.run({"sentiment": random_sentiment, "social_post": social_post})
```

> Output:

```
> Entering new  chain...

> Finished chain.
"I'm so excited to be learning about LangChain! The instructors have been so helpful so far and I'm having a lot of fun learning. Thank you for your interest in LangChain!"
```
