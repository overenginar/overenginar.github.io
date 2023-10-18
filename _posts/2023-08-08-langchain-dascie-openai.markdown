---
layout: post
title:  "DaScie Langchain with OpenAI"
date:   2023-08-08 12:30:00 +0000
categories: langchain openai dascie tools agents wikipedia serpapi
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/blob/published/LLM%2003%20-%20Multi-stage%20Reasoning*

### Examples

```shell
pip install langchain==0.0.229
pip install openai==0.27.8
pip install wikipedia==1.4.0
pip install google-search-results==2.4.2 
pip install sqlalchemy==2.0.15
pip install seaborn
```

> API Keys:

```py
import os
os.environ["SERPAPI_API_KEY"] = "<FILL IN>"
os.environ["OPENAI_API_KEY"] = "<FILL IN>"
```

- **Jekyll and Hyde**

> Import Libraries:

```py
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import (
    AgentType,
)
from langchain.llms import OpenAI
```

> Load LLM, tools and agent:

```py
llm = OpenAI()
tools = load_tools(["wikipedia", "serpapi", "python_repl", "terminal"], llm=llm)
dascie = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```

> Run the agent - analyze data:

```py
dascie.run(
    "Create a dataset (DO NOT try to download one, you MUST create one based on what you find) on the performance of the Mercedes AMG F1 team in 2020 and do some analysis. You need to plot your results."
)
```

> Run the agent - analyze data with seaborn:

```py
dascie.run(
    "Create a detailed dataset (DO NOT try to download one, you MUST create one based on what you find) on the performance of each driver in the Mercedes AMG F1 team in 2020 and do some analysis with at least 3 plots, use a subplot for each graph so they can be shown at the same time, use seaborn to plot the graphs."
)
```

>

> Create dataset:

```py
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
datasci_data_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
datasci_data_df['target'] = diabetes.target

dascie = create_pandas_dataframe_agent(
    OpenAI(temperature=0), datasci_data_df, verbose=True
)
```

> Analyze the dataset:

```py
dascie.run("Analyze this data, tell me any interesting trends. Make some pretty plots.")
```

> Build a model:

```py
dascie.run(
    "Train a random forest regressor to predict target using the most important features. Show me the what variables are most influential to this model"
)
```