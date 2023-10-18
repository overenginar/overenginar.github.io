---
layout: post
title:  "Langchain RetrievalQA with reviews"
date:   2023-08-09 12:30:00 +0000
categories: langchain retrievalqa huggingface chroma vector-store
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/blob/published/LLM%2003%20-%20Multi-stage%20Reasoning*

### Examples

```shell
pip install langchain==0.0.229
pip install transformers==4.31.0
pip install chromadb==0.3.21
pip install tiktoken==0.3.3
pip install sqlalchemy==2.0.15
```

> API Keys:

```py
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<FILL IN>"
```

> Load document

```py
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

laptop_reviews = TextLoader(
    "edx/reviews/fake_laptop_reviews.txt", encoding="utf8"
)
document = laptop_reviews.load()
document
```

> Split the document with langchain splitter:

```py
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
texts = text_splitter.split_documents(document)
```

> Create embeddings using HuggingFace embeddings:

```py
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name
)
```

> Create vector index:

```py
from langchain.vectorstores import Chroma
chromadb_index = Chroma.from_documents(
    texts, embeddings, persist_directory='./chroma-store'
)
```

> RetrievalQA:

```py
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

retriever = chromadb_index.as_retriever()

hf_llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={
        "temperature": 0,
        "max_length": 128,
    },
)

chain_type = "stuff"
laptop_qa = RetrievalQA.from_chain_type(
    llm=hf_llm, chain_type=chain_type, retriever=retriever
)
```

> Question1:

```py
laptop_name = laptop_qa.run("What is the full name of the laptop?")
laptop_name
```

> Output

```
Raytech Supernova
```

> Question2:

```py
laptop_features = laptop_qa.run("What are some of the laptop's features?")
laptop_features
```

> Output:

```
The 4K display, powerful GPU, and fast SSD
```

> Question3:

```py
laptop_reviews = laptop_qa.run("What is the general sentiment of the reviews?")
laptop_reviews
```

> Output:

```
positive
```
