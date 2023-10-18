---
layout: post
title:  "Langchain RetrievalQA with book"
date:   2023-08-10 12:30:00 +0000
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

> Load the document

```py
from langchain.document_loaders import GutenbergLoader

loader = GutenbergLoader(
    "https://www.gutenberg.org/cache/epub/100/pg100.txt"
)
all_shakespeare_text = loader.load()
```

> Split the document with langchain splitter:

```py
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=256)
texts = text_splitter.split_documents(all_shakespeare_text)
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
docsearch = Chroma.from_documents(
    texts, embeddings, persist_directory='./chroma-store'
)
```

> Huggingface LLM:

```py
hf_llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={
        "temperature": 0,
        "max_length": 1024,
    },
)
```

> Question1:

```py
qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="stuff", retriever=docsearch.as_retriever())
query = "What happens in the play Hamlet?"
query_results_hamlet = qa.run(query)
query_results_hamlet
```

> Output

```
The Poisoner with some three or four Mutes, comes in again, seeming to lament with her. The dead body is carried away. The Poisoner woos the Queen with gifts. She seems loth and unwilling awhile, but in the end accepts his love.
```

> Question2:

```py
qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type="map_reduce", retriever=docsearch.as_retriever())
query = "Who is the main character in the Merchant of Venice?"
query_results_hamlet = qa.run(query)
query_results_hamlet
```

> Output

```
Lucius
```

> Question3:

```py
qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type='refine', retriever=docsearch.as_retriever())
query = "What happens to romeo and juliet?"
query_results_romeo = qa.run(query)
query_results_romeo
```

> Output

```
The Poisoner with some three or four Mutes, comes in again, seeming to lament with her. The dead body is carried away. The Poisoner woos the Queen with gifts. She seems loth and unwilling awhile, but in the end accepts his love._ [_Exeunt._]
```
