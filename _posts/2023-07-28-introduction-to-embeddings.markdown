---
layout: post
title:  "Magic of Embeddings: Decoding Their Power and Applications"
date:   2023-07-28 12:30:00 +0000
categories: embedding openai vertexai nlp gensim
author: Ali Cabukel
---

*Content was generated by ChatGPT*

### Introduction

In the realm of natural language processing and machine learning, embeddings have emerged as a remarkable advancement that has revolutionized the way computers understand and process human language. These compact numerical representations capture the semantic relationships between words and concepts, unlocking a treasure trove of possibilities for a wide array of applications. In this blog, we'll delve into the fascinating world of embeddings, exploring their inner workings, benefits, and diverse applications across various domains.

### What are Embeddings?

At its core, an embedding is a mathematical representation of a word, phrase, or even an entire document within a multi-dimensional vector space. This concept enables words with similar meanings or contextual relevance to be closer together in this space, allowing machines to grasp semantic relationships and context in a manner more aligned with human understanding.

### The Magic of Word Embeddings

Word embeddings are among the most well-known and extensively used types of embeddings. Techniques like Word2Vec, GloVe (Global Vectors for Word Representation), and FastText have gained immense popularity for their ability to map words into continuous vector spaces. This empowers machines to comprehend linguistic nuances, such as word analogies and context, by calculating vector distances and similarities.

### Applications Across the Board

- **Natural Language Understanding (NLU):** Embeddings play a pivotal role in various NLU tasks, including sentiment analysis, named entity recognition, and text classification. By encoding words into meaningful vectors, machines can identify sentiments, entities, and categories more accurately.

- **Machine Translation:** Embeddings have transformed machine translation by enabling models to decipher the context and nuances of words in different languages. This has led to improved translation accuracy and fluency.

- **Information Retrieval:** In search engines and recommendation systems, embeddings assist in retrieving relevant documents or products by capturing semantic similarities between queries and indexed items.

- **Speech Processing:** Embeddings aren't limited to written text; they also find application in speech processing. By transforming spoken words into embeddings, machines can analyze and compare audio data, aiding in speech recognition and speaker identification.

- **Image Analysis:** Cross-modal embeddings facilitate the fusion of text and image data. This has applications in image captioning, where the model generates textual descriptions of images, and in visual question answering, where models answer questions about images.

- **Recommendation Systems:** Embeddings enable recommendation systems to comprehend user preferences and item attributes more effectively, resulting in more accurate and personalized recommendations.

### Challenges and Considerations

While embeddings offer an array of benefits, they are not without challenges. One key challenge is handling out-of-vocabulary (OOV) words or rare terms that the model hasn't encountered during training. Techniques such as subword embeddings and character-level embeddings mitigate this issue.

Additionally, selecting the appropriate embedding dimensionality and training data size requires careful consideration. An overly high dimensionality can lead to overfitting, while a low dimensionality might not capture the complexity of the data.

### Conclusion

Embeddings have sparked a paradigm shift in natural language processing and machine learning. Their ability to capture intricate semantic relationships within a compact numerical representation has opened doors to a wide range of applications across industries. As technology continues to advance, we can only anticipate more innovative uses for embeddings, propelling us further toward human-like language understanding and interaction. So the next time you marvel at a chatbot's eloquence or a recommendation system's accuracy, remember that beneath it all, embeddings are at play, orchestrating a symphony of meaning in the realm of zeros and ones.

### Examples

- **gensim**

> Install package:

```sh
pip install gensim
```

> Python code:

```py
import json
import gensim.downloader as api
data_list = api.info()
# print(json.dumps(data_list, indent=4))
print('Corpora: ', data_list['corpora'].keys())
print()
print('Models: ', data_list['models'].keys())
```

> Output:

```
Corpora:  dict_keys(['semeval-2016-2017-task3-subtaskBC', 'semeval-2016-2017-task3-subtaskA-unannotated', 'patent-2017', 'quora-duplicate-questions', 'wiki-english-20171001', 'text8', 'fake-news', '20-newsgroups', '__testing_matrix-synopsis', '__testing_multipart-matrix-synopsis'])

Models:  dict_keys(['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', '__testing_word2vec-matrix-synopsis'])
```

> Python code:

```py
path = api.load("word2vec-google-news-300", return_path=True)
print(path)
```

> Output:

```
/home/jupyter/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz
```

> Python code:

```py
model = api.load("word2vec-google-news-300")
```

> Python code:

```py
model.most_similar('tree')
```

> Output:

```
[('trees', 0.8293122053146362),
 ('pine_tree', 0.7622087001800537),
 ('oak_tree', 0.731893002986908),
 ('evergreen_tree', 0.6926872730255127),
 ('fir_tree', 0.6917218565940857),
 ('willow_tree', 0.6845874190330505),
 ('pine_trees', 0.6824266910552979),
 ('maple_tree', 0.6803498268127441),
 ('sycamore_tree', 0.6681810617446899),
 ('tress', 0.6547872424125671)]
```

> Python code:

```py
model.most_similar('glass')
```

> Output:

```
[('R._Mazzei_fused', 0.6665399670600891),
 ('Christian_Audigier_nightclub', 0.6632695198059082),
 ('copper_alloy_garnets', 0.6343654990196228),
 ('Nelmeus', 0.6274422407150269),
 ('fiber_fusion_splicing', 0.6229819655418396),
 ('Plexiglass', 0.5858588814735413),
 ('slashing_Leonardo_DiCaprio', 0.5850011110305786),
 ('plexiglass', 0.5823022723197937),
 ('Plexiglas', 0.5803930759429932),
 ("#Q'##_unaudited", 0.5798528790473938)]
```

> Python code:

```py
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between word embeddings
word1 = "apple"
word2 = "orange"
word3 = "table"

# Check if both words are present in the vocabulary
if word1 in model.key_to_index and word2 in model.key_to_index and word3 in model.key_to_index:
    embedding1 = model.get_vector(word1).reshape(1, -1)
    embedding2 = model.get_vector(word2).reshape(1, -1)
    embedding3 = model.get_vector(word3).reshape(1, -1)

    similarity_score_1vs2 = cosine_similarity(embedding1, embedding2)[0][0]
    similarity_score_1vs3 = cosine_similarity(embedding1, embedding3)[0][0]
    similarity_score_2vs3 = cosine_similarity(embedding2, embedding3)[0][0]
    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity_score_1vs2:.4f}")
    print(f"Cosine similarity between '{word1}' and '{word3}': {similarity_score_1vs3:.4f}")
    print(f"Cosine similarity between '{word2}' and '{word3}': {similarity_score_2vs3:.4f}")
else:
    print("One or both words not found in the vocabulary.")
```

> Output:

```
Cosine similarity between 'apple' and 'orange': 0.3920
Cosine similarity between 'apple' and 'table': 0.0768
Cosine similarity between 'orange' and 'table': 0.0402
```

- **openai**

*Reference: https://github.com/Azure/azure-openai-samples/blob/main/fundamentals/document_analysis/notebooks/01-get-embeddings.ipynb*

*Reference: https://platform.openai.com/docs/guides/embeddings/what-are-embeddings*

> Install package:

```sh
pip install openai
```

> Python code:

```py
import os
import openai
os.environ['OPENAI_API_KEY'] = 'YOUR API KEY'
```

> Python code:

```py
embedding1 = openai.Embedding.create(
    input="I love playing soccer",
    deployment_id="text-embedding-ada-002"
)

len(embedding1["data"][0]["embedding"])
```

> Output:

```
1536
```

> Python code:

```py
embedding2 = openai.Embedding.create(
    input="I enjoy playing football",
    deployment_id="text-embedding-ada-002"
)

len(embedding2["data"][0]["embedding"])
```

> Output:

```
1536
```

> Python code:

```py
embedding3 = openai.Embedding.create(
    input="Run run as fast as you can",
    deployment_id="text-embedding-ada-002"
)

len(embedding3["data"][0]["embedding"])
```

> Output:

```
1536
```

> Python code:

```py
from scipy.spatial import distance
print('Euclidean sentence1vs2: ', distance.euclidean(embedding1["data"][0]["embedding"], embedding2["data"][0]["embedding"]))
print('Euclidean sentence1vs3: ', distance.euclidean(embedding1["data"][0]["embedding"], embedding3["data"][0]["embedding"]))
print('Euclidean sentence2vs3: ', distance.euclidean(embedding2["data"][0]["embedding"], embedding3["data"][0]["embedding"]))
```

> Output:

```
Euclidean sentence1vs2:  0.38906803470461504
Euclidean sentence1vs3:  0.6835550521490075
Euclidean sentence2vs3:  0.6879057796857235
```

> Python code:

```py
from scipy.spatial import distance
print('Cosine sentence1vs2: ', distance.cosine(embedding1["data"][0]["embedding"], embedding2["data"][0]["embedding"]))
print('Cosine sentence1vs3: ', distance.cosine(embedding1["data"][0]["embedding"], embedding3["data"][0]["embedding"]))
print('Cosine sentence2vs3: ', distance.cosine(embedding2["data"][0]["embedding"], embedding3["data"][0]["embedding"]))
```

> Output:

```
Cosine sentence1vs2:  0.07568696871399849
Cosine sentence1vs3:  0.23362376452580347
Cosine sentence2vs3:  0.23660718341319753
```

- **vertexai**

*Reference: https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/examples/langchain-intro/intro_langchain_palm_api.ipynb*

*Reference: https://python.langchain.com/docs/integrations/text_embedding/google_vertex_ai_palm*

> Install Package:

```sh
pip install langchain
pip install google-cloud-aiplatform
```

> Python code:

```py
from langchain.embeddings import VertexAIEmbeddings

embeddings = VertexAIEmbeddings()

query_result1 = embeddings.embed_query("I love playing soccer")
query_result2 = embeddings.embed_query("I enjoy playing football")
query_result3 = embeddings.embed_query("Run run as fast as you can")

len(query_result1), len(query_result2), len(query_result3)
```

> Output:

```
(768, 768, 768)
```

> Python code:

```py
from scipy.spatial import distance
print('Cosine sentence1vs2: ', distance.cosine(query_result1, query_result2))
print('Cosine sentence1vs3: ', distance.cosine(query_result1, query_result3))
print('Cosine sentence2vs3: ', distance.cosine(query_result2, query_result3))
```

> Output:

```
Cosine sentence1vs2:  0.2334968652052557
Cosine sentence1vs3:  0.4390746238229084
Cosine sentence2vs3:  0.4524773239411106
```
