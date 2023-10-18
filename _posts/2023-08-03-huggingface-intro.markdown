---
layout: post
title:  "Introduction to Huggingface by examples"
date:   2023-08-03 12:30:00 +0000
categories: huggingface transformers tokenizers datasets summarizer sentiment xsum translation text2text-generation t5 zero-shot few-shot auto-tokenizer auto-model-for-seq2seq t5-tokenizer t5-for-conditional-generation
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/tree/published/LLM%2001%20-%20Applications%20with%20LLMs*

### Examples

```shell
pip install transformers==4.31.0 --user
pip install datasets==2.14.4 --user
pip install sacremoses==0.0.53 --user
```

- **Summarizer**

> Import libraries:

```py
from datasets import load_dataset
from transformers import pipeline
```

> Load dataset:

```py
xsum_dataset = load_dataset(
    "xsum", version="1.2.0"
)
xsum_dataset 
```

> Output:

```
DatasetDict({
    train: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 204045
    })
    validation: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 11332
    })
    test: Dataset({
        features: ['document', 'summary', 'id'],
        num_rows: 11334
    })
})
```

> Sampling:

```py
xsum_sample = xsum_dataset["train"].select(range(10))
xsum_sample.to_pandas()
```

> Summarizer pipeline:

```py
summarizer = pipeline(
    task="summarization",
    model="t5-small",
    min_length=20,
    max_length=40,
    truncation=True
) 
```

> Summarize the documents:

```py
summarizer(xsum_sample["document"][0])
```

> Output

```
[{'summary_text': 'the full cost of damage in Newton Stewart is still being assessed . many roads in peeblesshire remain badly affected by standing water . a flood alert remains in place across the'}]
```

> Transform into pandas data frame:

```py
import pandas as pd

results = summarizer(xsum_sample["document"])
(
    pd.DataFrame.from_dict(results)
    .rename({"summary_text": "generated_summary"}, axis=1)
    .join(pd.DataFrame.from_dict(xsum_sample))[
        ["generated_summary", "summary", "document"]
    ]
)
```

- **Sentiment Analysis**

> Load dataset:

```py
poem_dataset = load_dataset(
    "poem_sentiment", version="1.0.0"
)
poem_sample = poem_dataset["train"].select(range(10))
poem_sample.to_pandas()
```

> Sentiment classifier pipeline:

```py
sentiment_classifier = pipeline(
    task="text-classification",
    model="nickwong64/bert-base-uncased-poems-sentiment",
)
```

> Classify the texts:

```py
results = sentiment_classifier(poem_sample["verse_text"])
```

> Join predictions and actuals:

```py
joined_data = (
    pd.DataFrame.from_dict(results)
    .rename({"label": "predicted_label"}, axis=1)
    .join(pd.DataFrame.from_dict(poem_sample).rename({"label": "true_label"}, axis=1))
)

sentiment_labels = {0: "negative", 1: "positive", 2: "no_impact", 3: "mixed"}
joined_data = joined_data.replace({"true_label": sentiment_labels})

joined_data[["predicted_label", "true_label", "score", "verse_text"]]

```

- **Translation**

> Translation pipeline

```py
en_to_es_translation_pipeline = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-es",
)
```

> Translate the text:

```py
en_to_es_translation_pipeline(
    "Existing, open-source (and proprietary) models can be used out-of-the-box for many applications."
)
```

> Output

```
[{'translation_text': 'Los modelos existentes, de código abierto (y propietario) se pueden utilizar fuera de la caja para muchas aplicaciones.'}]
```

- **Text2Text Generation**

> Text2Text Generation pipeline:

```py
t5_small_pipeline = pipeline(
    task="text2text-generation",
    model="t5-small",
    max_length=50,
)
```

> Generate texts - translate en to fr:

```py
t5_small_pipeline(
    "translate English to French: Existing, open-source (and proprietary) models can be used out-of-the-box for many applications."
)
```

> Output

```
[{'generated_text': 'Les modèles existants, libres (et propriétaires) peuvent être utilisés hors de la boîte de commande pour de nombreuses applications.'}]
```

> Generate texts - translate en to ro:

```py
t5_small_pipeline(
    "translate English to Romanian: Existing, open-source (and proprietary) models can be used out-of-the-box for many applications."
)
```

> Output

```
[{'generated_text': 'Modelele existente, deschise (şi proprietăţi) pot fi utilizate în afara legii pentru multe aplicaţii.'}]
```

- **Zero-shot Pipeline**

> Zero-shot pipeline:

```py
zero_shot_pipeline = pipeline(
    task="zero-shot-classification",
    model="cross-encoder/nli-deberta-v3-small",
)
```

> Categorize article function:

```py
import pandas as pd
def categorize_article(article: str) -> None:
    """
    This helper function defines the categories (labels) which the model must use to label articles.
    Note that our model was NOT fine-tuned to use these specific labels,
    but it "knows" what the labels mean from its more general training.

    This function then prints out the predicted labels alongside their confidence scores.
    """
    results = zero_shot_pipeline(
        article,
        candidate_labels=[
            "politics",
            "finance",
            "sports",
            "science and technology",
            "pop culture",
            "breaking news",
        ],
    )
    del results["sequence"]
    return pd.DataFrame(results)
```

> Categorize the given article:

```py
categorize_article(
    """
Simone Favaro got the crucial try with the last move of the game, following earlier touchdowns by Chris Fusaro, Zander Fagerson and Junior Bulumakau.
Rynard Landman and Ashton Hewitt got a try in either half for the Dragons.
Glasgow showed far superior strength in depth as they took control of a messy match in the second period.
Home coach Gregor Townsend gave a debut to powerhouse Fijian-born Wallaby wing Taqele Naiyaravoro, and centre Alex Dunbar returned from long-term injury, while the Dragons gave first starts of the season to wing Aled Brew and hooker Elliot Dee.
Glasgow lost hooker Pat McArthur to an early shoulder injury but took advantage of their first pressure when Rory Clegg slotted over a penalty on 12 minutes.
It took 24 minutes for a disjointed game to produce a try as Sarel Pretorius sniped from close range and Landman forced his way over for Jason Tovey to convert - although it was the lock's last contribution as he departed with a chest injury shortly afterwards.
Glasgow struck back when Fusaro drove over from a rolling maul on 35 minutes for Clegg to convert.
But the Dragons levelled at 10-10 before half-time when Naiyaravoro was yellow-carded for an aerial tackle on Brew and Tovey slotted the easy goal.
The visitors could not make the most of their one-man advantage after the break as their error count cost them dearly.
It was Glasgow's bench experience that showed when Mike Blair's break led to a short-range score from teenage prop Fagerson, converted by Clegg.
Debutant Favaro was the second home player to be sin-binned, on 63 minutes, but again the Warriors made light of it as replacement wing Bulumakau, a recruit from the Army, pounced to deftly hack through a bouncing ball for an opportunist try.
The Dragons got back within striking range with some excellent combined handling putting Hewitt over unopposed after 72 minutes.
However, Favaro became sinner-turned-saint as he got on the end of another effective rolling maul to earn his side the extra point with the last move of the game, Clegg converting.
Dragons director of rugby Lyn Jones said: "We're disappointed to have lost but our performance was a lot better [than against Leinster] and the game could have gone either way.
"Unfortunately too many errors behind the scrum cost us a great deal, though from where we were a fortnight ago in Dublin our workrate and desire was excellent.
"It was simply error count from individuals behind the scrum that cost us field position, it's not rocket science - they were correct in how they played and we had a few errors, that was the difference."
Glasgow Warriors: Rory Hughes, Taqele Naiyaravoro, Alex Dunbar, Fraser Lyle, Lee Jones, Rory Clegg, Grayson Hart; Alex Allan, Pat MacArthur, Zander Fagerson, Rob Harley (capt), Scott Cummings, Hugh Blake, Chris Fusaro, Adam Ashe.
Replacements: Fergus Scott, Jerry Yanuyanutawa, Mike Cusack, Greg Peterson, Simone Favaro, Mike Blair, Gregor Hunter, Junior Bulumakau.
Dragons: Carl Meyer, Ashton Hewitt, Ross Wardle, Adam Warren, Aled Brew, Jason Tovey, Sarel Pretorius; Boris Stankovich, Elliot Dee, Brok Harris, Nick Crosswell, Rynard Landman (capt), Lewis Evans, Nic Cudd, Ed Jackson.
Replacements: Rhys Buckley, Phil Price, Shaun Knight, Matthew Screech, Ollie Griffiths, Luc Jones, Charlie Davies, Nick Scott.
"""
)
```

> Output

```
    labels	scores
0	sports	0.469011
1	breaking news	0.223165
2	science and technology	0.107025
3	pop culture	0.104471
4	politics	0.057390
5	finance	0.038938
```

> Categorize the given article:

```py
categorize_article(
    """
The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.
Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.
Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct.
Many businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town.
First Minister Nicola Sturgeon visited the area to inspect the damage.
The waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare.
Jeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit.
However, she said more preventative work could have been carried out to ensure the retaining wall did not fail.
"It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we're neglected or forgotten," she said.
"That may not be true but it is perhaps my perspective over the last few days.
"Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?"
Meanwhile, a flood alert remains in place across the Borders because of the constant rain.
Peebles was badly hit by problems, sparking calls to introduce more defences in the area.
Scottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs.
The Labour Party's deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand.
He said it was important to get the flood protection plan right but backed calls to speed up the process.
"I was quite taken aback by the amount of damage that has been done," he said.
"Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses."
He said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans.
Have you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.
"""
)
```

> Output

```
    labels	scores
0	breaking news	0.208211
1	politics	0.173790
2	pop culture	0.173753
3	science and technology	0.157181
4	sports	0.154562
5	finance	0.132503
```

- **Few-shot Pipeline**

> Few shot pipeline:

```py
few_shot_pipeline = pipeline(
    task="text-generation",
    model="EleutherAI/gpt-neo-1.3B",
    max_new_tokens=10,
)
```

> EOS token id:

```py
eos_token_id = few_shot_pipeline.tokenizer.encode("###")[0]
```

> Prediction with no example, sentiment:

```py
results = few_shot_pipeline(
    """For each tweet, describe its sentiment:

[Tweet]: "This new music video was incredible"
[Sentiment]:""",
    eos_token_id=eos_token_id,
)
print(results[0]["generated_text"])
```

> Output

```
For each tweet, describe its sentiment:

[Tweet]: "This new music video was incredible"
[Sentiment]: "I think it's going to be really nice
```

> Prediction with one examples, sentiment:

```PY
results = few_shot_pipeline(
    """For each tweet, describe its sentiment:

[Tweet]: "This is the link to the article"
[Sentiment]: Neutral
###
[Tweet]: "This new music video was incredible"
[Sentiment]:""",
    eos_token_id=eos_token_id,
)
print(results[0]["generated_text"])
```

> Output

```
For each tweet, describe its sentiment:

[Tweet]: "This is the link to the article"
[Sentiment]: Neutral
###
[Tweet]: "This new music video was incredible"
[Sentiment]: Neutral
[Sentiment]: Positive
###
```

> Prediction with three examples, sentiment:

```py
results = few_shot_pipeline(
    """For each tweet, describe its sentiment:

[Tweet]: "I hate it when my phone battery dies."
[Sentiment]: Negative
###
[Tweet]: "My day has been 👍"
[Sentiment]: Positive
###
[Tweet]: "This is the link to the article"
[Sentiment]: Neutral
###
[Tweet]: "This new music video was incredible"
[Sentiment]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])
```

> Output

```
For each tweet, describe its sentiment:

[Tweet]: "I hate it when my phone battery dies."
[Sentiment]: Negative
###
[Tweet]: "My day has been 👍"
[Sentiment]: Positive
###
[Tweet]: "This is the link to the article"
[Sentiment]: Neutral
###
[Tweet]: "This new music video was incredible"
[Sentiment]: Positive
###
```

> Prediction with three examples, food-drink pairing:

```py
results = few_shot_pipeline(
    """For each food, suggest a good drink pairing:

[food]: tapas
[drink]: wine
###
[food]: pizza
[drink]: soda
###
[food]: jalapenos poppers
[drink]: beer
###
[food]: scone
[drink]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])
```

> Output

```
For each food, suggest a good drink pairing:

[food]: tapas
[drink]: wine
###
[food]: pizza
[drink]: soda
###
[food]: jalapenos poppers
[drink]: beer
###
[food]: scone
[drink]: beer

#
[food]: cheeseb
```

> Prediction with three examples, feeling:

```py
results = few_shot_pipeline(
    """Given a word describing how someone is feeling, suggest a description of that person.  The description should not include the original word.

[word]: happy
[description]: smiling, laughing, clapping
###
[word]: nervous
[description]: glancing around quickly, sweating, fidgeting
###
[word]: sleepy
[description]: heavy-lidded, slumping, rubbing eyes
###
[word]: confused
[description]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])
```

> Output

```
Given a word describing how someone is feeling, suggest a description of that person.  The description should not include the original word.

[word]: happy
[description]: smiling, laughing, clapping
###
[word]: nervous
[description]: glancing around quickly, sweating, fidgeting
###
[word]: sleepy
[description]: heavy-lidded, slumping, rubbing eyes
###
[word]: confused
[description]: confused, muddled
###
```

> Prediction with three examples, book summary:

```py
results = few_shot_pipeline(
    """Generate a book summary from the title:

[book title]: "Stranger in a Strange Land"
[book description]: "This novel tells the story of Valentine Michael Smith, a human who comes to Earth in early adulthood after being born on the planet Mars and raised by Martians, and explores his interaction with and eventual transformation of Terran culture."
###
[book title]: "The Adventures of Tom Sawyer"
[book description]: "This novel is about a boy growing up along the Mississippi River. It is set in the 1840s in the town of St. Petersburg, which is based on Hannibal, Missouri, where Twain lived as a boy. In the novel, Tom Sawyer has several adventures, often with his friend Huckleberry Finn."
###
[book title]: "Dune"
[book description]: "This novel is set in the distant future amidst a feudal interstellar society in which various noble houses control planetary fiefs. It tells the story of young Paul Atreides, whose family accepts the stewardship of the planet Arrakis. While the planet is an inhospitable and sparsely populated desert wasteland, it is the only source of melange, or spice, a drug that extends life and enhances mental abilities.  The story explores the multilayered interactions of politics, religion, ecology, technology, and human emotion, as the factions of the empire confront each other in a struggle for the control of Arrakis and its spice."
###
[book title]: "Blue Mars"
[book description]:""",
    eos_token_id=eos_token_id,
    max_new_tokens=50,
)

print(results[0]["generated_text"])
```

> Output

```
Generate a book summary from the title:

[book title]: "Stranger in a Strange Land"
[book description]: "This novel tells the story of Valentine Michael Smith, a human who comes to Earth in early adulthood after being born on the planet Mars and raised by Martians, and explores his interaction with and eventual transformation of Terran culture."
###
[book title]: "The Adventures of Tom Sawyer"
[book description]: "This novel is about a boy growing up along the Mississippi River. It is set in the 1840s in the town of St. Petersburg, which is based on Hannibal, Missouri, where Twain lived as a boy. In the novel, Tom Sawyer has several adventures, often with his friend Huckleberry Finn."
###
[book title]: "Dune"
[book description]: "This novel is set in the distant future amidst a feudal interstellar society in which various noble houses control planetary fiefs. It tells the story of young Paul Atreides, whose family accepts the stewardship of the planet Arrakis. While the planet is an inhospitable and sparsely populated desert wasteland, it is the only source of melange, or spice, a drug that extends life and enhances mental abilities.  The story explores the multilayered interactions of politics, religion, ecology, technology, and human emotion, as the factions of the empire confront each other in a struggle for the control of Arrakis and its spice."
###
[book title]: "Blue Mars"
[book description]: "Set in Australia during the 1950s, this novel depicts the first contact between humans and Australia 'blue men' who look like blue women. The novel tells the story of their arrival and their attempt to establish a culture and friendship with an Australian country
```

- **Summarizer Pipeline with options**

> Dataset:

```py
xsum_sample.to_pandas()
```

> Summarize the first document:

```py
summarizer(xsum_sample["document"][0])
```

> Output

```
[{'summary_text': 'the full cost of damage in Newton Stewart is still being assessed . many roads in peeblesshire remain badly affected by standing water . a flood alert remains in place across the'}]
```

> Summarize the first document with num_beams:

```py
summarizer(xsum_sample["document"][0], num_beams=10)
```

> Output

```
[{'summary_text': 'the full cost of damage in Newton Stewart is still being assessed . many roads in peeblesshire remain badly affected by standing water . a flood alert remains in place across the'}]
```

> Summarize the first document with do_sample:

```py
summarizer(xsum_sample["document"][0], do_sample=True)
```

> Output

```
[{'summary_text': 'the full cost of damage in Newton Stewart is still being assessed . many roads in peeblesshire remain badly affected by standing water . a flood alert remains in place across the'}]
```

> Summarize the first document with do_sample, top_k and top_p:

```py
summarizer(xsum_sample["document"][0], do_sample=True, top_k=10, top_p=0.8)
```

> Output


```
[{'summary_text': 'repairs are ongoing in Hawick and many roads in peeblesshire remain badly affected . many businesses and householders were affected by flooding in the town . the water breached'}]
```

- **AutoModel**

> AutoTokenizer and AutoModelForSeq2SeqLM:

```py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
```

> Transform articles from xsum dataset:

```py
articles = list(map(lambda article: "summarize: " + article, xsum_sample["document"]))
pd.DataFrame(articles, columns=["prompts"])
```

> Prepare input with tokenizer:

```py
inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)
print("input_ids:")
print(inputs["input_ids"])
print("attention_mask:")
print(inputs["attention_mask"])
```

> Predictions:

```py
summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    num_beams=2,
    min_length=0,
    max_length=40,
)
print(summary_ids)
```

> Decode the predictions:

```py
decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
pd.DataFrame(decoded_summaries, columns=["decoded_summaries"])
```

- **T5Model**

> T5Tokenizer and T5ForConditionalGeneration

```py
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained(
    "t5-small"
)
```

> Tokenize the articles:

```py
inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)
```

> Predictions:

```py
summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    num_beams=2,
    min_length=0,
    max_length=40,
)
```

> Decode the predictions:

```py
decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

pd.DataFrame(decoded_summaries, columns=["decoded_summaries"])
```

- **Translation Eng2Jpn**

> Load dataset and sampling:

```py
jpn_dataset = load_dataset(
    "Helsinki-NLP/tatoeba_mt",
    version="1.0.0",
    language_pair="eng-jpn",
)
jpn_sample = (
    jpn_dataset["test"]
    .select(range(10))
    .rename_column("sourceString", "English")
    .rename_column("targetString", "Japanese")
    .remove_columns(["sourceLang", "targetlang"])
)
jpn_sample.to_pandas()
```

> Translation pipeline:

```py
translation_pipeline = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-jap-en",
)
```

> Predictions:

```py
translation_results = translation_pipeline(jpn_sample["Japanese"])
```

> Transform predictions into data frame:

```py
translation_results_df = pd.DataFrame.from_dict(translation_results).join(
    jpn_sample.to_pandas()
)
translation_results_df
```
