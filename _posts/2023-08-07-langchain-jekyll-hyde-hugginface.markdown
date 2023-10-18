---
layout: post
title:  "Jekyll and Hyde Langchain with HuggingFace"
date:   2023-08-07 12:30:00 +0000
categories: langchain huggingface transformers jekyll hyde prompt-template llm-chain sequential-chain profanity
author: Ali Cabukel
---

*Reference: https://github.com/databricks-academy/large-language-models/blob/published/LLM%2003%20-%20Multi-stage%20Reasoning*

### Examples

```shell
pip install better-profanity==0.7.0
pip install langchain==0.0.229
pip install transformers==4.31.0
```

> API Keys:

```py
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<FILL IN>"
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

> HuggingFace LLM:

```py
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
model_id = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
     "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device_map='auto'
)
jekyll_llm = HuggingFacePipeline(pipeline=pipe)
```


> Jekyll Chain:

```py
from langchain.chains import LLMChain
jekyll_chain = LLMChain(
    llm=jekyll_llm,
    prompt=jekyll_prompt_template,
    output_key="tokenized_dataset",
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
Jekyll said:
Post: I don't understand how you can be sure that is the one that will be solved.
Comment: 

Post: Here is the link explaining what the solution actually is and why people are not satisfied.
Comment:  Caldwell et al explain all of this in their paper https://goo.gl/YZ8OtR

When it comes to AI, as the title says, you should not be surprised if you don't know everything and the professors are not your personal assistants. But that said, you seem to be good at posting. 
How did you learn to comment? What can you do to improve the quality of your comments? 

A:

There are many ways to acquire the skill of commenting. Some of them are not easy to learn and require practice and hard work, others are very simple to learn and a couple of them are free. Each one offers a different perspective. In this case, all we can do is to try and evaluate which one fits best to you depending on your learning style, time and interest. 
I would suggest to follow @benniesh answer. For my other part, I would suggest to read this excellent answer from @lunastep. It's a very nice and useful answer about how to make comments on SO 

A:

I used to comment on any question or answer I liked, and I think I did pretty well. Now I post on programming questions, and I rarely comment on a question.
If programming questions are more of a passion of mine than a career, why do I post on them? Because you can really learn a lot from posting on these questions, it can make you a better programmer.
The amount of feedback on the program, and the helpful advice you receive from other users are probably why I stay active on SO. I don't comment to the best of my ability. I usually just say "thanks" and then move on to the next answer.

A:

I used to comment on any questions I liked, and I think I did pretty well. Now I post on programming questions, and I rarely comment on a question.

There's good reason to do both.  The OP has the option of posting something (even a link to an article) that is both a "good" answer to their own question and one they've found helpful elsewhere.  You, on the other hand, have the option of posting something that is a
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
hyde_llm = jekyll_llm
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
Hyde says: 
I'm really struggling with understanding your comment, and I don't understand why you seem to think that a comment is always on your side.  I'm not making any judgments about an answer you don't like; that's just not my role in this forum.  I'm talking about how an answer is perceived by others.  By your own comment, you seem to believe that it can be either or.  
In the case of the question linked in your comment, I'm still pretty much neutral Epicurus:  the OP is asking and answering the question.  The answer (as it stands) is neither positive nor negative in any way, by your own actions nationals.  I've no desire to be the only source of negative feedback on the site, and I've no inclination to be the first source to comment negatively on a question.  
That's just a comment by another user, not necessarily in any way my "personal opinion" :) 
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
'"So glad I found this MOOC I\'ve heard of Databricks and now so much more. It\'s very exciting to start and I can\'t wait to dive in."\nPost modified to "Thank you so much for taking the time to review and comment on this post. Your comment is appreciated and helps us improve the post. If you have any questions or comments feel free to post a new comment to get help. Best wishes! \nGavin  campaigns@databricks.com"\nComment edited to "Thank you for your review and for the helpful comment. The post has been improved and I\'ve responded as necessary. Please check out LangChain again! \n"\nScore: 6 (6)\nYou have received no comments with score of 0 (0)\nOriginal comment\n" This is my first attempt at any kind of coding and I have to say I am just in awe of this course. It does not get any easier! I am really impressed with the community and the staff." \nYou haven\'t commented for 2 days\n\nI hope this is helpful, or at least helps you get started.\nI will post this response once I get it. \nThanks.\n\nA:\n\nYour score was 0.\nThere was one comment in this thread with score of 0. So you haven\'t gotten any comments.\n\n'
```
