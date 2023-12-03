# Depression Detection System

This research involves building a model to predict the mental state of individuals
based on their choice of words when interacting on social media.

## Folder Structure

- **datasets:** initial datasets.
- **palm2_datasets:** chosen dataset in PaLM2 format.
- **training:** contains notebook and assets used in model training, evaluation
  and testing.
- **app:** contains code used for deployment.

## Context

### Researching

We started to explore research papers that talked about the negative effects mental
health has on college students. We explored different factors which including
family, friends, a sense of self, grades and relationships. Exploring each of these
factors will be great to tell if a student is depressed or not.

### The Solution

We noticed that young adults spend a lot of their time on social media.
As social beings, one of the primary ways we express ourselves is through words.
We set out to find a dataset that dealt with something similar.

- _Dataset 1:_ Scraped data from Facebook comments and posts. (focused on
  college students)
- _Dataset 2:_ From Reddit posts. (a general scope)

We eventually picked up _Dataset 2_ after exploring both datasets. Detailed
reasons are provided in the jupyter notebook.

### Scaling

The reddit dataset allowed us to cover depression detection for more than just
college students, but individuals on social media (which is almost everyone who
has internet access).

### Fine-tuned Model Vs Building From Scratch

#### Initial plan

We initially planned to use Natural Language Processing (NLP) libraries along with
other classification models in a Deep Neural Network. The caveat is that, our model
might work for a handful of individuals, but not be fairly representative to be
production ready given the sensitive nature of our topic.

#### The Pivot

We pivoted to use pre-trained models and fine-tune them with our dataset. That
way, we do not only rely on our personal research, but also leverage the work
of experts around the world.

We fine-tuned the **PaLM2 text-bison model** with our dataset to get a new Large
Language Model that can detect if an individuals is going through depression with
a minimum confidence level of 95%.

## Dataset Description

| Label        | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| clean_text   | Scraped text from reddit                                     |
| is_depressed | 0 or 1. Inidicates whether the text suggests on is depressed |

## Challenges

- **Challenge:** Google Cloud had internal errors during AutoML evaluation that require
  Google Support Team.
- **Workaround:** Used api endpoint of model to query for predicted results and used
  scikit accuracy metrics to evaluate the model.
