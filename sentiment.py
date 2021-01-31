import pandas as pd
import numpy as np
import re
import spacy


def process_text(text):
    hashtags = re.compile(r"^#\S+|\s#\S+")
    mentions = re.compile(r"^@\S+|\s@\S+")
    urls = re.compile(r"(https|http)?://\S+")
    # Apostrophe is kept to allow word contractions, for eg: don't
    punctuation = re.compile(r"[^\w\s’]")

    for pattern in [hashtags, mentions, urls, punctuation]:
        text = pattern.sub(' ', text)

    # Fix word lengthening. Most words in english have a maximum of two repeated characters.
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # Convert entire text to lowercase and remove redundant whitespaces.
    cleaned_text = " ".join(text.lower().split())

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(cleaned_text)

    # Lemmatise words from original text that are not stopwords.
    processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop])

    return processed_text
