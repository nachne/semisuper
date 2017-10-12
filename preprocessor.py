import pandas as pd
import numpy as np
import urllib.request
import multiprocessing as multi
import pickle
from replacer import HypernymMapper, load_hypernyms
import loader
from helpers import take, flatten
import timeit

import string
import re
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk import TreebankWordTokenizer, WhitespaceTokenizer, RegexpTokenizer, wordpunct_tokenize
from nltk import sent_tokenize
from nltk import pos_tag


class BasicPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, punct=None, lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.punct = punct or set(string.punctuation)
        self.mapper = HypernymMapper()
        self.tokenizer = TreebankWordTokenizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [", ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(sentence)) for sentence in X
        ]

    def tokenize(self, sentence):
        # Break the sentence into part of speech tagged tokens

        for token, tag in pos_tag(self.tokenizer.tokenize(sentence)):
            # Apply preprocessing to the token
            # token = self.mapper.replace(token)
            token = re_replace(token)
            token = token.lower() if self.lower else token
            token = token.strip() if self.strip else token
            token = token.strip('*') if self.strip else token
            token = token.strip('.') if self.strip else token

            # If punctuation, ignore token and continue
            if all(char in self.punct for char in token):
                continue

            yield token

def re_replace(token):
    """replaces abbreviations matching simple REs, e.g. for numbers, percentages, gene names, by class tokens"""

    # ranges, (in)equalities, multiples
    if re.findall("^-?\d*\.?\d+--?\d*\.?\d+$", token):
        return "_range_"
    if re.findall("[a-zA-Z][=<>≤≥]\d", token):
        return "_ineq_"
    if re.findall("^\d+-fold$", token):
        return "_n_fold_"
    if re.findall("^\d+/\d+$|^\d+:\d+$", token):
        return "_ratio_"

    # units
    if re.findall("^\d*((kg|g|mg|ug|ng)|(m|cm|mm|um|nm)|(l|ml|cl|ul|mol|mumol))$", token):
        return ("_unit_")
    # percentages, negative percentages
    if re.findall("^\d*\.?\d*%$", token):
        return "_ratio_"
    if re.findall("^-\d+\.?\d*%$", token):
        return "_ratio_"

    # abbreviations starting with a letter and continues with digits and optional letters
    m = re.findall("^([a-zA-Z])[a-zA-Z]*-?\w*\d+", token)
    if m:
        return "_abbrev_"

    # numbers
    if re.findall("^(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                  "hundred|thousand|first|second|third|1st|2nd|3rd|\d+-?th)$", token):
        return "_num_"
    # positive and negative ints and floats
    if re.findall("^\d+$", token):
        return "_num_"
    if re.findall("^-\d+$", token):
        return "_num_"
    if re.findall("^\d*\.\d+$", token):
        return "_num_"
    if re.findall("^-\d*\.\d+$", token):
        return "_num_"

    # misc.
    if re.findall("^(v|V)(s|S)\.?$|^(v|V)ersus$", token):
        return "vs"

    return token

# -----------------------------------------------------------------------------
# non-OO version

hyper_mapper = HypernymMapper()

def preprocess_text(text, mapper=hyper_mapper):
    """text -> list of preprocessed sentences"""
    return [preprocess_sentence(s) for s in nltk.sent_tokenize(text)]


def preprocess_sentence(sentence, mapper=hyper_mapper):
    """text -> list of POS-tagged tokens"""
    words = nltk.word_tokenize(sentence)
    mapped = mapper.replace_all(words)
    tagged = nltk.pos_tag(mapped)
    return tagged


# -----------------------------------------------------------------------------

def preprocess_civic():
    """load and preprocess CIViC summaries and corresponding abstracts

    returns (summaries, abstracts) as lists of lists of POS-tagged tokens"""

    civic, abstracts = loader.load_civic_abstracts()

    # print(civic.describe())
    # print(civic["evidence_statement"][0:100])

    # pm_ids = get_pm_ids(civic)
    print("No. of PubMed IDs:\t", len(civic["evidence_statement"]))

    # abstracts = get_abstracts(pm_ids[0:100]) #[0:100] #for quick test
    print("No. of abstracts:\t", len(abstracts))
    # print(abstracts["abstract"])

    start_time = timeit.default_timer()

    with multi.Pool(processes=multi.cpu_count()) as p:
        preprocessed_abstracts = p.map(preprocess_text, abstracts["abstract"])
        preprocessed_summaries = p.map(preprocess_text, civic["evidence_statement"])

    print("preprocessing took", timeit.default_timer() - start_time, "s")

    return preprocessed_summaries, preprocessed_abstracts


# execute (Test)

def dummy_test():
    ppsums, ppabss = loader.sentences_civic_abstracts()

    for pps in ppsums[0:20]:
        print(pps)

    for ppa in ppabss[0:20]:
        print(ppa)
