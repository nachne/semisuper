import pandas as pd
import numpy as np
import urllib.request
import nltk
import multiprocessing as multi
import pickle
from replacer import HypernymMapper, load_hypernyms
import loader
from helpers import take, flatten
import timeit

hyper_mapper =  HypernymMapper()

def preprocess_text(text, mapper=hyper_mapper):
    """text -> list of preprocessed sentences"""
    return [preprocess_sentence(s) for s in nltk.sent_tokenize(text)]

def preprocess_sentence(sentence, mapper=hyper_mapper):
    """text -> list of POS-tagged tokens"""
    words  = nltk.word_tokenize(sentence)
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

    print("preprocessing took", timeit.default_timer()-start_time, "s")

    return preprocessed_summaries, preprocessed_abstracts

# execute (Test)

ppsums, ppabss = preprocess_civic()

for pps in ppsums[0:20]:
    print(pps)

for ppa in ppabss[0:20]:
    print(ppa)