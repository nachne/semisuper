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

concept_mapper =  HypernymMapper()

def map_concepts(text):
    return concept_mapper.replace_all(text)

# transform list of texts into list of preprocessed sentences

def preprocess_text(text):
    return [preprocess_sentence(s) for s in nltk.sent_tokenize(text)]

def preprocess_sentence(sentence):
    words  = nltk.word_tokenize(sentence)
    mapped = concept_mapper.replace_all(words)
    tagged = nltk.pos_tag(mapped)
    return tagged

# -----------------------------------------------------------------------------
# execute (Test)

# hypernym_dict = load_hypernyms()
# print(take(100, hypernym_dict.items()))

civic, abstracts = loader.load_civic_abstracts()

# civic = read_civic("/Users/emperor/HU/BA/civic/nightly-ClinicalEvidenceSummaries.tsv")
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

print("preprocessing took ", timeit.default_timer()-start_time, "s")

for pa in preprocessed_abstracts[0:20]:
    print(pa)

for ps in preprocessed_summaries[0:20]:
    print(ps)


# obsolete

def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()


def get_pubmed(id, timeout=60):
    prefix = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
              "?db=pubmed&retmode=text&rettype=medline&id=")
    text = load_url(prefix + id, timeout)
    return medline_to_df(text)


def medline_to_df(text):
    record = Medline.read(text)
    return record
