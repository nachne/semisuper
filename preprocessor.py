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

class Preprocessor(object):

    def __init__(self):
        """construct hypernym dict and replacer on initialisation"""
        self.concept_mapper =  HypernymMapper()

    def preprocess_text(self, text):
        """text -> list of preprocessed sentences"""
        return [self.preprocess_sentence(s) for s in nltk.sent_tokenize(text)]

    def preprocess_sentence(self, sentence):
        """sentence -> list of POS annotated tokens"""
        words = nltk.word_tokenize(sentence)
        mapped = self.concept_mapper.replace_all(words)
        tagged = nltk.pos_tag(mapped)
        return tagged

    def map_concepts(self, text):
        """replace tokens by their hypernyms"""
        return self.concept_mapper.replace_all(text)






# -----------------------------------------------------------------------------
# execute (Test)

civic, abstracts = loader.load_civic_abstracts()
preprocessor = Preprocessor()

# print(civic.describe())
# print(civic["evidence_statement"][0:100])

# pm_ids = get_pm_ids(civic)
print("no. of PubMed IDs:\t", len(civic["evidence_statement"]))

# abstracts = get_abstracts(pm_ids[0:100]) #[0:100] #for quick test
print("no. of abstracts:\t", len(abstracts))
# print(abstracts["abstract"])

start_time = timeit.default_timer()

with multi.Pool(processes=multi.cpu_count()) as p:
    preprocessed_abstracts = p.map(preprocessor.preprocess_text, abstracts["abstract"])
    preprocessed_summaries = p.map(preprocessor.preprocess_text, civic["evidence_statement"])

print("preprocessing took", timeit.default_timer()-start_time, "s")

for pa in preprocessed_abstracts[0:20]:
    print(pa)

for ps in preprocessed_summaries[0:20]:
    print(ps)
