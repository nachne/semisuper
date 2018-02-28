# coding=utf-8
from __future__ import absolute_import, division, print_function

import cPickle as pickle
import os
import time

from semisuper import loaders
import key_sentence_predictor


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


try:
    with open(file_path("semisuper/pickles/sent_test_abstract_dicts.pickle"), "rb") as f:
        abstracts = pickle.load(f)[:120000]
except:
    medline_abstracts = loaders.fetch(loaders.get_pmids_from_query(term="cancer", max_ids=1200))
    abstracts = [{"pmid": a["PMID"], "abstract": a["AB"]} for a in medline_abstracts if a.get("AB") and a.get("PMID")]
    with open(file_path("semisuper/pickles/sent_test_abstract_dicts.pickle"), "wb") as f:
        pickle.dump(abstracts, f, protocol=2)

print(abstracts[0].keys())
print('astracts')

results = {}

batch_size=400
print("batch size:", batch_size)
predictor = key_sentence_predictor.KeySentencePredictor(batch_size=batch_size)
start_time = time.time()
results = predictor.transform(abstracts)

print("Preprocessing and predicting relevant sentences for", len(abstracts), " abstracts",
      "took", time.time() - start_time, "seconds")

for i in range(100):
    a = abstracts[i]
    idx = a["pmid"]
    print(idx, ":")
    for tup in results[idx]:
        print("%.4f" % tup[2], "\t", a["abstract"][tup[0]:tup[1]])
    print("\n")
