import pickle
import os
import time
import multiprocessing
import random
from itertools import cycle

from semisuper import loaders, helpers
import key_sentence_predictor


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


try:
    with open(file_path("../pickles/sent_test_abstract_dicts.pickle"), "rb") as f:
        abstracts = pickle.load(f)[:120000]
except:
    medline_abstracts = loaders.fetch(loaders.get_pmids_from_query(term="cancer", max_ids=120000))
    abstracts = [{"pmid": a["PMID"], "abstract": a["AB"]} for a in medline_abstracts if a.get("AB") and a.get("PMID")]
    with open(file_path("../pickles/sent_test_abstract_dicts.pickle"), "wb") as f:
        pickle.dump(abstracts, f)


def predict_with_predictor(abs_pred):
    abstracts, predictor = abs_pred
    return predictor.transform(abstracts)

pool_size = min(multiprocessing.cpu_count(), 12)

for batch_size in [1, 10, 100, 200, 300, 400, 500, 1000]:
    print("batch size:", batch_size)
    with multiprocessing.Pool(pool_size) as p:
        predictors = [key_sentence_predictor.KeySentencePredictor(batch_size=batch_size) for _ in
                      range(pool_size)]
        start_time = time.time()
        results = helpers.merge_dicts(p.map(predict_with_predictor,
                                            zip(helpers.partition(abstracts, len(abstracts) // pool_size),
                                                predictors),
                                            chunksize=1))

        print("Preprocessing and predicting relevant sentences for", len(abstracts), " abstracts",
              "took", time.time() - start_time, "seconds")

for i in range(100):
    a = abstracts[i]
    idx = a["pmid"]
    print(idx, ":")
    for tup in results[idx]:
        print("%.4f" % tup[2], "\t", a["abstract"][tup[0]:tup[1]])
    print("\n")
