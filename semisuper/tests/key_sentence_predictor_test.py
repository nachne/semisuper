import pickle
import os
import time
import multiprocessing
import random

from semisuper import loaders, helpers
import key_sentence_predictor


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


try:
    with open(file_path("../pickles/sent_test_abstract_dicts.pickle"), "rb") as f:
        abstracts = pickle.load(f)
except:
    medline_abstracts = loaders.fetch(loaders.get_pmids_from_query(term="cancer", max_ids=120000))
    abstracts = [{"pmid": a["PMID"], "abstract": a["AB"]} for a in medline_abstracts if a.get("AB") and a.get("PMID")]
    with open(file_path("../pickles/sent_test_abstract_dicts.pickle"), "wb") as f:
        pickle.dump(abstracts, f)

print("Building predictor")
predictor = key_sentence_predictor.KeySentencePredictor()

def predict_with_predictor(abs_pred):
    abstracts, predictor = abs_pred
    return predictor.transform(abstracts)

with multiprocessing.Pool(4) as p:
    predictors = [key_sentence_predictor.KeySentencePredictor() for _ in range(4)]
    start_time = time.time()
    results = helpers.flatten(p.map(predict_with_predictor,
                                    zip(helpers.partition(abstracts, 30000), predictors),
                                    chunksize=1))
    print("Preprocessing and predicting relevant sentences for", len(abstracts), " abstracts",
          "took", time.time() - start_time, "seconds")

for i in range(100):
    record = results(list(results)[i])
    print(i, "\t", "%.4g" % record[2], abstracts[i][record[0]:record[1]])
