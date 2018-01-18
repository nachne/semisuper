import pickle
import os
import time

from semisuper import loaders
import key_sentence_predictor


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


try:
    with open(file_path("../pickles/sent_test_abstract_dicts.pickle"), "rb") as f:
        abstracts = pickle.load(f)
except:
    medline_abstracts = loaders.fetch(loaders.get_pmids_from_query(term="cancer", max_ids=30000))
    abstracts = [{"pmid": a["PMID"], "abstract": a["AB"]} for a in medline_abstracts if a.get("AB") and a.get("PMID")]
    with open(file_path("../pickles/sent_test_abstract_dicts.pickle"), "wb") as f:
        pickle.dump(abstracts, f)

print("Building predictor")
predictor = key_sentence_predictor.KeySentencePredictor()

start_time = time.time()
results = predictor.transform(abstracts)
print("Preprocessing and predicting relevant sentences for", len(abstracts), " abstracts",
      "took", time.time() - start_time, "seconds")

for i in range(0, 100):
    # range(len(results)):
    for start, end, score in results[i]["relevant"]:
        print("%.4g" % score, "\t", abstracts[i]["abstract"][start:end])
    print("\n")
