import os
import random
import time

import numpy as np
import pandas as pd

import pickle

import build_corpus_and_ss_classifier
from semisuper import loaders, super_model_selection
from semisuper.helpers import num_rows

# ----------------------------------------------------------------
# Loading silver standard
# ----------------------------------------------------------------


def load_silver_standard(path=None):
    if path is None:
        path = file_path("./semisuper/output/silver_standard.tsv")

    try:
        corpus = pd.read_csv(path, sep="\t")
        return corpus
    except:
        pass

    return build_corpus_and_ss_classifier.train_build(from_scratch=False)


def X_y_from_csv(csv):
    """extract features and labels from csv"""

    text = csv["text"].values
    pos = csv["sentence_pos"].values
    title = csv["title"].values

    dec_fn = csv["decision_function"].values

    labels = csv["label"].values

    # missing_label = 0.5
    # prevlabels = np.insert(labels[:-1], 0, missing_label)
    # prevlabels[np.where(pos == 0.0)] = missing_label

    X = np.vstack((text, pos, title, dec_fn)).T
    y = labels

    return X, y

def max_score_from_csv(csv):
    """get maximum absolute decision function/probability value in corpus"""

    vals = csv["decision_function"].values
    return np.max(vals)


# ----------------------------------------------------------------
# Building supervised pipeline
# ----------------------------------------------------------------

def build_classifier(outpath="./semisuper/pickles/super_pipeline.pickle"):
    corpus_csv = load_silver_standard()
    X_train, y_train = X_y_from_csv(corpus_csv)

    model = super_model_selection.best_model_cross_val(X_train, y_train, fold=5)  # TODO restore 10

    if outpath:
        with open(outpath, "wb") as f:
            print("saving supervised pipeline to", outpath)
            pickle.dump(model, f)

    new_abstracts = np.array(loaders.abstract_pmid_pos_sentences_query(anew=False, max_ids=1000))
    pmid, pos, text, title = [0, 1, 2, 3]

    print("\nSupervised model\n")

    X = np.vstack((new_abstracts[:, text],
                   new_abstracts[:, pos].astype(float),
                   new_abstracts[:, title])).T
    y = model.predict(X)

    print("Prediction for new PubMed abstracts:", np.sum(y), "/", num_rows(y), "(", np.sum(y) / num_rows(y), ")")
    print("Some positive sentences:")
    [print(x) for x in random.sample(X[np.where(y == 1.0)].tolist(), 10)]
    print("Some negative sentences:")
    [print(x) for x in random.sample(X[np.where(y == 0.0)].tolist(), 10)]

    # ----------------------------------------------------------------
    # TODO remove tests below this line

    print("\nInductive Semi-Supervised model\n")

    semi_pipeline = build_corpus_and_ss_classifier.train_pipeline(from_scratch=False, ratio=1.0)

    X_ss = new_abstracts[:, text]
    y_ss = semi_pipeline.predict(X_ss)

    print("Prediction for new PubMed abstracts:", np.sum(y_ss), "/", num_rows(y_ss), "(", np.sum(y_ss) / num_rows(y_ss),
          ")")
    print("Some positive sentences:")
    [print(x) for x in random.sample(X_ss[np.where(y_ss == 1.0)].tolist(), 10)]
    print("Some negative sentences:")
    [print(x) for x in random.sample(X_ss[np.where(y_ss == 0.0)].tolist(), 10)]

    print("\nTest abstracts\n")
    start_time = time.time()
    max_abs = 1000
    # test_abstracts = np.array(loaders.abstract_pmid_pos_sentences_idlist([str(x) for x in range(20000000, 20001000)]))
    test_abstracts = np.array(loaders.abstract_pmid_pos_sentences_query(max_ids=max_abs, anew=True, term="cancer"))
    print("fetching", max_abs, "abstracts took %s seconds\n" % (time.time() - start_time))
    print(test_abstracts)

    start_time = time.time()
    y_sup = model.predict(np.vstack((test_abstracts[:, text],
                                     test_abstracts[:, pos].astype(float),
                                     test_abstracts[:, title])).T)
    print("\nsupervised classification of", num_rows(test_abstracts),
          "sentences took %s seconds\n" % (time.time() - start_time))

    start_time = time.time()
    y_semi = semi_pipeline.predict(test_abstracts[:, text])
    print("\ninductive classification of", num_rows(test_abstracts),
          "sentences took %s seconds\n" % (time.time() - start_time))

    print("Supervised:", np.sum(y_sup), "inductive:", np.sum(y_semi),
          "agreement:", np.size(np.where(y_sup == y_semi)) / num_rows(test_abstracts))
    print(y_sup + 2 * y_semi)

    # ----------------------------------------------------------------

    return model


# ----------------------------------------------------------------

def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


# ----------------------------------------------------------------
# Execution
# ----------------------------------------------------------------

if __name__ == "__main__":
    build_classifier()
