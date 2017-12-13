import os
import random

import numpy as np
import pandas as pd

import build_corpus
from semisuper import loaders, super_model_selection
from semisuper.helpers import num_rows


def load_silver_standard(path=None):
    if path is None:
        path = file_path("./semisuper/output/silver_standard.tsv")
    try:
        corpus = pd.read_csv(path, sep="\t")
        return corpus
    except:
        pass

    return build_corpus.train_build(from_scratch=False)


def X_y_weights_from_csv(csv):
    """extract features and labels from csv"""

    text = csv["text"].values
    pos = csv["sentence_pos"].values.astype(float)

    weights = np.abs(csv["decision_function"].values.astype(float))
    weights *= 1.0 / np.max(weights)

    labels = csv["label"].values.astype(float)

    missing_label = 0.5
    prevlabels = np.insert(labels[:-1], 0, missing_label)
    prevlabels[np.where(pos == 0.0)] = missing_label

    # print(np.vstack((pos, labels, prevlabels)).T[0:40])

    X = np.vstack((text, pos)).T
    y = labels

    return X, y, weights


def run():

    new_abstracts = np.array(loaders.abstract_pmid_pos_sentences_query(anew=True, max_ids=2000000))

    corpus_csv = load_silver_standard()
    X_train, y_train, weights = X_y_weights_from_csv(corpus_csv)

    # _ = super_model_selection.get_best_model(X_train, y_train, weights)
    model = super_model_selection.get_best_model(X_train, y_train)

    X = np.vstack((new_abstracts[:, 2], new_abstracts[:, 1].astype(float))).T
    y = model.predict(X)

    # TODO compare with inductive setting

    print("Prediction for new PubMed abstracts:", np.sum(y), "/", num_rows(y), "(", np.sum(y) / num_rows(y), ")")
    print("Some positive sentences:")
    print(random.sample(X[np.where(y == 1.0)].tolist(), 10))
    print("Some negative sentences:")
    print(random.sample(X[np.where(y == 0.0)].tolist(), 10))

    return


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


if __name__ == "__main__":
    run()
