import os
import pandas as pd
import numpy as np
import build_corpus
from semisuper import loaders



def load_silver_standard(path=None):
    if path is None:
        path = file_path("./semisuper/output/silver_standard.tsv")
    try:
        corpus = pd.read_csv(path, sep="\t")
        return corpus
    except:
        pass

    return build_corpus.train_build(from_scratch=False)


def csv2array(csv):
    """extract """

    missing_label = 0.5

    labels = csv["label"].values
    decision_function = np.abs(csv["decision_function"].values)
    labels = csv["label"].values.astype(float)
    pos = csv["sentence_pos"].values
    text = csv["text"].values

    prevlabels = np.insert(labels[:-1], 0, missing_label)
    prevlabels[np.where(pos == 0.0)] = missing_label

    print(np.vstack((pos, labels, prevlabels)).T[0:40])

    return np.vstack((pos, labels, prevlabels)).T[0:40]




def run():
    corpus_csv = load_silver_standard()
    corpus_array = csv2array(corpus_csv)

    return corpus_array








def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


if __name__ == "__main__":
    run()
