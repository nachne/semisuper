from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, Normalizer, normalize
from sklearn.linear_model import SGDClassifier
from sklearn import svm, naive_bayes
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer, VectorizerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split as tts
from operator import itemgetter
from random import sample
from helpers import identity
import numpy as np
from numpy import array_equal as equal
from scipy import sparse
import pickle

from transformers import BasicPreprocessor, TextStats, FeatureNamePipeline
from proba_label_nb import proba_label_MNB

# ----------------------------------------------------------------

# ----------------------------------------------------------------
# NB-EM-related

def reliable_neg_EM(P, U, RN, outpath=None, max_pos_ratio=0.5, tolerance=0.05, text=True):
    """second step PU method: train NB with P and RN to get probabilistic labels for U, then iterate EM"""

    initial_model = i_EM(P, RN, outpath, max_pos_ratio, tolerance, text)

    y_P = np.array([[1., 0.]] * np.shape(P)[0])

    ypU = initial_model.predict_proba(U)
    ypN = initial_model.predict_proba(RN)

    model = iterate_EM(P,
                       np.concatenate((RN, U)),
                       y_P,
                       np.concatenate((ypN, ypU)),
                       tolerance, text, max_pos_ratio)

    print("Classification Report:\n")
    y_pred = model.predict(np.concatenate((P, U)))
    print(y_pred)
    print(clsr([p[0] for p in y_P] + [0. for u in ypU], y_pred))

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


def i_EM(P, U, outpath=None, max_imbalance=1.5, max_pos_ratio=0.5, tolerance=0.05, text=True):
    """all-in-one PU method: I-EM algorithm for positive set P and unlabelled set U

    iterate NB classifier with updated labels for unlabelled set (initially negative) until convergence
    if U is much larger than P, randomly samples to max_imbalance-fold of |P|"""

    num_P = np.shape(P)[0]
    num_U = np.shape(U)[0]

    y_P = np.array([[1., 0.]] * num_P)

    if num_U > max_imbalance * num_P:
        U = np.array(sample(list(U), int(max_imbalance * num_P)))

    ypU = np.array([[0., 1.]] * np.shape(U)[0])

    model = iterate_EM(P, U, y_P, ypU, tolerance, text, max_pos_ratio)

    print("Classification Report:\n")
    y_pred = model.predict(np.concatenate((P, U)))
    print(y_pred)
    print(clsr([p[0] for p in y_P] + [0. for u in ypU], y_pred))

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


# TODO yield models in order to be able to choose best one
def iterate_EM(P, U, y_P=None, ypU=None, tolerance=0.05, text=True, max_pos_ratio=0.5):
    """EM algorithm for positive set P and unlabelled set U

        iterate NB classifier with updated labels for unlabelled set (with optional initial labels) until convergence"""
    if y_P is None:
        y_P = np.array([[1., 0.]] * np.shape(P)[0])
    if ypU is None:
        ypU = np.array([[0., 1.]] * np.shape(U)[0])

    ypU_old = [[-1, -1]]

    iterations = 0
    model = None

    while not almost_equal(ypU_old, ypU, tolerance):

        iterations += 1

        print("Iteration #", iterations)

        print("building new model using probabilistic labels")

        model = build_proba_MNB(np.concatenate((P, U)),
                                np.concatenate((y_P, ypU)), text=text)

        ypU_old = ypU
        print("predicting probabilities for U")
        ypU = model.predict_proba(U)
        print(ypU)

        print("labels for U")
        predU = [round(p[0]) for p in ypU]
        pos_ratio = sum(predU) / len(U)

        print("Unlabelled instances classified as positive:", sum(predU), "/", len(U),
              "(", pos_ratio * 100, "%)\n")

        if pos_ratio >= max_pos_ratio:
            print("Acceptable ratio of positively labelled sentences in U is reached.")
            break

    print("Returning final model after", iterations, "iterations")
    return model


# ----------------------------------------------------------------
# general MNB model builder

# TODO: Make versatile module for this and reuse in respective functions
def build_proba_MNB(X, y,
                    verbose=True, text=True):
    """build multinomial Naive Bayes classifier that accepts probabilistic labels"""

    def build(X, y):
        """
        Inner build function that builds a single model.
        """
        if text:
            model = Pipeline([
                ('preprocessor', BasicPreprocessor()),
                ('vectorizer', FeatureUnion(transformer_list=[
                    ("words", CountVectorizer(
                            tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1, 3))
                     )
                    # ,
                    # ("stats", FeatureNamePipeline([
                    #     ("stats", TextStats()),
                    #     ("vect", DictVectorizer())
                    # ]))
                ]
                )),
                ('classifier', proba_label_MNB(alpha=1))
            ])
        else:
            model = Pipeline([
                ('classifier', proba_label_MNB(alpha=1))
            ])

        model.fit(X, y)
        return model

    if verbose:
        print("Building model ...")
    model = build(X, y)

    return model


# ----------------------------------------------------------------
# helpers

def almost_equal(pairs1, pairs2, tolerance=0.05):
    """helper function that checks if difference of probabilistic labels is smaller than tolerance for all indices"""
    zipped = zip(pairs1, pairs2)
    diffs = [abs(p1[0] - p2[0]) < tolerance
             for p1, p2 in zipped]
    return all(diffs)
