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

def s_EM(P, U, outpath=None, spy_ratio=0.1, max_pos_ratio=0.5, tolerance=0.05, text=True):
    """S-EM algorithm as desscribed in \"Partially Supervised Classification...\". Two-step PU learning technique.

    1st step: get Reliable Negative documents using Spy Documents
    2nd step: iterate EM with P, U-RN, and RN
    """

    U_minus_RN, RN = get_RN_Spies(P, U, text)

    print("Classification Report:\n")
    y_pred = model.predict(np.concatenate((P, U)))
    print(y_pred[:10], y_pred[-10:])
    print(clsr([1 for _ in P] + [0 for _ in U], y_pred))

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


def get_RN_Spies(P, U, spy_ratio=0.1, max_pos_ratio=0.5, tolerance=0.2, text=True):
    """Compute reliable negative docs from P using the Spy Document technique"""

    P_minus_spies, spies = spy_partition(P, spy_ratio)
    U_plus_spies = np.concatenate((U, spies))

    model = iterate_EM(P_minus_spies, U_plus_spies, tolerance=tolerance, text=text, max_pos_ratio=max_pos_ratio)

    y_spies = model.predict_proba(spies)
    y_U = model.predict_proba(U)

    U_minus_RN, RN = select_RN_with_spy_scores(y_spies, U, y_U)

    return U_minus_RN, RN

def select_RN_with_spy_scores(y_spies, U, y_U, noise_lvl = 0.15):
    y_spies_sorted = sorted(y_spies, key=itemgetter(0))

    # choose probability threshold such that a noise_lvl-th part of spy docs is rated lower
    threshold = y_spies_sorted[int(noise_lvl * np.shape(y_spies_sorted)[0])]
    print("thrsh", threshold)

    neg_idx = np.where(y_U<=threshold)

    pos_idx = np.ones(np.shape(y_U)[0], dtype=bool)
    pos_idx[neg_idx] = False


    RN = U[neg_idx]
    print(RN)

    U_minus_RN = U[pos_idx]
    print(U_minus_RN)

    return U_minus_RN, RN




def run_EM_with_RN(P, U, RN, outpath=None, max_pos_ratio=0.5, tolerance=0.05, text=True):
    """second step PU method: train NB with P and RN to get probabilistic labels for U, then iterate EM"""

    if np.shape(P)[0] > 1.5 * np.shape(RN)[0]:
        P_init = np.array(sample(list(P), int(1.5 * np.shape(RN)[0])))
    else:
        P_init = P

    print("\nbuilding classifier from Positive and Reliable Negative set")
    initial_model = build_proba_MNB(np.concatenate((P_init, RN)),
                                    [1] * np.shape(P_init)[0] + [0] * np.shape(RN)[0],
                                    verbose=False, text=True)

    y_P = np.array([[1., 0.]] * np.shape(P)[0])

    print("\ncalculating initial probabilistic labels for Reliable Negative and Unlabelled set")
    ypU = initial_model.predict_proba(U)
    ypN = initial_model.predict_proba(RN)

    print("\niterating EM algorithm on P, RN and U\n")
    model = iterate_EM(P, np.concatenate((RN, U)),
                       y_P, np.concatenate((ypN, ypU)),
                       tolerance=tolerance, text=text, max_pos_ratio=max_pos_ratio)

    print("Classification Report\n")
    y_pred = model.predict(np.concatenate((P, U)))
    print(y_pred[:10], y_pred[-10:])
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

    if np.shape(U)[0] > max_imbalance * np.shape(P)[0]:
        U = np.array(sample(list(U), int(max_imbalance * np.shape(P)[0])))

    model = iterate_EM(P, U, tolerance=tolerance, text=text, max_pos_ratio=max_pos_ratio)

    print("Classification Report:\n")
    y_pred = model.predict(np.concatenate((P, U)))
    print(y_pred[:10], y_pred[-10:])
    print(clsr([1 for _ in P] + [0 for _ in U], y_pred))

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
        # print(ypU[:10], ypU[-10:])

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
# TODO: Do vectorization only once (not every time a new model is made in an iteration)
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


def spy_partition(P, spy_ratio=0.1):
    print(P)

    num_P = np.shape(P)[0]
    num_idx = int(spy_ratio * num_P)

    # define spy partition
    idx = np.array(sample(range(num_P), num_idx))
    spies = P[idx]

    # define rest partition
    mask = np.ones(num_P, dtype=bool)
    mask[idx] = False
    P_minus_spies = P[mask]

    return P_minus_spies, spies
