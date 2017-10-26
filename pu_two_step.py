from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn import svm, naive_bayes
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer, VectorizerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split as tts
from operator import itemgetter
from random import sample
import numpy as np
from numpy import array_equal as equal
from scipy import sparse
import pickle

from helpers import identity, num_rows, arrays, partition_pos_neg
from pu_ranking import ranking_cos_sim, rocchio
from transformers import BasicPreprocessor, TextStats, FeatureNamePipeline
from proba_label_nb import ProbaLabelMNB
from dummy_pipeline import build_and_evaluate, show_most_informative_features


# ----------------------------------------------------------------
# COMPLETE 2-STEP METHODS
# ----------------------------------------------------------------

# TODO implement model selection/iteration halting formula
def s_EM(P, U, spy_ratio=0.1, max_pos_ratio=0.5, tolerance=0.05, text=True, outpath=None):
    """S-EM algorithm as desscribed in \"Partially Supervised Classification...\". Two-step PU learning technique.

    1st step: get Reliable Negative documents using Spy Documents
    2nd step: iterate EM with P, U-RN, and RN
    """

    P, U = arrays([P, U])

    # step 1
    print("Determining confidence threshold using Spy Documents and rough I-EM\n")
    U_minus_RN, RN = get_RN_Spy_Docs(P, U, spy_ratio=spy_ratio, tolerance=0.5, noise_lvl=0.05, text=text)

    # step2
    print("\nIterating I-EM with P, U-RN, and RN")
    model = run_EM_with_RN(P, U_minus_RN, RN, tolerance=tolerance, max_pos_ratio=max_pos_ratio, text=text)

    report_save(model, P, U, outpath)

    return model

# TODO something wrong with partitioning!!!!!
def cr_SVM(P, U, max_neg_ratio=0.05, noise_lvl=0.1, alpha=16, beta=4, text=True, outpath=None):
    P, U = arrays([P, U])

    # step 1
    print("Determining RN using Cosine Similarity threshold and Rocchio\n")
    U_minus_RN, RN = get_RN_cosine_rocchio(P, U, noise_lvl=noise_lvl, alpha=alpha, beta=beta, text=text)

    # step2
    print("\nIterating SVM with P, U-RN, and RN")
    model = iterate_SVM(P, U_minus_RN, RN, text=text, max_neg_ratio=max_neg_ratio,
                        clf=svm.LinearSVC(class_weight='balanced'))

    # TODO list/array probs
    # report_save(model, P, U, outpath)

    return model


def i_EM(P, U, outpath=None, max_imbalance=1.5, max_pos_ratio=0.5, tolerance=0.05, text=True):
    """all-in-one PU method: I-EM algorithm for positive set P and unlabelled set U

    iterate NB classifier with updated labels for unlabelled set (initially negative) until convergence
    if U is much larger than P, randomly samples to max_imbalance-fold of |P|"""

    if num_rows(U) > max_imbalance * num_rows(P):
        U = np.array(sample(list(U), int(max_imbalance * num_rows(P))))

    model = iterate_EM(P, U, tolerance=tolerance, text=text, max_pos_ratio=max_pos_ratio)

    report_save(model, P, U, outpath)

    return model


# TODO noise level is quite crucial, should be set very high to give reasonable results
def standalone_cos_rocchio(P, U, noise_lvl=0.4, alpha=16, beta=4, text=True, outpath=None):
    """Naive P vs. U classification: Use first step cosine-Rocchio model as final classifier"""

    P, U = arrays([P, U])

    print("Computing ranking (cosine similarity to mean positive example)")
    mean_p_ranker = ranking_cos_sim(P)

    sims_P = mean_p_ranker.predict_proba(P)
    sims_U = mean_p_ranker.predict_proba(U)

    print("Choosing Potential Negative examples with ranking threshold")
    _, PN = select_PN_below_score(sims_P, U, sims_U, noise_lvl=noise_lvl)

    print("Building Rocchio model")
    model = rocchio(P, PN, alpha, beta, text)

    # TODO something weird is going on
    # print(model.predict(P))
    # print(model.predict(U))

    # TODO list/array probs
    # report_save(model, P, U, outpath)

    return model


# ----------------------------------------------------------------
# FIRST STEP TECHNIQUES
# ----------------------------------------------------------------

# TODO shortcut I-EM (should be only a handful of iterations)
def get_RN_Spy_Docs(P, U, spy_ratio=0.1, max_pos_ratio=0.5, tolerance=0.2, noise_lvl=0.05, text=True):
    """First step technique: Compute reliable negative docs from P using Spy Documents and I-EM"""

    P_minus_spies, spies = spy_partition(P, spy_ratio)
    U_plus_spies = np.concatenate((U, spies))

    model = iterate_EM(P_minus_spies, U_plus_spies, tolerance=tolerance, text=text, max_pos_ratio=max_pos_ratio)

    y_spies = model.predict_proba(spies)
    y_U = model.predict_proba(U)

    U_minus_RN, RN = select_PN_below_score(y_spies, U, y_U, noise_lvl=noise_lvl)

    return U_minus_RN, RN


def get_RN_cosine_rocchio(P, U, noise_lvl=0.05, alpha=16, beta=4, text=True):
    """extract Reliable Negative documents using cosine similarity and BinaryRocchio algorithm

    similarity is the cosine similarity compared to the mean positive sample.
    firstly, select Potential Negative docs that have lower similarity than the worst l% in P.
    source: negative harmful
    """

    P, U = arrays([P, U])

    print("Computing ranking (cosine similarity to mean positive example)")
    mean_p_ranker = ranking_cos_sim(P)

    sims_P = mean_p_ranker.predict_proba(P)
    sims_U = mean_p_ranker.predict_proba(U)

    print("Choosing Potential Negative examples with ranking threshold")
    _, PN = select_PN_below_score(sims_P, U, sims_U, noise_lvl=noise_lvl)

    print("Building Rocchio model to determine Reliable Negative examples")
    model = rocchio(P, PN, alpha, beta, text)

    y_U = model.predict(U)

    U_minus_RN, RN = partition_pos_neg(U, y_U)

    return U_minus_RN, RN


# ----------------------------------------------------------------
# SECOND STEP TECHNIQUES
# ----------------------------------------------------------------

# TODO alternative: use U in first iteration, with .5 probability labels?
def run_EM_with_RN(P, U, RN, max_pos_ratio=0.5, tolerance=0.05, text=True):
    """second step PU method: train NB with P and RN to get probabilistic labels for U, then iterate EM"""

    if num_rows(P) > 1.5 * num_rows(RN):
        P_init = np.array(sample(list(P), int(1.5 * num_rows(RN))))
    else:
        P_init = P

    print("\nBuilding classifier from Positive and Reliable Negative set")
    initial_model = build_proba_MNB(np.concatenate((P_init, RN)),
                                    [1] * num_rows(P_init) + [0] * num_rows(RN),
                                    verbose=False, text=text)

    y_P = np.array([1] * num_rows(P))

    print("\nCalculating initial probabilistic labels for Reliable Negative and Unlabelled set")
    ypU = initial_model.predict_proba(U)
    ypN = initial_model.predict_proba(RN)

    print("\nIterating EM algorithm on P, RN and U\n")
    model = iterate_EM(P, np.concatenate((RN, U)),
                       y_P, np.concatenate((ypN, ypU)),
                       tolerance=tolerance, text=text, max_pos_ratio=max_pos_ratio)

    return model


# TODO yield models in order to be able to choose best one
def iterate_EM(P, U, y_P=None, ypU=None, tolerance=0.05, text=True, max_pos_ratio=0.5):
    """EM algorithm for positive set P and unlabelled set U

        iterate NB classifier with updated labels for unlabelled set (with optional initial labels) until convergence"""

    if y_P is None:
        y_P = np.array([1.] * num_rows(P))
    if ypU is None:
        ypU = np.array([0.] * num_rows(U))

    ypU_old = [-999]

    iterations = 0
    model = None

    while not almost_equal(ypU_old, ypU, tolerance):

        iterations += 1

        print("Iteration #", iterations)
        print("Building new model using probabilistic labels")

        model = build_proba_MNB(np.concatenate((P, U)),
                                np.concatenate((y_P, ypU)), text=text)

        ypU_old = ypU
        print("Predicting probabilities for U")
        ypU = model.predict_proba(U)
        # print(ypU[:10], ypU[-10:])

        predU = [round(p) for p in ypU]
        pos_ratio = sum(predU) / len(U)

        print("Unlabelled instances classified as positive:", sum(predU), "/", len(U),
              "(", pos_ratio * 100, "%)\n")

        if pos_ratio >= max_pos_ratio:
            print("Acceptable ratio of positively labelled sentences in U is reached.")
            break

    print("Returning final model after", iterations, "iterations")
    return model


def iterate_SVM(P, U, RN, text=True, max_neg_ratio=0.05, clf=svm.LinearSVC(class_weight='balanced')):
    """runs an SVM classifier trained on P and RN iteratively, augmenting RN

    after each iteration, the documents in U classified as negative are moved to RN until there are none left.
    max_neg_ratio is the maximum accepted ratio of P to be classified as negative by final classifier.
    if the final classifier regards more than max_neg_ratio of P as negative, return the initial one."""

    y_P = np.ones(num_rows(P))
    y_RN = np.zeros(num_rows(RN))

    print("Building initial SVM classifier with Positive and Reliable Negative docs")
    initial_model = build_and_evaluate(np.concatenate((P, RN)),
                                       np.concatenate((y_P, y_RN)),
                                       classifier=clf,
                                       text=text)

    print("Predicting U with initial SVM, adding negatively classified docs to RN for iteration")
    y_U = initial_model.predict(U)
    Q, W = partition_pos_neg(U, y_U)
    iteration = 0

    # iterate SVM, each turn augmenting RN by the documents in Q classified negative
    while num_rows(W) > 0:
        print("new negative docs since last iteration:", num_rows(W))
        iteration += 1
        print("Iteration #", iteration)

        RN = np.concatenate((RN, W))
        y_RN = np.zeros(num_rows(RN))

        model = build_and_evaluate(np.concatenate((P, RN)),
                                   np.concatenate((y_P, y_RN)),
                                   classifier=clf,
                                   text=text)

        y_U = model.predict(Q)
        Q, W = partition_pos_neg(Q, y_U)

    y_P_initial = initial_model.predict(P)
    y_P_final = model.predict(P)

    initial_neg_ratio = 1 - num_rows(np.nonzero(y_P_initial)) / num_rows(y_P_initial)
    final_neg_ratio = 1 - num_rows(np.nonzero(y_P_final)) / num_rows(y_P_final)

    print("Ratio of positive samples classified as negative by initial SVM:", initial_neg_ratio)
    print("Ratio of positive samples classified as negative by final SVM:", final_neg_ratio)

    if (final_neg_ratio > max_neg_ratio):
        print("Final classifier discards too many positive examples.")
        print("Returning initial classifier instead")
        return initial_model
    else:
        print("Returning final classifier")
        return model


# ----------------------------------------------------------------
# general MNB model builder
# ----------------------------------------------------------------

# TODO: Make versatile module for this and reuse in respective functions
# TODO: Do vectorization only once (not every time a new model is made in an iteration)
def build_proba_MNB(X, y, verbose=True, text=True):
    """build multinomial Naive Bayes classifier that accepts probabilistic labels
    if text is true, preprocess Text with binary encoding"""

    def build(X, y):
        """
        Inner build function that builds a single model.
        """
        if text:
            model = Pipeline([
                ('preprocessor', BasicPreprocessor()),
                ('vectorizer', CountVectorizer(binary=True, tokenizer=identity, lowercase=False, ngram_range=(1, 3))),
                ('classifier', ProbaLabelMNB(alpha=1))
            ])
        else:
            model = Pipeline([
                ('classifier', ProbaLabelMNB(alpha=1))
            ])

        model.fit(X, y)
        return model

    if verbose:
        print("Building model ...")
    model = build(X, y)

    return model


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------

def almost_equal(pairs1, pairs2, tolerance=0.1):
    """helper function that checks if difference of probabilistic labels is smaller than tolerance for all indices"""
    zipped = zip(pairs1, pairs2)
    diffs = [abs(p1 - p2) <= tolerance
             for p1, p2 in zipped]
    return all(diffs)


def spy_partition(P, spy_ratio=0.1):
    """Partition P, extracting Spy Documents"""

    num_P = num_rows(P)
    num_idx = int(spy_ratio * num_P)

    # define spy partition
    idx = sample(range(num_P), num_idx)
    spies = P[idx]

    # define rest partition
    mask = np.ones(num_P, dtype=bool)
    mask[idx] = False
    P_minus_spies = P[mask]

    return P_minus_spies, spies


def select_PN_below_score(y_pos, U, y_U, noise_lvl=0.1):
    """given the scores of positive docs, a set of unlabelled docs, and their scores, extract potential negative set"""

    y_pos_sorted = sorted(y_pos)

    # choose probability threshold such that a noise_lvl-th part of spy docs is rated lower
    threshold = y_pos_sorted[int(noise_lvl * num_rows(y_pos_sorted))]
    print("Threshold given noise level:", threshold)

    neg_idx = np.where(y_U < threshold)

    pos_idx = np.ones(num_rows(y_U), dtype=bool)
    pos_idx[neg_idx] = False

    PN = U[neg_idx]
    print("Unlabelled docs below threshold:", num_rows(PN), "of", num_rows(U), "\n")

    U_minus_PN = U[pos_idx]
    # print(U_minus_RN[:10])

    return U_minus_PN, PN


def report_save(model, P, N, outpath=None):
    print("Classification Report:\n")
    y_pred = model.predict(np.concatenate((P, N)))
    print(y_pred[:10], y_pred[-10:])
    print(clsr([1. for _ in P] + [0. for _ in N], y_pred))

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))
