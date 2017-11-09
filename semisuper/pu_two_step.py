from sklearn import svm, naive_bayes
from sklearn.metrics import classification_report as clsr
from sklearn.model_selection import train_test_split as tts
from operator import itemgetter
from random import sample
import numpy as np
from numpy import array_equal as equal
from scipy import sparse
import pickle
from semisuper.helpers import identity, num_rows, arrays, partition_pos_neg, pu_measure
from semisuper.pu_ranking import ranking_cos_sim, rocchio
from semisuper.proba_label_nb import build_proba_MNB
from semisuper.dummy_pipeline import build_and_evaluate, show_most_informative_features


# ----------------------------------------------------------------
# COMPLETE 2-STEP METHODS
# ----------------------------------------------------------------

# TODO noise level is quite crucial, should be set very high to give reasonable results
def cr_SVM(P, U, max_neg_ratio=0.05, noise_lvl=0.2, alpha=16, beta=4, text=True, outpath=None):
    P, U = arrays([P, U])

    # step 1
    print("Determining RN using Cosine Similarity threshold and Rocchio\n")
    U_minus_RN, RN = get_RN_cosine_rocchio(P, U, noise_lvl=noise_lvl, alpha=alpha, beta=beta, text=text)

    # step2
    print("\nIterating SVM with P, U-RN, and RN")
    model = iterate_SVM(P, U_minus_RN, RN, text=text, max_neg_ratio=max_neg_ratio)

    report_save(model, P, U, outpath)

    return model


def roc_SVM(P, U, max_neg_ratio=0.05, alpha=16, beta=4, text=True, outpath=None):
    P, U = arrays([P, U])

    # step 1
    print("Determining RN using Rocchio method\n")
    U_minus_RN, RN = get_RN_rocchio(P, U, alpha=alpha, beta=beta, text=text)

    # step2
    print("\nIterating SVM with P, U-RN, and RN")
    model = iterate_SVM(P, U_minus_RN, RN, text=text, max_neg_ratio=max_neg_ratio)

    report_save(model, P, U, outpath)

    return model


# TODO implement model selection/iteration halting formula
# Pr[f(X) ≠ Y] = Pr[f(X) = 1] - Pr[Y = 1] + 2 Pr[f(X) = 0 | Y = 1] * Pr[Y = 1]
def s_EM(P, U, spy_ratio=0.1, max_pos_ratio=1.0, tolerance=0.1, noise_lvl=0.1, text=True, clf_selection=True,
         outpath=None):
    """S-EM algorithm as desscribed in \"Partially Supervised Classification...\". Two-step PU learning technique.

    1st step: get Reliable Negative documents using Spy Documents
    2nd step: iterate EM with P, U-RN, and RN
    """

    P, U = arrays([P, U])

    # step 1
    print("Determining confidence threshold using Spy Documents and I-EM\n")
    U_minus_RN, RN = get_RN_Spy_Docs(P, U,
                                     spy_ratio=spy_ratio, tolerance=tolerance, noise_lvl=noise_lvl, text=text)

    # step2
    print("\nIterating I-EM with P, U-RN, and RN")
    model = run_EM_with_RN(P, U_minus_RN, RN,
                           tolerance=tolerance, max_pos_ratio=max_pos_ratio, text=text,
                           clf_selection=clf_selection)

    report_save(model, P, U, outpath)

    return model


def i_EM(P, U, outpath=None, max_imbalance=1.5, max_pos_ratio=1.0, tolerance=0.1, text=True):
    """all-in-one PU method: I-EM algorithm for positive set P and unlabelled set U

    iterate NB classifier with updated labels for unlabelled set (initially negative) until convergence
    if U is much larger than P, randomly samples to max_imbalance-fold of |P|"""

    if num_rows(U) > max_imbalance * num_rows(P):
        U = np.array(sample(list(U), int(max_imbalance * num_rows(P))))

    model = iterate_EM(P, U, tolerance=tolerance, text=text, max_pos_ratio=max_pos_ratio, clf_selection=False)

    report_save(model, P, U, outpath)

    return model


# TODO PEBL (1-DNF, SVM)

# ----------------------------------------------------------------
# FIRST STEP TECHNIQUES
# ----------------------------------------------------------------

# TODO shortcut I-EM (should be only a handful of iterations)
def get_RN_Spy_Docs(P, U, spy_ratio=0.1, max_pos_ratio=0.5, tolerance=0.2, noise_lvl=0.05, text=True):
    """First step technique: Compute reliable negative docs from P using Spy Documents and I-EM"""

    P_minus_spies, spies = spy_partition(P, spy_ratio)
    U_plus_spies = np.concatenate((U, spies))

    model = iterate_EM(P_minus_spies, U_plus_spies, tolerance=tolerance, text=text, max_pos_ratio=max_pos_ratio,
                       clf_selection=False)

    y_spies = model.predict_proba(spies)
    y_U = model.predict_proba(U)

    U_minus_RN, RN = select_PN_below_score(y_spies, U, y_U, noise_lvl=noise_lvl)

    return U_minus_RN, RN


def get_RN_rocchio(P, U, alpha=16, beta=4, text=True):
    """extract Reliable Negative documents using BinaryRocchio algorithm"""

    P, U = arrays([P, U])

    print("Building Rocchio model to determine Reliable Negative examples")
    model = rocchio(P, U, alpha=alpha, beta=beta, text=text)

    y_U = model.predict(U)

    U_minus_RN, RN = partition_pos_neg(U, y_U)
    print("Reliable Negative examples in U:", num_rows(RN), "(", 100 * num_rows(RN) / num_rows(U), "%)")

    return U_minus_RN, RN


def get_RN_cosine_rocchio(P, U, noise_lvl=0.20, alpha=16, beta=4, text=True):
    """extract Reliable Negative documents using cosine similarity and BinaryRocchio algorithm

    similarity is the cosine similarity compared to the mean positive sample.
    firstly, select Potential Negative docs that have lower similarity than the worst l% in P.
    source: negative harmful
    """

    P, U = arrays([P, U])

    print("Computing ranking (cosine similarity to mean positive example)")
    mean_p_ranker = ranking_cos_sim(P, text=text)

    sims_P = mean_p_ranker.predict_proba(P)
    sims_U = mean_p_ranker.predict_proba(U)

    # TODO write useful method for this; with PU-score, it's terrible
    # noise_lvl = choose_noise_lvl(sims_P, sims_U)
    # print("Choosing noise level that maximises score:", noise_lvl)

    print("Choosing Potential Negative examples with ranking threshold")
    _, PN = select_PN_below_score(sims_P, U, sims_U, noise_lvl=noise_lvl)

    print("Building Rocchio model to determine Reliable Negative examples")
    model = rocchio(P, PN, alpha=alpha, beta=beta, text=text)

    y_U = model.predict(U)

    U_minus_RN, RN = partition_pos_neg(U, y_U)
    print("Reliable Negative examples in U:", num_rows(RN), "(", 100 * num_rows(RN) / num_rows(U), "%)")

    return U_minus_RN, RN


# TODO 1-DNF for PEBL


# ----------------------------------------------------------------
# SECOND STEP TECHNIQUES
# ----------------------------------------------------------------

def run_EM_with_RN(P, U, RN, max_pos_ratio=1.0, tolerance=0.05, text=True, clf_selection=True):
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
                       tolerance=tolerance, text=text, max_pos_ratio=max_pos_ratio,
                       clf_selection=clf_selection)

    return model


# TODO yield models in order to be able to choose best one
def iterate_EM(P, U, y_P=None, ypU=None, tolerance=0.05, max_pos_ratio=1.0, text=True, clf_selection=False):
    """EM algorithm for positive set P and unlabelled set U

        iterate NB classifier with updated labels for unlabelled set (with optional initial labels) until convergence"""

    if y_P is None:
        y_P = np.array([1.] * num_rows(P))
    if ypU is None:
        ypU = np.array([0.] * num_rows(U))

    ypU_old = [-999]

    iterations = 0
    old_model = None
    new_model = None

    while not almost_equal(ypU_old, ypU, tolerance):

        iterations += 1

        print("Iteration #", iterations)
        print("Building new model using probabilistic labels")

        if clf_selection:
            old_model = new_model

        new_model = build_proba_MNB(np.concatenate((P, U)),
                                    np.concatenate((y_P, ypU)), text=text)

        print("Predicting probabilities for U")
        ypU_old = ypU
        ypU = new_model.predict_proba(U)

        predU = [round(p) for p in ypU]
        pos_ratio = sum(predU) / len(U)

        print("Unlabelled instances classified as positive:", sum(predU), "/", len(U),
              "(", pos_ratio * 100, "%)\n")

        if clf_selection and old_model is not None:
            if em_getting_worse(old_model, new_model, P, U):
                print("Approximated error has grown since last iteration.\n"
                      "Aborting and returning classifier #", iterations - 1)
                return old_model

        if pos_ratio >= max_pos_ratio:
            print("Acceptable ratio of positively labelled sentences in U is reached.")
            break

    print("Returning final model after", iterations, "iterations")
    return new_model


def iterate_SVM(P, U, RN, text=True, max_neg_ratio=0.05):
    """runs an SVM classifier trained on P and RN iteratively, augmenting RN

    after each iteration, the documents in U classified as negative are moved to RN until there are none left.
    max_neg_ratio is the maximum accepted ratio of P to be classified as negative by final classifier.
    if the final classifier regards more than max_neg_ratio of P as negative, return the initial one."""

    y_P = np.ones(num_rows(P))
    y_RN = np.zeros(num_rows(RN))

    print("Building initial SVM classifier with Positive and Reliable Negative docs")
    initial_model = build_and_evaluate(np.concatenate((P, RN)),
                                       np.concatenate((y_P, y_RN)),
                                       classifier=svm.SVC(kernel='linear', class_weight='balanced', probability=True),
                                       text=text)

    print("Predicting U with initial SVM, adding negatively classified docs to RN for iteration")
    y_U = initial_model.predict(U)
    Q, W = partition_pos_neg(U, y_U)
    iteration = 0
    model = None

    # iterate SVM, each turn augmenting RN by the documents in Q classified negative
    while num_rows(W) > 0 and num_rows(Q) > 0:
        print("new negative docs:", num_rows(W))
        iteration += 1
        print("Iteration #", iteration)

        RN = np.concatenate((RN, W))
        y_RN = np.zeros(num_rows(RN))

        model = build_and_evaluate(np.concatenate((P, RN)),
                                   np.concatenate((y_P, y_RN)),
                                   classifier=svm.SVC(kernel='linear', class_weight='balanced', probability=True),
                                   text=text)
        y_U = model.predict(Q)
        Q, W = partition_pos_neg(Q, y_U)

    print("Iterative SVM converged. Positive examples remaining in U:", num_rows(Q), "(",
          100 * num_rows(Q) / num_rows(U), "%)")

    y_P_initial = initial_model.predict(P)
    initial_neg_ratio = 1 - np.average(y_P_initial)
    print("Ratio of positive samples classified as negative by initial SVM:", initial_neg_ratio)

    if model is None:
        return initial_model

    y_P_final = model.predict(P)
    final_neg_ratio = 1 - np.average(y_P_final)
    print("Ratio of positive samples classified as negative by final SVM:", final_neg_ratio)

    if final_neg_ratio > max_neg_ratio:
        print("Final classifier discards too many positive examples.")
        print("Returning initial classifier instead")
        return initial_model

    print("Returning final classifier")
    return model


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------

def almost_equal(probas1, probas2, tolerance=0.1):
    """helper function that checks if vectors of probabilistic labels are similar"""
    return np.array_equiv(np.round(probas1), np.round(probas2))

    # # below: somewhat more precise, slower convergence criterion
    # zipped = zip(probas1, probas2)
    # diffs = [abs(p1 - p2) <= tolerance
    #          for p1, p2 in zipped]
    # return all(diffs)


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


def em_getting_worse(old_model, new_model, P, U):
    """calculates approximated change in probability of error for iterative EM

    should be used in S-EM, but not in I-EM,
    according to \"Partially Supervised Classification of Text Documents\""""

    # probability of error:
    # Pr[f(X) =\= Y] = Pr[f(X) = 1] - Pr[Y = 1] + 2 * Pr[f(X) = 0 | Y = 1] * Pr[Y = 1]
    # change in probability of error has to be approximated since ground truth is unavailable:
    # Delta_i = Pr_U

    # predict P and U with old models to compare predicted class distributions
    y_P_old = old_model.predict(P)
    y_U_old = old_model.predict(U)

    y_P_new = new_model.predict(P)
    y_U_new = new_model.predict(U)

    Pr_U_pos_old = num_rows(y_U_old[y_U_old == 1])
    Pr_P_neg_old = num_rows(y_P_old[y_P_old == 0])

    Pr_U_pos_new = num_rows(y_U_new[y_U_new == 1])
    Pr_P_neg_new = num_rows(y_P_new[y_P_new == 0])

    Delta_i = (Pr_U_pos_new - Pr_U_pos_old
               + 2 * (Pr_P_neg_new - Pr_P_neg_old) * Pr_U_pos_old)

    print("Delta_i:", Delta_i)

    return Delta_i > 0


def choose_noise_lvl(sims_P, sims_U):
    """selects the best percentage of assumed noise in terms of PU score (r_P^2 / Pr_{P+U} [f(X)=1])"""
    lvls = [0.01 * x for x in range(50)]
    scores = []

    for lvl in lvls:
        U_minus_PN, PN = select_PN_below_score(y_pos=sims_P, U=sims_U, y_U=sims_U,
                                               noise_lvl=lvl)
        y_U = [1] * num_rows(U_minus_PN) + \
              [0] * num_rows(PN)
        y_P = [1] * int((1 - lvl) * num_rows(sims_P)) + \
              [0] * int((lvl) * num_rows(sims_P))

        scores.append(pu_measure(y_P, y_U))

    [print(x) for x in zip(lvls, scores)]

    return lvls[np.argmax(scores)]


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