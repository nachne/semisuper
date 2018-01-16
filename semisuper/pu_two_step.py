import random
from multiprocessing import cpu_count
import warnings

import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier

from semisuper.helpers import num_rows, arrays, partition_pos_neg, pu_score, train_report, select_PN_below_score
from semisuper.proba_label_nb import build_proba_MNB
from semisuper.pu_cos_roc import ranking_cos_sim, rocchio


# ----------------------------------------------------------------
# COMPLETE 2-STEP METHODS
# ----------------------------------------------------------------


def cr_SVM(P, U, max_neg_ratio=0.1, noise_lvl=0.2, alpha=16, beta=4, kernel=None, C=0.1, verbose=False):
    """Two-Step technique based on Cosine Similarity, Rocchio and SVM


    Step 1.1: Find Potentially Negative docs (less similar to mean(P) than noise_lvl of docs in P)
    Step 1.2: Find Reliable Negative docs using Rocchio (similarity to mean positive/PN vector)
    Step 2: Iterate SVM starting from P and RN sets until classification of U converges

    noise level is quite crucial, should be >=20% to give reasonable results"""

    print("Running CR-SVM")

    # step 1
    if verbose: print("Determining RN using Cosine Similarity threshold and Rocchio\n")
    U_minus_RN, RN = get_RN_cosine_rocchio(P, U, noise_lvl=noise_lvl, alpha=alpha, beta=beta,
                                           verbose=verbose)

    # step2
    if verbose: print("\nIterating SVM with P, U-RN, and RN")
    model = iterate_SVM(P, U_minus_RN, RN, kernel=kernel, C=C, max_neg_ratio=max_neg_ratio, verbose=verbose)

    if verbose: train_report(model, P, U)

    return model


# TODO: good default C param
def roc_SVM(P, U, max_neg_ratio=0.1, alpha=16, beta=4, kernel=None, C=0.1, verbose=False):
    """Two-Step technique based on Rocchio and SVM

    Step 1: Find Reliable Negative docs using Rocchio (similarity to mean positive/unlabelled vector)
    Step 2: Iterate SVM starting from P and RN sets until classification of U converges"""

    print("Running Roc-SVM")

    # step 1
    if verbose: print("Determining RN using Rocchio method\n")
    U_minus_RN, RN = get_RN_rocchio(P, U, alpha=alpha, beta=beta, verbose=verbose)

    # step2
    if verbose: print("\nIterating SVM with P, U-RN, and RN")
    model = iterate_SVM(P, U_minus_RN, RN, kernel=kernel, C=C, max_neg_ratio=max_neg_ratio, verbose=verbose)

    if verbose: train_report(model, P, U)

    return model


def s_EM(P, U, spy_ratio=0.1, max_pos_ratio=1.0, tolerance=0.1, noise_lvl=0.1, clf_selection=True, verbose=False):
    """S-EM two-step PU learning as described in \"Partially Supervised Classification...\".

    1st step: get Reliable Negative documents using Spy Documents
    2nd step: iterate EM with P, U-RN, and RN
    """

    print("Running S-EM")

    # step 1
    if verbose: print("Determining confidence threshold using Spy Documents and I-EM\n")
    U_minus_RN, RN = get_RN_Spy_Docs(P, U,
                                     spy_ratio=spy_ratio, tolerance=tolerance, noise_lvl=noise_lvl,
                                     verbose=verbose)

    # step2
    if verbose: print("\nIterating I-EM with P, U-RN, and RN")
    model = run_EM_with_RN(P, U_minus_RN, RN,
                           tolerance=tolerance, max_pos_ratio=max_pos_ratio,
                           clf_selection=clf_selection, verbose=verbose)

    if verbose: train_report(model, P, U)

    return model


def i_EM(P, U, max_imbalance=10.0, max_pos_ratio=1.0, tolerance=0.1, verbose=False):
    """all-in-one PU method: I-EM algorithm for positive set P and unlabelled set U

    iterate NB classifier with updated labels for unlabelled set (initially negative) until convergence
    if U is much larger than P, randomly samples max_imbalance*|P| docs from U"""

    print("Running I-EM")

    # if num_rows(U) > max_imbalance * num_rows(P):
    #     U = np.array(random.sample(list(U), int(max_imbalance * num_rows(P))))

    model = iterate_EM(P, U, tolerance=tolerance, max_pos_ratio=max_pos_ratio, clf_selection=False, verbose=verbose)

    if verbose: train_report(model, P, U)

    return model


# ----------------------------------------------------------------
# OTHER COMBINATIONS
# ----------------------------------------------------------------

def standalone_rocchio(P, U, alpha=16, beta=4, verbose=False):
    """1-step Rocchio method"""

    print("Running Rocchio")

    if verbose: print("Building Rocchio model to determine Reliable Negative examples")
    model = rocchio(P, U, alpha=alpha, beta=beta)

    y_U = model.predict(U)

    U_minus_RN, RN = partition_pos_neg(U, y_U)
    if verbose:
        print("Reliable Negative examples in U:", num_rows(RN), "(", 100 * num_rows(RN) / num_rows(U), "%)")
        train_report(model, P, U)

    return model


def spy_SVM(P, U, spy_ratio=0.1, max_neg_ratio=0.1, tolerance=0.1, noise_lvl=0.1, verbose=False):
    """S-EM two-step PU learning as described in \"Partially Supervised Classification...\".

    1st step: get Reliable Negative documents using Spy Documents
    2nd step: iterate EM with P, U-RN, and RN
    """

    print("Running Spy-SVM")

    # step 1
    if verbose: print("Determining confidence threshold using Spy Documents and I-EM\n")
    U_minus_RN, RN = get_RN_Spy_Docs(P, U,
                                     spy_ratio=spy_ratio, tolerance=tolerance, noise_lvl=noise_lvl,
                                     verbose=verbose)

    # step2
    if verbose: print("\nIterating SVM with P, U-RN, and RN")
    model = iterate_SVM(P, U_minus_RN, RN, max_neg_ratio=max_neg_ratio, verbose=verbose)

    if verbose: train_report(model, P, U)

    return model


def roc_EM(P, U, max_pos_ratio=0.5, tolerance=0.1, clf_selection=True,
           alpha=16, beta=4, verbose=False):
    """S-EM two-step PU learning as described in \"Partially Supervised Classification...\".

    1st step: get Reliable Negative documents using Spy Documents
    2nd step: iterate EM with P, U-RN, and RN
    """

    print("Running Roc-EM")

    # step 1
    if verbose: print("Determining RN using Rocchio method\n")
    U_minus_RN, RN = get_RN_rocchio(P, U, alpha=alpha, beta=beta, verbose=verbose)

    # step2
    if verbose: print("\nIterating I-EM with P, U-RN, and RN")
    model = run_EM_with_RN(P, U_minus_RN, RN,
                           tolerance=tolerance, max_pos_ratio=max_pos_ratio,
                           clf_selection=clf_selection, verbose=verbose)

    if verbose: train_report(model, P, U)

    return model


# ----------------------------------------------------------------
# FIRST STEP TECHNIQUES
# ----------------------------------------------------------------

def get_RN_Spy_Docs(P, U, spy_ratio=0.1, max_pos_ratio=0.5, tolerance=0.2, noise_lvl=0.05, verbose=False):
    """First step technique: Compute reliable negative docs from P using Spy Documents and I-EM"""

    P_minus_spies, spies = spy_partition(P, spy_ratio)
    U_plus_spies = np.concatenate((U, spies))

    model = iterate_EM(P_minus_spies, U_plus_spies, tolerance=tolerance, max_pos_ratio=max_pos_ratio,
                       clf_selection=False, verbose=verbose)

    y_spies = model.predict_proba(spies)[:, 1]
    y_U = model.predict_proba(U)[:, 1]

    U_minus_RN, RN = select_PN_below_score(y_spies, U, y_U, noise_lvl=noise_lvl)

    return U_minus_RN, RN


def get_RN_rocchio(P, U, alpha=16, beta=4, verbose=False):
    """extract Reliable Negative documents using Binary Rocchio algorithm"""

    if verbose: print("Building Rocchio model to determine Reliable Negative examples")
    model = rocchio(P, U, alpha=alpha, beta=beta)

    y_U = model.predict(U)

    U_minus_RN, RN = partition_pos_neg(U, y_U)
    if verbose: print("Reliable Negative examples in U:", num_rows(RN), "(", 100 * num_rows(RN) / num_rows(U), "%)")

    return U_minus_RN, RN


def get_RN_cosine_rocchio(P, U, noise_lvl=0.20, alpha=16, beta=4, verbose=False):
    """extract Reliable Negative documents using cosine similarity and Binary Rocchio algorithm

    similarity is the cosine similarity compared to the mean positive sample.
    firstly, select Potential Negative docs that have lower similarity than the worst l% in P.
    source: negative harmful
    """

    if verbose: print("Computing ranking (cosine similarity to mean positive example)")
    mean_p_ranker = ranking_cos_sim(P)

    sims_P = mean_p_ranker.predict_proba(P)
    sims_U = mean_p_ranker.predict_proba(U)

    if verbose: print("Choosing Potential Negative examples with ranking threshold")
    _, PN = select_PN_below_score(sims_P, U, sims_U, noise_lvl=noise_lvl, verbose=verbose)

    if verbose: print("Building Rocchio model to determine Reliable Negative examples")
    model = rocchio(P, PN, alpha=alpha, beta=beta)

    y_U = model.predict(U)

    U_minus_RN, RN = partition_pos_neg(U, y_U)
    if verbose: print("Reliable Negative examples in U:", num_rows(RN), "(", 100 * num_rows(RN) / num_rows(U), "%)")

    return U_minus_RN, RN


# ----------------------------------------------------------------
# SECOND STEP TECHNIQUES
# ----------------------------------------------------------------

def run_EM_with_RN(P, U, RN, max_pos_ratio=1.0, tolerance=0.05, max_imbalance_P_RN=10.0, clf_selection=True,
                   verbose=False):
    """second step PU method: train NB with P and RN to get probabilistic labels for U, then iterate EM"""

    if num_rows(P) > max_imbalance_P_RN * num_rows(RN):
        P_init = np.array(random.sample(list(P), int(max_imbalance_P_RN * num_rows(RN))))
    else:
        P_init = P

        if verbose: print("\nBuilding classifier from Positive and Reliable Negative set")
    initial_model = build_proba_MNB(np.concatenate((P_init, RN)),
                                    np.concatenate((np.ones(num_rows(P_init)),
                                                    np.zeros(num_rows(RN)))),
                                    verbose=verbose)

    if num_rows(U) == 0:
        print("Warning: EM: All of U was classified as negative.")
        return initial_model

    y_P = np.array([1] * num_rows(P))

    if verbose: print("\nCalculating initial probabilistic labels for Reliable Negative and Unlabelled set")
    ypU = initial_model.predict_proba(U)[:, 1]
    ypN = initial_model.predict_proba(RN)[:, 1]

    if verbose: print("\nIterating EM algorithm on P, RN and U\n")
    model = iterate_EM(P, np.concatenate((RN, U)),
                       y_P, np.concatenate((ypN, ypU)),
                       tolerance=tolerance, max_pos_ratio=max_pos_ratio,
                       clf_selection=clf_selection, verbose=verbose)

    return model


def iterate_EM(P, U, y_P=None, ypU=None, tolerance=0.05, max_pos_ratio=1.0, clf_selection=False, verbose=False):
    """EM algorithm for positive set P and unlabelled set U

        iterate NB classifier with updated labels for unlabelled set (with optional initial labels) until convergence"""

    if y_P is None:
        y_P = np.ones(num_rows(P))
    if ypU is None:
        ypU = np.zeros(num_rows(U))

    ypU_old = [-999]

    iterations = 0
    old_model = None
    new_model = None

    while not almost_equal(ypU_old, ypU, tolerance):

        iterations += 1

        if verbose: print("Iteration #", iterations, "\tBuilding new model using probabilistic labels")

        if clf_selection:
            old_model = new_model

        new_model = build_proba_MNB(np.concatenate((P, U)),
                                    np.concatenate((y_P, ypU)), verbose=verbose)

        if verbose: print("Predicting probabilities for U")

        ypU_old = ypU
        ypU = new_model.predict_proba(U)[:, 1]

        predU = [round(p) for p in ypU]
        pos_ratio = sum(predU) / len(U)

        if verbose: print("Unlabelled instances classified as positive:", sum(predU), "/", len(U),
                          "(", pos_ratio * 100, "%)\n")

        if clf_selection and old_model is not None:
            if em_getting_worse(old_model, new_model, P, U):
                if verbose: print("Approximated error has grown since last iteration.\n"
                                  "Aborting and returning classifier #", iterations - 1)
                return old_model

        if pos_ratio >= max_pos_ratio:
            if verbose: print("Acceptable ratio of positively labelled sentences in U is reached.")
            break

    print("Returning final NB after", iterations, "iterations")
    return new_model


def iterate_SVM(P, U, RN, max_neg_ratio=0.2, clf_selection=True, kernel=None, C=0.1, n_estimators=9, verbose=False):
    """runs an SVM classifier trained on P and RN iteratively, augmenting RN

    after each iteration, the documents in U classified as negative are moved to RN until there are none left.
    max_neg_ratio is the maximum accepted ratio of P to be classified as negative by final classifier.
    if clf_selection is true and the final classifier regards more than max_neg_ratio of P as negative,
    return the initial one."""

    y_P = np.ones(num_rows(P))
    y_RN = np.zeros(num_rows(RN))

    if kernel is not None:
        if verbose: print("Building initial Bagging SVC (", n_estimators, "clfs)",
                          "with Positive and Reliable Negative docs")
        clf = (
            BaggingClassifier(
                    svm.SVC(class_weight='balanced', kernel=kernel, C=C)
                    , bootstrap=True, n_estimators=n_estimators, n_jobs=min(n_estimators, cpu_count()),
                    max_samples=(1.0 if n_estimators < 4 else 1.0 / (n_estimators - 2))
            )
        )
    else:
        if verbose: print("Building initial linearSVM classifier with Positive and Reliable Negative docs")
        clf = svm.LinearSVC(class_weight='balanced', C=C)

    initial_model = clf.fit(np.concatenate((P, RN)), np.concatenate((y_P, y_RN)))

    if num_rows(U) == 0:
        print("Warning: SVM: All of U was classified as negative.")
        return initial_model

    if verbose: print("Predicting U with initial SVM, adding negatively classified docs to RN for iteration")

    y_U = initial_model.predict(U)
    Q, W = partition_pos_neg(U, y_U)
    iteration = 0
    model = None

    if num_rows(Q) == 0 or num_rows(W) == 0:
        print("Warning: Returning initial SVM because all of U was assigned label", y_U[0])
        return initial_model

    # iterate SVM, each turn augmenting RN by the documents in Q classified negative
    while num_rows(W) > 0 and num_rows(Q) > 0:
        iteration += 1

        RN = np.concatenate((RN, W))
        y_RN = np.zeros(num_rows(RN))

        if verbose: print("\nIteration #", iteration, "\tReliable negative examples:", num_rows(RN))

        if kernel is not None:
            clf = (BaggingClassifier(
                    svm.SVC(class_weight='balanced', kernel=kernel, C=C)
                    , bootstrap=True, n_estimators=n_estimators, n_jobs=min(n_estimators, cpu_count()),
                    max_samples=(1.0 if n_estimators < 4 else 1.0 / (n_estimators - 2))
            )
            )
        else:
            clf = svm.LinearSVC(class_weight='balanced', C=C)

        model = clf.fit(np.concatenate((P, RN)), np.concatenate((y_P, y_RN)))
        y_U = model.predict(Q)
        Q, W = partition_pos_neg(Q, y_U)

    RN = np.concatenate((RN, W))
    model = clf.fit(np.concatenate((P, RN)), np.concatenate((y_P, y_RN)))

    if verbose: print("Iterative SVM converged. Reliable negative examples:", num_rows(RN))

    if clf_selection:

        y_P_initial = initial_model.predict(P)
        initial_neg_ratio = 1 - np.average(y_P_initial)

        if verbose: print("Ratio of positive examples misclassified as negative by initial SVM:", initial_neg_ratio)

        if model is None: return initial_model

        y_P_final = model.predict(P)
        final_neg_ratio = 1 - np.average(y_P_final)

        if verbose: print("Ratio of positive examples misclassified as negative by final SVM:", final_neg_ratio)

        if final_neg_ratio > max_neg_ratio and final_neg_ratio > initial_neg_ratio:
            print(iteration, "iterations - final SVM discards too many positive examples.",
                  "Returning initial SVM instead")

            return initial_model

    print("Returning final SVM after", iteration, "iterations")
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
    idx = random.sample(range(num_P), num_idx)
    spies = P[idx]

    # define rest partition
    mask = np.ones(num_P, dtype=bool)
    mask[idx] = False
    P_minus_spies = P[mask]

    return P_minus_spies, spies


def em_getting_worse(old_model, new_model, P, U, verbose=False):
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

    if verbose: print("Delta_i:", Delta_i)

    return Delta_i > 0
