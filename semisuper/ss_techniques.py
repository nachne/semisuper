from __future__ import absolute_import, division, print_function

import multiprocessing as multi
import random

import numpy as np
from sklearn import semi_supervised
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from semisuper import pu_two_step
from semisuper.helpers import num_rows, partition_pos_neg, partition_pos_neg_unsure, arrays, concatenate, densify
from semisuper.proba_label_nb import build_proba_MNB
from semisuper.pu_two_step import almost_equal


# ----------------------------------------------------------------
#
# ----------------------------------------------------------------


def self_training(P, N, U, clf=None, confidence=0.75, verbose=False):
    """Generic Self-Training with optional classifier (must implement predict_proba) and confidence threshold.
    Default: Logistic Regression"""

    print("Running standard Self-Training with confidence threshold", confidence,
          "and classifier", (clf or "Logistic Regression"))

    if verbose:
        print("Training initial classifier")

    if clf is not None:
        if isinstance(clf, type):
            model = clf()
        else:
            model = clf
    else:
        model = LogisticRegression(solver='sag', C=1.0)
    model.fit(concatenate((P, N)), concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

    if hasattr(model, 'predict_proba'):
        ypU = model.predict_proba(U)
    else:
        ypU = model.decision_function(U)
        ypU = np.vstack((-ypU, ypU)).T
    RP, RN, U = partition_pos_neg_unsure(U, ypU, confidence)

    iteration = 0

    while np.size(RP) or np.size(RN):
        iteration += 1
        if verbose:
            print("Iteration #", iteration, "\tRP", num_rows(RP), "\tRN", num_rows(RN), "\tunclear:", num_rows(U))
        P = concatenate((P, RP)) if np.size(RP) else P
        N = concatenate((N, RN)) if np.size(RN) else N

        model.fit(concatenate((P, N)), concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

        if not np.size(U):
            break

        if hasattr(model, 'predict_proba'):
            ypU = model.predict_proba(U)
        else:
            ypU = model.decision_function(U)
            ypU = np.vstack((-ypU, ypU)).T
        RP, RN, U = partition_pos_neg_unsure(U, ypU, confidence)

    print("Returning final model after", iteration, "iterations.")
    return model


def self_training_lin_svc(P, N, U, confidence=0.5, clf=None, verbose=False):
    print("Running standard Self-Training with confidence threshold", confidence,
          "and classifier", (clf or "LinearSVC"))

    if verbose:
        print("Training initial classifier")

    if clf is not None:
        if isinstance(clf, type):
            model = clf()
        else:
            model = clf
    else:
        model = LinearSVC(C=1.0, class_weight='balanced')
    model.fit(concatenate((P, N)), concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

    ypU = model.decision_function(U)
    ypU = np.vstack((-ypU, ypU)).T
    RP, RN, U = partition_pos_neg_unsure(U, ypU, confidence)

    iteration = 0

    while np.size(RP) or np.size(RN):
        iteration += 1
        if verbose:
            print("Iteration #", iteration, "\tRP", num_rows(RP), "\tRN", num_rows(RN), "\tunclear:", num_rows(U))
        P = concatenate((P, RP)) if np.size(RP) else P
        N = concatenate((N, RN)) if np.size(RN) else N

        model.fit(concatenate((P, N)), concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

        if not np.size(U):
            break

        ypU = model.decision_function(U)
        ypU = np.vstack((-ypU, ypU)).T
        RP, RN, U = partition_pos_neg_unsure(U, ypU, confidence)

    print("Returning final model after", iteration, "iterations.")
    return model


def neg_self_training(P, N, U, clf=None, verbose=False):
    """Iteratively augment negative set. Optional classifier (must implement predict)
    Default: Logistic Regression"""

    print("Iteratively augmenting negative set with", (clf or "Logistic Regression"), "classifier")

    if verbose:
        print("Training initial classifier")

    if clf is not None:
        if isinstance(clf, type):
            model = clf()
        else:
            model = clf
    else:
        model = LogisticRegression(solver='sag', C=1.0)

    model.fit(concatenate((P, N)), concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

    ypU = model.predict(U)
    U, RN = partition_pos_neg(U, ypU)

    iteration = 0

    while np.size(RN) and np.size(U):
        iteration += 1
        if verbose:
            print("Iteration #", iteration, "\tRN", num_rows(RN), "\tremaining:", num_rows(U))

        N = concatenate((N, RN))

        model.fit(concatenate((P, N)), concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

        if not np.size(U):
            break

        ypU = model.predict(U)
        U, RN = partition_pos_neg(U, ypU)

    if np.size(RN):
        N = concatenate((N, RN))
        model.fit(concatenate((P, N)), concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

    print("Returning final model after", iteration, "iterations.")
    return model


def iterate_linearSVC(P, N, U, C=1.0, verbose=False):
    """run SVM iteratively until labels for U converge"""

    print("Running iterative linear SVM with C = ", C)

    return pu_two_step.iterate_SVM(P=P, U=U, RN=N, C=C,
                                   kernel=None,
                                   max_neg_ratio=0.1, clf_selection=False, verbose=verbose)


def EM(P, N, U, ypU=None, max_pos_ratio=1.0, tolerance=0.05, max_imbalance_P_N=10.0, verbose=False):
    """Iterate EM until estimates for U converge.

    Train NB with P and N to get probabilistic labels for U, or use assumed priors if passed as parameter"""

    print("Running EM")

    if num_rows(P) > max_imbalance_P_N * num_rows(N):
        P_init = np.array(random.sample(list(P), int(max_imbalance_P_N * num_rows(N))))
    else:
        P_init = P

    if ypU is None:
        if verbose:
            print("\nBuilding classifier from Positive and Reliable Negative set")
        initial_model = build_proba_MNB(concatenate((P_init, N)),
                                        concatenate((np.ones(num_rows(P_init)), np.zeros(num_rows(N)))))

        if verbose:
            print("\nCalculating initial probabilistic labels for Unlabelled set")
        ypU = initial_model.predict_proba(U)[:, 1]
    else:
        print("Using assumed probabilities/weights for initial probabilistic labels of Unlabelled set")

    if verbose:
        print("\nIterating EM algorithm on P, N, and U\n")
    model = iterate_EM_PNU(P=P, N=N, U=U, ypU=ypU, tolerance=tolerance, max_pos_ratio=max_pos_ratio, verbose=verbose)

    return model


def iterate_EM_PNU(P, N, U, y_P=None, y_N=None, ypU=None, tolerance=0.05, max_pos_ratio=1.0, verbose=False):
    """EM algorithm for positive set P and unlabelled set U

        iterate NB classifier with updated labels for unlabelled set (with optional initial labels) until convergence"""

    if y_P is None:
        y_P = ([1.] * num_rows(P))
    if y_N is None:
        y_N = ([0.] * num_rows(N))
    if ypU is None:
        ypU = ([0.] * num_rows(U))

    ypU_old = [-999]

    iterations = 0
    new_model = None

    while not almost_equal(ypU_old, ypU, tolerance):

        iterations += 1

        if verbose:
            print("Iteration #", iterations, "\tBuilding new model using probabilistic labels")

        new_model = build_proba_MNB(concatenate((P, N, U)),
                                    concatenate((y_P, y_N, ypU)))

        if verbose:
            print("Predicting probabilities for U")
        ypU_old = ypU
        ypU = new_model.predict_proba(U)[:, 1]

        predU = [round(p) for p in ypU]
        pos_ratio = sum(predU) / num_rows(U)

        if verbose:
            print("Unlabelled instances classified as positive:", sum(predU), "/", num_rows(U),
                  "(", pos_ratio * 100, "%)\n")

        if pos_ratio >= max_pos_ratio:
            if verbose:
                print("Acceptable ratio of positively labelled sentences in U is reached.")
            break

    if verbose:
        print("Returning final model after", iterations, "iterations")
    return new_model


def iterate_knn(P, N, U, n_neighbors=7, thresh=0.6, verbose=False):
    p_init, n_init = num_rows(P), num_rows(N)
    P, N, U = densify(P), densify(N), densify(U)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', n_jobs=multi.cpu_count() - 1)
    knn.fit(concatenate((P, N)), concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

    y_pred = knn.predict_proba(U)
    U_pos, U_neg, U = partition_pos_neg_unsure(U, y_pred, confidence=thresh)
    i = 0

    while num_rows(U_pos) and num_rows(U_neg) and num_rows(U):
        if verbose:
            print("Iteration #", i)
            print("New confidently predicted examples: \tpos", num_rows(U_pos), "\tneg", num_rows(U_neg))
            print("Remaining unlabelled:", num_rows(U))

        P = concatenate((P, U_pos))
        N = concatenate((N, U_neg))

        knn.fit(concatenate((P, N)), concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

        y_pred = knn.predict_proba(U)
        U_pos, U_neg, U = partition_pos_neg_unsure(U, y_pred, confidence=thresh)
        i += 1

    print("knn converged with", num_rows(U), "sentences remaining unlabelled after", i, "iterations.",
          "\nLabelled pos:", num_rows(P) - p_init,
          "\tneg:", num_rows(N) - n_init,
          "Returning classifier")
    return knn


# slow
def iterate_SVC(P, N, U, kernel="rbf", verbose=False):
    """run SVM iteratively until labels for U converge"""

    print("Running iterative SVM with", kernel, "kernel")

    return pu_two_step.iterate_SVM(P=P, U=U, RN=N,
                                   kernel=kernel,
                                   max_neg_ratio=0.1, clf_selection=False, verbose=verbose)


# horrible results!
def label_propagation(P, N, U,
                      method="propagation", kernel='knn', n_neighbors=7, max_iter=30,
                      n_jobs=-1):
    """wrapper for sklearn's LabelPropagation/LabelPropagation avoiding sparse matrix errors"""

    if method == "propagation":
        clf = semi_supervised.LabelPropagation(kernel=kernel, n_neighbors=n_neighbors, max_iter=max_iter, n_jobs=n_jobs)
    else:
        clf = semi_supervised.LabelSpreading(kernel=kernel, n_neighbors=n_neighbors, max_iter=max_iter, n_jobs=n_jobs)

    X = concatenate((P, N, U))
    y_init = concatenate((np.ones(num_rows(P)),
                          np.zeros(num_rows(N)),
                          -np.ones(num_rows(U))))

    class propagation():
        def __init__(self, clf):
            # self.clf = semi_supervised.LabelPropagation()
            self.clf = clf
            return

        def fit(self, X, y):
            self.clf.fit(densify(X), y)
            return self

        def predict(self, X):
            return self.clf.predict(densify(X))

        def predict_proba(self, X):
            return self.clf.predict_proba(densify(X))

        def score(self, X, y):
            return self.clf.score(densify(X), y)

    lp = propagation(clf).fit(X, y_init)

    return lp


# ----------------------------------------------------------------
# versions with flipped parameters for partial application
# ----------------------------------------------------------------


def label_propagation_method(method, kernel, P, N, U):
    return label_propagation(P, N, U, method, kernel)


def iterate_linearSVC_C(C, P, N, U):
    return iterate_linearSVC(P, N, U, C=C)


def neg_self_training_clf(clf, P, N, U):
    return neg_self_training(P, N, U, clf)


def self_training_clf_conf(clf, confidence, P, N, U):
    return self_training(P, N, U, clf, confidence)


# ----------------------------------------------------------------
# wrappers for supervised classifier to work with P, N, and U parameters for comparison
# ----------------------------------------------------------------

# TODO move to own script for supervised stuff

def nb(P, N, U=None, verbose=False):
    X = concatenate((P, N))
    y = concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = GridSearchCV(estimator=MultinomialNB(),
                         param_grid={
                             'alpha': [x / 10.0 for x in range(1, 11)],
                         },
                         ).fit(X, y)
    print("Best hyperparameters for Naive Bayes:", model.best_params_)
    print("Grid search score:", model.best_score_)
    return model.best_estimator_


def logreg(P, N, U=None, verbose=False):
    X = concatenate((P, N))
    y = concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = GridSearchCV(estimator=LogisticRegression(),
                         param_grid={
                             'C'           : [x for x in range(1, 11)],
                             'solver'      : ['sag'],
                             'class_weight': ['balanced']
                         },
                         ).fit(X, y)
    print("Best hyperparameters for Logistic Regression:", model.best_params_)
    print("Grid search score:", model.best_score_)
    return model.best_estimator_


def sgd(P, N, U, loss="modified_huber", verbose=False):
    X = concatenate((P, N))
    y = concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = SGDClassifier(loss=loss, class_weight="balanced").fit(X, y)
    return model


def grid_search_linearSVM(P, N, U, verbose=False):
    model = LinearSVC()

    grid_search = GridSearchCV(model,
                               param_grid={'C'           : [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0],
                                           'class_weight': ['balanced'],
                                           'loss'        : ['squared_hinge'],
                                           },
                               cv=3,
                               n_jobs=2,  # min(multi.cpu_count(), 16),
                               verbose=0)

    if verbose:
        print("Grid searching parameters for Linear SVC")
    X = concatenate((P, N))
    y = concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))

    grid_search.fit(X, y)

    print("SVC parameters:", grid_search.best_params_, "\tscore:", grid_search.best_score_)

    return grid_search.best_estimator_
