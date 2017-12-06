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
from semisuper.helpers import num_rows, partition_pos_neg, partition_pos_neg_unsure, arrays
from semisuper.proba_label_nb import build_proba_MNB
from semisuper.pu_two_step import almost_equal


# ----------------------------------------------------------------
# top level
# ----------------------------------------------------------------


def self_training(P, N, U, confidence=0.8, clf=None, verbose=True):
    """Generic Self-Training with optional classifier (must implement predict_proba) and confidence threshold.
    Default: Logistic Regression"""

    print("Running standard Self-Training with confidence threshold", confidence,
          "and classifier", (clf or "Logistic Regression"))

    if verbose: print("Training initial classifier")

    if clf is not None:
        if isinstance(clf, type):
            model = clf()
        else:
            model = clf
    else:
        model = LogisticRegression(solver='sag', C=1.0)
    model.fit(np.concatenate((P, N)), np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

    ypU = model.predict_proba(U)
    RP, RN, U = partition_pos_neg_unsure(U, ypU, confidence)

    iteration = 0

    while np.size(RP) or np.size(RN):
        iteration += 1
        if verbose:
            print("Iteration #", iteration, "\tRP", num_rows(RP), "\tRN", num_rows(RN), "\tunclear:", num_rows(U))
        P = np.concatenate((P, RP)) if np.size(RP) else P
        N = np.concatenate((N, RN)) if np.size(RN) else N

        model.fit(np.concatenate((P, N)), np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

        if not np.size(U):
            break

        ypU = model.predict_proba(U)
        RP, RN, U = partition_pos_neg_unsure(U, ypU, confidence)

    print("Returning final model after", iteration, "iterations.")
    return model


def self_training_lin_svc(P, N, U, confidence=0.5, clf=None, verbose=True):
    print("Running standard Self-Training with confidence threshold", confidence,
          "and classifier", (clf or "LinearSVC"))

    if verbose: print("Training initial classifier")

    if clf is not None:
        if isinstance(clf, type):
            model = clf()
        else:
            model = clf
    else:
        model = LinearSVC(C=1.0)
    model.fit(np.concatenate((P, N)), np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

    ypU = model.decision_function(U)
    ypU = np.vstack((-ypU, ypU)).T
    RP, RN, U = partition_pos_neg_unsure(U, ypU, confidence)

    iteration = 0

    while np.size(RP) or np.size(RN):
        iteration += 1
        if verbose:
            print("Iteration #", iteration, "\tRP", num_rows(RP), "\tRN", num_rows(RN), "\tunclear:", num_rows(U))
        P = np.concatenate((P, RP)) if np.size(RP) else P
        N = np.concatenate((N, RN)) if np.size(RN) else N

        model.fit(np.concatenate((P, N)), np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

        if not np.size(U):
            break

        ypU = model.decision_function(U)
        ypU = np.vstack((-ypU, ypU)).T
        RP, RN, U = partition_pos_neg_unsure(U, ypU, confidence)

    print("Returning final model after", iteration, "iterations.")
    return model


def neg_self_training(P, N, U, clf=None, verbose=True):
    """Iteratively augment negative set. Optional classifier (must implement predict_proba) and confidence threshold.
    Default: Logistic Regression"""

    print("Iteratively augmenting negative set with", (clf or "Logistic Regression"), "classifier")

    if verbose: print("Training initial classifier")

    if clf is not None:
        if isinstance(clf, type):
            model = clf()
        else:
            model = clf
    else:
        model = LogisticRegression(solver='sag', C=1.0)

    model.fit(np.concatenate((P, N)), np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

    ypU = model.predict(U)
    U, RN = partition_pos_neg(U, ypU)

    iteration = 0

    while np.size(RN):
        iteration += 1
        if verbose:
            print("Iteration #", iteration, "\tRN", num_rows(RN), "\tremaining:", num_rows(U))

        N = np.concatenate((N, RN)) if np.size(RN) else N

        model.fit(np.concatenate((P, N)), np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N)))))

        if not np.size(U):
            break

        ypU = model.predict(U)
        U, RN = partition_pos_neg(U, ypU)

    print("Returning final model after", iteration, "iterations.")
    return model


neg_self_training_logit = neg_self_training


def neg_self_training_sgd(P, N, U, loss="modified_huber", n_jobs=min(16, multi.cpu_count()), verbose=True):
    return neg_self_training(P, N, U, clf=SGDClassifier(loss=loss, n_jobs=n_jobs), verbose=verbose)


def iterate_linearSVC(P, N, U, C=0.5, verbose=True):
    """run SVM iteratively until labels for U converge
    :param C:
    """

    print("Running iterative linear SVM")

    return pu_two_step.iterate_SVM(P=P, U=U, RN=N, C=C,
                                   kernel=None,
                                   max_neg_ratio=0.1, clf_selection=False, verbose=verbose)


def EM(P, N, U, ypU=None, max_pos_ratio=1.0, tolerance=0.05, max_imbalance_P_N=10.0, verbose=True):
    """Iterate EM until estimates for U converge.

    Train NB with P and N to get probabilistic labels for U, or use assumed priors if passed as parameter"""

    print("Running EM")

    # TODO balance P and N better
    if num_rows(P) > max_imbalance_P_N * num_rows(N):
        P_init = np.array(random.sample(list(P), int(max_imbalance_P_N * num_rows(N))))
    else:
        P_init = P

    if ypU is None:
        if verbose: print("\nBuilding classifier from Positive and Reliable Negative set")
        initial_model = build_proba_MNB(np.concatenate((P_init, N)),
                                        np.concatenate((np.ones(num_rows(P_init)), np.zeros(num_rows(N)))))

        if verbose: print("\nCalculating initial probabilistic labels for Unlabelled set")
        ypU = initial_model.predict_proba(U)[:, 1]
    else:
        print("Using assumed probabilities/weights for initial probabilistic labels of Unlabelled set")

    if verbose: print("\nIterating EM algorithm on P, N, and U\n")
    model = iterate_EM_PNU(P=P, N=N, U=U, ypU=ypU, tolerance=tolerance, max_pos_ratio=max_pos_ratio, verbose=verbose)

    return model


def iterate_knn(P, N, U, n_neighbors=7, thresh=0.6):
    P_, N_, U_ = arrays((P, N, U))

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', n_jobs=multi.cpu_count() - 1)
    knn.fit(np.concatenate((P_, N_)), np.concatenate((np.ones(num_rows(P_)), np.zeros(num_rows(N_)))))

    y_pred = knn.predict_proba(U_)
    U_pos, U_neg, U_ = partition_pos_neg_unsure(U_, y_pred, confidence=thresh)
    i = 0

    while num_rows(U_pos) and num_rows(U_neg) and num_rows(U_):
        print("Iteration #", i)
        print("New confidently predicted examples: \tpos", num_rows(U_pos), "\tneg", num_rows(U_neg))
        print("Remaining unlabelled:", num_rows(U_))

        P_ = np.concatenate((P_, U_pos))
        N_ = np.concatenate((N_, U_neg))

        knn.fit(np.concatenate((P_, N_)), np.concatenate((np.ones(num_rows(P_)), np.zeros(num_rows(N_)))))

        y_pred = knn.predict_proba(U_)
        U_pos, U_neg, U_ = partition_pos_neg_unsure(U_, y_pred, confidence=thresh)
        i += 1

    print("Converged with", num_rows(U_), "sentences remaining unlabelled. ",
          "\nLabelled pos:", num_rows(P_) - num_rows(P),
          "\tneg:", num_rows(N_) - num_rows(N),
          "Returning classifier")
    return knn


# slow
def iterate_SVC(P, N, U, kernel="rbf", verbose=True):
    """run SVM iteratively until labels for U converge"""

    print("Running iterative SVM with", kernel, "kernel")

    return pu_two_step.iterate_SVM(P=P, U=U, RN=N,
                                   kernel=kernel,
                                   max_neg_ratio=0.1, clf_selection=False, verbose=verbose)


# horrible results!
def propagate_labels(P, N, U, kernel='knn', n_neighbors=7, max_iter=30, n_jobs=-1):
    X = np.concatenate((P, N, U))
    y_init = np.concatenate((np.ones(num_rows(P)),
                             -np.ones(num_rows(N)),
                             np.zeros(num_rows(U))))
    propagation = semi_supervised.LabelPropagation(kernel=kernel, n_neighbors=n_neighbors, max_iter=max_iter,
                                                   n_jobs=n_jobs)
    propagation.fit(X, y_init)
    return propagation


# ----------------------------------------------------------------
# implementations
# ----------------------------------------------------------------

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

        if verbose: print("Iteration #", iterations, "\tBuilding new model using probabilistic labels")

        new_model = build_proba_MNB(np.concatenate((P, N, U)),
                                    np.concatenate((y_P, y_N, ypU)))

        if verbose: print("Predicting probabilities for U")
        ypU_old = ypU
        ypU = new_model.predict_proba(U)[:, 1]

        predU = [round(p) for p in ypU]
        pos_ratio = sum(predU) / len(U)

        if verbose: print("Unlabelled instances classified as positive:", sum(predU), "/", len(U),
                          "(", pos_ratio * 100, "%)\n")

        if pos_ratio >= max_pos_ratio:
            if verbose: print("Acceptable ratio of positively labelled sentences in U is reached.")
            break

    if verbose: print("Returning final model after", iterations, "iterations")
    return new_model


# ----------------------------------------------------------------
# supervised
# ----------------------------------------------------------------

# RandomForestClassifier
# LogisticRegression, SGDClassifier, Lasso, ElasticNet
# SVC, LinearSVC
# DecisionTreeClassifier
# MLPClassifier
# MultinomialNB

# quite good and fast (grid search: not so fast)
def logreg(P, N, U=None, verbose=True):
    X = np.concatenate((P, N))
    y = np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = GridSearchCV(estimator=LogisticRegression(),
                         param_grid={
                             'C'           : [10 ** x for x in range(-3, 3)],
                             'solver'      : ['sag'],
                             'class_weight': ['balanced']
                         },
                         ).fit(X, y)
    print("Best hyperparameters for Logistic Regression:", model.best_params_)
    print("Grid search score:", model.best_score_)
    return model.best_estimator_


def sgd(P, N, U, loss="modified_huber", verbose=True):
    X = np.concatenate((P, N))
    y = np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = SGDClassifier(loss=loss).fit(X, y)
    return model


# bad (biased towards one class)
def mnb(P, N, U, verbose=True):
    X = np.concatenate((P, N))
    y = np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = MultinomialNB().fit(X, y)
    return model


# very good but slow
def mlp(P, N, U, verbose=True):
    X = np.concatenate((P, N))
    y = np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = MLPClassifier().fit(X, y)
    return model


# ok but slow
def dectree(P, N, U, verbose=True):
    X = np.concatenate((P, N))
    y = np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = DecisionTreeClassifier().fit(X, y)
    return model


# TODO move to Supervised
def grid_search_linearSVM(P, N, U, verbose=True):
    model = LinearSVC()

    grid_search = GridSearchCV(model,
                               param_grid={'C'           : [10 ** x for x in range(-5, 5, 2)],
                                           'class_weight': ['balanced'],
                                           'loss'        : ['hinge', 'squared_hinge'],
                                           },
                               cv=3,
                               n_jobs=2,  # min(multi.cpu_count(), 16),
                               verbose=0)

    if verbose:
        print("Grid searching parameters for Linear SVC")
    X = np.concatenate((P, N))
    y = np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))

    grid_search.fit(X, y)

    print("SVC parameters:", grid_search.best_params_, "\tscore:", grid_search.best_score_)

    return grid_search.best_estimator_


# TODO move to Supervised
def grid_search_SVC(P, N, U, verbose=True):
    model = SVC()

    grid_search = GridSearchCV(model,
                               param_grid={'C'           : [10 ** x for x in range(-5, 5, 2)],
                                           'class_weight': ['balanced'],
                                           'kernel'      : ['linear', 'poly', 'rbf', 'sigmoid']
                                           },
                               cv=3,
                               n_jobs=min(multi.cpu_count(), 16),
                               verbose=0)

    if verbose:
        print("Grid searching parameters for SVC")
    X = np.concatenate((P, N))
    y = np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))

    grid_search.fit(X, y)

    print("SVC parameters:", grid_search.best_params_, "\tscore:", grid_search.best_score_)

    return grid_search.best_estimator_


# terrible
def lasso(P, N, U, verbose=True):
    X = np.concatenate((P, N))
    y = np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = Lasso().fit(X, y)
    return model


# terrible
def elastic(P, N, U, verbose=True):
    X = np.concatenate((P, N))
    y = np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = Lasso().fit(X, y)
    return model


# bad
def randomforest(P, N, U, verbose=True):
    X = np.concatenate((P, N))
    y = np.concatenate((np.ones(num_rows(P)), np.zeros(num_rows(N))))
    model = RandomForestClassifier().fit(X, y)
    return model
