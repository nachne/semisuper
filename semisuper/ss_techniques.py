import random

import numpy as np
import sklearn.semi_supervised as ss
from semisuper import pu_two_step
from semisuper.helpers import num_rows
from semisuper.proba_label_nb import build_proba_MNB
from semisuper.pu_two_step import almost_equal




def iterate_SVM(P, N, U, verbose=True):
    """run SVM iteratively until labels for U converge"""

    print("Running iterative SVM")

    return pu_two_step.iterate_SVM(P=P, U=U, RN=N, max_neg_ratio=0.1, clf_selection=False, verbose=verbose)


def EM(P, N, U, ypU=None, max_pos_ratio=1.0, tolerance=0.05, max_imbalance_P_N=1.5, verbose=True):
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
                                        [1] * num_rows(P) + [0] * num_rows(N))

        if verbose: print("\nCalculating initial probabilistic labels for Unlabelled set")
        ypU = initial_model.predict_proba(U)[:, 1]
    else:
        print("Using assumed probabilities/weights for initial probabilistic labels of Unlabelled set")

    if verbose: print("\nIterating EM algorithm on P, N, and U\n")
    model = iterate_EM_PNU(P=P, N=N, U=U, ypU=ypU, tolerance=tolerance, max_pos_ratio=max_pos_ratio, verbose=verbose)

    return model


# horrible results!
def propagate_labels(P, N, U, kernel='knn', n_neighbors=7, max_iter=30, n_jobs=-1):
    X = np.concatenate((P, N, U))
    y_init = np.concatenate((np.ones(num_rows(P)),
                             -np.ones(num_rows(N)),
                             np.zeros(num_rows(U))))
    propagation = ss.LabelPropagation(kernel=kernel, n_neighbors=n_neighbors, max_iter=max_iter, n_jobs=n_jobs)
    propagation.fit(X, y_init)
    return propagation



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

