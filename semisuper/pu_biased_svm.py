from functools import partial
from multiprocessing import Pool, cpu_count

from numpy import zeros, ones
from semisuper.helpers import num_rows, pu_score, partition_pos_neg, train_report, concatenate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import BaggingClassifier


def biased_SVM_grid_search(P, U, Cs=None, kernel='linear', n_estimators=9, verbose=False):
    if Cs is None:
        Cs = [10 ** x for x in range(-12, 12, 2)]

    if verbose:
        print("Running Biased-SVM with balanced class weights and grid search over", len(Cs), "C values")

    model = BaggingClassifier(LinearSVC())

    grid_search = GridSearchCV(model,
                               param_grid={'base_estimator__C'           : Cs,
                                           'base_estimator__class_weight': ['balanced'],
                                           ### not applicable for LinearSVC
                                           # 'base_estimator__kernel'      : [kernel],
                                           # 'base_estimator__cache_size'  : [8000],
                                           # 'base_estimator__probability' : [True],
                                           ### fit parameters for Bagging wrapper
                                           'bootstrap'                   : [True],
                                           'n_estimators'                : [n_estimators],
                                           ### parallelization incompatible with multiprocessing
                                           # 'n_jobs'                      : [n_estimators]
                                           },
                               scoring=pu_scorer,
                               verbose=0)

    if verbose:
        print("Grid searching parameters for biased-SVM")
    X = concatenate((P, U))
    y = concatenate((ones(num_rows(P)), zeros(num_rows(U))))

    grid_search.fit(X, y)

    if verbose:
        train_report(grid_search.best_estimator_, P, U)
    print("Biased-SVM parameters:", grid_search.best_params_, "\tPU score:", grid_search.best_score_)

    return grid_search.best_estimator_


def biased_SVM_weight_selection(P, U,
                                Cs_neg=None, Cs_pos_factors=None, Cs=None,
                                kernel='linear', test_size=0.2,
                                verbose=False):
    """run biased SVMs with combinations of class weight values, choose the one with the best pu_measure"""

    # default values
    if Cs is None:
        Cs = [10 ** x for x in range(-12, 12, 2)]
    if Cs_neg is None:
        Cs_neg = [1]  # arange(0.01, 0.63, 0.04)
    if Cs_pos_factors is None:
        Cs_pos_factors = range(1, 1100, 200)

    Cs = [(C, C_neg * j, C_neg)
          for C in Cs for C_neg in Cs_neg for j in Cs_pos_factors]

    if verbose:
        print("Running Biased-SVM with range of C and positive class weight factors.",
              num_rows(Cs), "parameter combinations.")

    P_train, P_test = train_test_split(P, test_size=test_size)
    U_train, U_test = train_test_split(U, test_size=test_size)
    X = concatenate((P_train, U_train))
    y = concatenate((ones(num_rows(P_train)), zeros(num_rows(U_train))))

    # with Pool(processes=min(cpu_count() - 1, num_rows(Cs))) as p:
    score_weights = map(partial(eval_params, X_train=X, y_train=y, P_test=P_test, U_test=U_test, kernel=kernel),
                        Cs)

    best_score_params = max(score_weights, key=lambda tup: tup[0])

    [print(s)
     for s in score_weights]
    if verbose:
        print("\nBest model has parameters", best_score_params[1], "and PU-score", best_score_params[0])
        print("Building final classifier")

    model = build_biased_SVM(concatenate((P, U)),
                             concatenate((ones(num_rows(P)), zeros(num_rows(U)))),
                             C_pos=best_score_params[1]['C_pos'],
                             C_neg=best_score_params[1]['C_neg'],
                             C=best_score_params[1]['C'],
                             probability=True, kernel=kernel)

    if verbose:
        train_report(model, P, U)
    print("Returning Biased-SVM with parameters", best_score_params[1], "and PU-score", best_score_params[0])
    return model


def eval_params(Cs, X_train, y_train, P_test, U_test, kernel='linear'):
    C, C_pos, C_neg = Cs
    model = build_biased_SVM(X_train, y_train, C_pos=C_pos, C_neg=C_neg, C=C, kernel=kernel)

    y_P = model.predict(P_test)
    y_U = model.predict(U_test)

    score = pu_score(y_P, y_U)
    params = model.get_class_weights()

    return score, params


# ----------------------------------------------------------------
# BIASED SVM BUILDERS AND CLASSES
# ----------------------------------------------------------------

def build_biased_SVM(X, y, C_pos, C_neg, C=1.0, kernel='linear', probability=False):
    """build biased-SVM classifier (weighting false positives and false negatives differently)

    C_pos is the weight for positive class, or penalty for false negative errors; C_neg analogously.
    C controls how hard the margin is in general."""

    class_weight = {1.0: C_pos / (C_pos + C_neg), 0.0: C_neg / (C_pos + C_neg)}  # normalizing version
    # print("Building biased-SVM with normalized weights. "
    #       "C+ :=", C_pos / (C_pos + C_neg), "\tC- :=", C_neg / (C_pos + C_neg),  "\tC :=", C)

    clf = BiasedSVM(C=C, class_weight=class_weight)

    # clf = BaggingClassifier(BiasedSVM(C=C, class_weight=class_weight))
    # clf.get_class_weights = clf.base_estimator.get_class_weights

    model = clf.fit(X, y)

    model.get_class_weights = clf.get_class_weights

    return model


class BiasedSVM(LinearSVC):
    """wrapper for sklearn SVC with get_class_weights function and linear default kernel"""

    def __init__(self, C=1.0, class_weight='balanced',
                 tol=1e-3, verbose=False, max_iter=1000,
                 random_state=None):
        self.param_class_weight = {'C_pos': class_weight[1], 'C_neg': class_weight[0], 'C': C}

        super(BiasedSVM, self).__init__(C=C, class_weight=class_weight,  # kernel='linear',
                                        tol=tol, verbose=verbose, max_iter=max_iter, random_state=random_state)

    def get_class_weights(self):
        return self.param_class_weight


def pu_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    y_P, y_U = partition_pos_neg(y_pred, y)
    return pu_score(y_P, y_U)
