from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from numpy import concatenate, zeros, ones, arange
from multiprocessing import Pool, cpu_count
from functools import partial
from semisuper.helpers import identity, num_rows, pu_measure
from semisuper.transformers import BasicPreprocessor, FeatureNamePipeline, TextStats


def biased_SVM_weight_selection(P, U, Cs_neg=None, Cs_pos_factors=None, Cs=None, kernel='linear', text=True,
                                test_size=0.3):
    """run biased SVMs with combinations of class weight values, choose the one with the best pu_measure"""

    # default values
    # TODO remove C as a parameter, find meaningful range of pos and neg weights
    if Cs is None:
        Cs = [10 ** x for x in range(0, 4)]
    if Cs_neg is None:
        Cs_neg = arange(0.01, 0.63, 0.04)
    if Cs_pos_factors is None:
        Cs_pos_factors = range(10, 210, 20)

    Cs = [(C, C_neg * j, C_neg)
          for C in Cs for C_neg in Cs_neg for j in Cs_pos_factors]

    print("There are", num_rows(Cs), "parameter combinations to be evaluated.")
    print("NON-NORMALIZED WEIGHTS")

    P_train, P_test = train_test_split(P, test_size=test_size)
    U_train, U_test = train_test_split(U, test_size=test_size)
    X = concatenate((P_train, U_train))
    y = concatenate((ones(num_rows(P_train)), zeros(num_rows(U_train))))

    with Pool(processes=cpu_count()) as p:
        score_weights = p.map(partial(eval_params,
                                      X_train=X, y_train=y, P_test=P_test, U_test=U_test,
                                      kernel=kernel, text=text),
                              Cs)

    best_score_params = max(score_weights, key=lambda tup: tup[0])

    print()
    [print(s) for s in score_weights]

    print("\nBest model has parameters", best_score_params[1], "and PU-score", best_score_params[0])
    print("Building final classifier")

    model = build_biased_SVM(concatenate((P, U)),
                             concatenate((ones(num_rows(P)), zeros(num_rows(U)))),
                             C_pos=best_score_params[1]['C_pos'],
                             C_neg=best_score_params[1]['C_neg'],
                             C=best_score_params[1]['C'],
                             probability=True, kernel=kernel, text=text)

    return model


def eval_params(Cs, X_train, y_train, P_test, U_test, kernel='linear', text=True):
    C, C_pos, C_neg = Cs
    model = build_biased_SVM(X_train, y_train, C_pos=C_pos, C_neg=C_neg, C=C, kernel=kernel, text=text)

    y_P = model.predict(P_test)
    y_U = model.predict(U_test)

    score = pu_measure(y_P, y_U)
    params = model.get_class_weights()

    return score, params


# ----------------------------------------------------------------
# BIASED SVM BUILDERS AND CLASSES
# ----------------------------------------------------------------

def build_biased_SVM(X, y, C_pos, C_neg, C=1.0, kernel='linear', probability=False, verbose=True, text=True):
    """build biased-SVM classifier (weighting false positives and false negatives differently)

    C_pos is the weight for positive class, or penalty for false negative errors; C_neg analogously.
    C controls how hard the margin is in general."""

    def build(X, y):
        """
        Inner build function that builds a single model.
        """

        class_weight = {1.0: C_pos, 0.0: C_neg}  # non-normalizing version
        # class_weight = {1.0: C_pos / (C_pos + C_neg), 0.0: C_neg / (C_pos + C_neg)}  # normalizing version

        clf = BiasedSVM(C=C, class_weight=class_weight, kernel=kernel, probability=probability)

        if text:
            model = Pipeline([
                ('preprocessor', BasicPreprocessor()),
                ('vectorizer', FeatureUnion(transformer_list=[
                    ("words", TfidfVectorizer(
                            tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1, 3))
                     ),
                    ("stats", FeatureNamePipeline([
                        ("stats", TextStats()),
                        ("vect", DictVectorizer())
                    ]))
                ]
                )),
                ('classifier', clf)
            ])
        else:
            model = Pipeline([
                ('classifier', clf)
            ])

        model.fit(X, y)
        model.get_class_weights = clf.get_class_weights

        return model

        # if verbose:
        # print('.', end='', flush=True)
        # print("Building model with parameters C+ := ", C_pos, ", C- :=", C_neg)

    model = build(X, y)

    return model


class BiasedSVM(SVC):
    """wrapper for sklearn SVC with get_class_weights function and linear default kernel"""

    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape=None,
                 random_state=None):
        self.param_class_weight = {'C_pos': class_weight[1], 'C_neg': class_weight[0], 'C': C}

        super(BiasedSVM, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                                        coef0=coef0, shrinking=shrinking, probability=probability,
                                        tol=tol, cache_size=cache_size, class_weight=class_weight,
                                        verbose=verbose, max_iter=max_iter,
                                        decision_function_shape=decision_function_shape,
                                        random_state=random_state)

    def get_class_weights(self):
        return self.param_class_weight
