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
import pickle

from transformers import BasicPreprocessor, TextStats, FeatureNamePipeline
from simple_nb import proba_label_MNB


# TODO: PROPERLY ADJUST PROBABILITIES INSTEAD OF LABELS
# TODO: wrapper method that finds appropriate max_imbalance
# (e.g. bisection method starting from 1.0 and |U|/|P+N|,
# satisfied when ratio of positively labelled u in U is in [0.1, 0.5]
# Alternative: some minimum positive ratio as stopping criterion
# Alternative: yield (model, ratio) so caller can choose desired one
def expectation_maximization(P, U, N=[], outpath=None, max_imbalance=1.5, max_pos_ratio=0.5):
    """EM algorithm for positive set P, unlabelled set U and (optional) negative set N

    iterate NB classifier with updated labels for unlabelled set (initially negative) until convergence
    if U is much larger than L=P+N, randomly samples to max_imbalance-fold of |L|"""

    if N:
        L = P + N
        y_L = np.concatenate(np.array([1] * len(P)),
                             np.array([0] * len(N)))
    else:
        L = P
        y_L = np.array([1] * len(P))

    if len(U) > max_imbalance * len(L):
        U = sample(U, int(max_imbalance * len(L)))

    ypU = np.array([0] * len(U))
    ypU_old = -1
    iterations = 0
    model = None

    while not equal(ypU_old, ypU):
        model = build_model(L + U, np.concatenate((y_L, ypU)))

        ypU_old = ypU
        ypU = model.predict_proba(U)

        iterations += 1

        predU = model.predict(U)
        pos_ratio = sum(predU) / len(U)

        print("Iteration #", iterations,
              "\nUnlabelled instances classified as positive:", sum(predU), "/", len(U),
              "(", pos_ratio * 100, "%)")

        if pos_ratio >= max_pos_ratio:
            print("Acceptable ratio of positively labelled sentences in U is reached.")
            break

    # Begin evaluation
    print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    evalmodel = build_model(X_train, y_train)

    print("Classification Report:\n")

    y_pred = evalmodel.predict(X_test)
    print(clsr(y_test, y_pred))

    print("Returning final model")

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


# ----------------------------------------------------------------
# general model builder

# TODO: Make versatile module for this and reuse in respective functions
def build_model(X, y,
                verbose=True):
    def build(X, y):
        """
        Inner build function that builds a single model.
        """

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
            ('classifier', proba_label_MNB(alpha=0.1))
        ])

        model.fit(X, y)
        return model

    if verbose:
        print("Building model ...")
    model = build(X, y)

    return model
