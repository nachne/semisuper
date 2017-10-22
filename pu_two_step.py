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
from simple_nb import proba_label_MNB


# TODO: PROPERLY ADJUST PROBABILITIES INSTEAD OF LABELS
# TODO: wrapper method that finds appropriate max_imbalance
# (e.g. bisection method starting from 1.0 and |U|/|P+N|,
# satisfied when ratio of positively labelled u in U is in [0.1, 0.5]
# Alternative: some minimum positive ratio as stopping criterion
# Alternative: yield (model, ratio) so caller can choose desired one
def expectation_maximization(P, U, N=[], outpath=None, max_imbalance=1.5, max_pos_ratio=0.5, tolerance=0.05, text=True):
    """EM algorithm for positive set P, unlabelled set U and (optional) negative set N

    iterate NB classifier with updated labels for unlabelled set (initially negative) until convergence
    if U is much larger than L=P+N, randomly samples to max_imbalance-fold of |L|"""

    if N:
        L = P + N
        y_L = np.concatenate(np.array([[1., 0.]] * np.shape(P)[0]),
                             np.array([[0., 1.]] * np.shape(N)[0]))
    else:
        L = P
        y_L = np.array([[1., 0.]] * np.shape(P)[0])

    if np.shape(U)[0] > max_imbalance * np.shape(L)[0]:
        U = np.array(sample(list(U), int(max_imbalance * np.shape(L)[0])))

    ynU = np.array([[0., 1.]] * np.shape(U)[0])
    ypU = np.array([[0., 1.]] * np.shape(U)[0])
    ypU_old = [[-1, -1]]
    iterations = 0
    model = None

    print("shape of L, y_L, U, ypU", np.shape(L), np.shape(y_L), np.shape(U), np.shape(ypU))

    while not almost_equal(ypU_old, ypU, tolerance):

        print("Iteration #", iterations)

        print("building new model using probabilistic labels")
        model = build_model(np.concatenate((L, U)),
                            np.concatenate((y_L, ypU)), text=text)

        ypU_old = ypU
        print("predicting probabilities for U")
        ypU = model.predict_proba(U)
        print(ypU)

        iterations += 1

        print("labels for U")
        predU = [round(p[0]) for p in ypU]
        pos_ratio = sum(predU) / len(U)

        print("Unlabelled instances classified as positive:", sum(predU), "/", len(U),
              "(", pos_ratio * 100, "%)\n")

        if pos_ratio >= max_pos_ratio:
            print("Acceptable ratio of positively labelled sentences in U is reached.")
            break

    # Begin evaluation
    print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(np.concatenate((P, U)),
                                           np.concatenate(([1 for x in P],
                                                           [0 for x in U])),
                                           test_size=0.2)
    evalmodel = build_model(X_train, y_train, text=text)

    print("Classification Report:\n")

    y_pred = evalmodel.predict(X_test)
    print(clsr(y_test, y_pred))

    print("Returning final model after", iterations, "refinement iterations")

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


# ----------------------------------------------------------------
# general model builder

# TODO: Make versatile module for this and reuse in respective functions
def build_model(X, y,
                verbose=True, text=True):
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


def almost_equal(pairs1, pairs2, tolerance=0.05):
    zipped = zip(pairs1, pairs2)
    diffs = [abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance
             for p1, p2 in zipped]
    return all(diffs)
