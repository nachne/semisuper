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
from transformers import BasicPreprocessor, TextStats, FeatureNamePipeline
import pickle


# TODO: wrapper method that finds appropriate max_imbalance
# (e.g. bisection method starting from 1.0 and |U|/|P|,
# satisfied when ratio of positively labelled u in U is in [0.1, 0.5]
# Alternative: some minimum positive ratio as stopping criterion
# Alternative: yield (model, ratio) so caller can choose desired one
def expectation_maximization(P, U, outpath=None, max_imbalance=1.5, max_pos_ratio=0.5):
    """iterate NB classifier with updated labels for unlabelled set (initially negative) until convergence

    if U is much larger than P, randomly samples to max_imbalance-fold of |P|"""

    if len(U) > max_imbalance * len(P):
        U = sample(U, int(max_imbalance * len(P)))

    y_P = np.array([1] * len(P))
    y_U = np.array([0] * len(U))
    y_U_old = -1
    iterations = 0

    while not equal(y_U_old, y_U):
        model = build_model(P + U, np.concatenate((y_P, y_U)), classifier=naive_bayes.MultinomialNB(alpha=1.0))

        y_U_old = y_U
        y_U = model.predict(U)

        iterations += 1

        pos_ratio = sum(y_U) / len(U)

        print("Iteration #", iterations,
              "\nUnlabelled instances classified as positive:", sum(y_U), "/", len(U),
              "(", pos_ratio*100, "%)")

        if pos_ratio >= max_pos_ratio:
            print("Acceptable ratio of positively labelled sentences in U is reached.")
            break


    print("Returning final model")

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


# ----------------------------------------------------------------
# general model builder

# TODO: Make versatile module for this and reuse in respective functions
def build_model(X, y, classifier=naive_bayes.MultinomialNB(alpha=1.0,
                                                           class_prior=None,
                                                           fit_prior=True),
                verbose=True):
    def build(classifier, X, y):
        """
        Inner build function that builds a single model.
        """

        if isinstance(classifier, type):
            classifier = classifier()

        model = Pipeline([
            ('preprocessor', BasicPreprocessor()),
            ('vectorizer', FeatureUnion(transformer_list=[
                ("words", TfidfVectorizer(
                        tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1, 3))
                 )
                # ,
                # ("stats", FeatureNamePipeline([
                #     ("stats", TextStats()),
                #     ("vect", DictVectorizer())
                # ]))
            ]
            )),
            ('classifier', classifier)
        ])

        model.fit(X, y)
        return model

    # Begin evaluation
    if verbose:
        print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    model = build(classifier, X_train, y_train)

    if verbose:
        print("Classification Report:\n")

    y_pred = model.predict(X_test)
    print(clsr(y_test, y_pred))

    if verbose:
        print("Building complete model ...")
    model = build(classifier, X, y)

    return model
