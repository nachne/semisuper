import pickle
from operator import itemgetter

from semisuper.helpers import identity
from semisuper.transformers import TokenizePreprocessor, TextStats, FeatureNamePipeline, Densifier, TextNormalizer
from sklearn import naive_bayes
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler
from sklearn.decomposition import *
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
import re


def train_clf(X_vec, y, classifier, binary=False, verbose=False):
    """build and train classifier on pre-vectorized data"""

    if verbose:
        print("Training classifier...")

    if isinstance(classifier, type):
        clf = classifier()
    else:
        clf = classifier

    if binary:
        model = Pipeline([('binarizer', Binarizer()),
                          ('clf', clf)])
    else:
        model = clf

    model.fit(X_vec, y)
    return model


def build_pipeline(X, y, classifier=None, outpath=None, verbose=False, wordgram_range=(1, 3), chargram_range=(3, 6),
                   binary=False, selection=True):
    """build complete pipeline"""

    if verbose:
        print("Building model pipeline...")

    if not classifier:
        clf = naive_bayes.MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
    elif isinstance(classifier, type):
        clf = classifier()
    else:
        clf = classifier

    model = Pipeline([
        ('features', vectorizer(chargrams=chargram_range, wordgrams=wordgram_range, binary=binary)),
        ('selector', None if not selection else
        # selector(score_func=score_func, percentile=percentile)),
        factorization()),
        ('classifier', clf)
    ])

    model.fit(X, y)

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)
            print("Model written out to", outpath)

    return model


def vectorizer(chargrams=(2, 6), min_df_char=0.001, wordgrams=None, min_df_word=0.001, lemmatize=False, rules=True,
               max_df=1.0, binary=False):
    return FeatureNamePipeline([
        ("text_normalizer", TextNormalizer()),
        ("features", FeatureUnion(n_jobs=2,
                                  transformer_list=[
                                      ("wordgrams", None if wordgrams is None else
                                      FeatureNamePipeline([
                                          ("preprocessor", TokenizePreprocessor(rules=rules, lemmatize=lemmatize)),
                                          ("word_tfidf", TfidfVectorizer(
                                                  analyzer='word',
                                                  min_df=min_df_word,  # TODO find reasonable value (5 <= n << 50)
                                                  max_df=max_df,
                                                  tokenizer=identity,
                                                  preprocessor=None,
                                                  lowercase=False,
                                                  ngram_range=wordgrams,
                                                  binary=binary, norm='l2' if not binary else None,
                                                  use_idf=not binary))
                                      ])),
                                      ("chargrams", None if chargrams is None else
                                      FeatureNamePipeline([
                                          ("char_tfidf", TfidfVectorizer(
                                                  analyzer='char',
                                                  min_df=min_df_char,
                                                  max_df=max_df,
                                                  preprocessor=partial(re.compile("[^\w\-=%]+").sub, " "),
                                                  lowercase=True,
                                                  ngram_range=chargrams,
                                                  binary=binary, norm='l2' if not binary else None,
                                                  use_idf=not binary))
                                      ])),
                                      ("stats", None if binary else
                                      FeatureNamePipeline([
                                          ("stats", TextStats()),
                                          ("vect", DictVectorizer())
                                      ]))
                                  ]))
    ])


class identitySelector():
    """feature selector that does nothing"""

    def __init__(self):
        print("Feature selection: None")
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def percentile_selector(score_func='chi2', percentile=20):
    """supervised feature selector"""

    funcs = {'chi2'               : chi2,
             'f_classif'          : f_classif,
             'f'                  : f_classif,
             'mutual_info_classif': mutual_info_classif,
             'mutual_info'        : mutual_info_classif,
             'm'                  : mutual_info_classif,
             }

    func = funcs.get(score_func, chi2)

    print("Supervised feature selection:,", percentile, "-th percentile in terms of", func)
    return SelectPercentile(score_func=func, percentile=percentile)


def factorization(method='TruncatedSVD', n_components=10):
    # PCA, IncrementalPCA, FactorAnalysis, FastICA, LatentDirichletAllocation, TruncatedSVD, fastica

    print("Unsupervised feature selection: matrix factorization with", method, "(", n_components, "components )")

    sparse = {
        'LatentDirichletAllocation': LatentDirichletAllocation(n_topics=n_components,
                                                               n_jobs=-1,
                                                               learning_method='online'),
        'TruncatedSVD'             : FeatureNamePipeline([("selector", TruncatedSVD(n_components)),
                                                          ("normalizer", StandardScaler())])
    }

    model = sparse.get(method, None)

    if model is not None:
        return model

    dense = {
        'PCA'           : PCA(n_components),
        'FactorAnalysis': FactorAnalysis(n_components)
    }

    model = dense.get(method, None)

    if model is not None:
        return FeatureNamePipeline([("densifier", Densifier()),
                                    ("selector", model),
                                    ("normalizer", StandardScaler())])  # TODO Standard or MinMax?

    else:

        return FeatureNamePipeline([("selector", TruncatedSVD(n_components)),
                                    ("normalizer", StandardScaler())])
