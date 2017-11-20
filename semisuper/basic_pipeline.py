import pickle
from operator import itemgetter

from semisuper.helpers import identity
from semisuper.transformers import TokenizePreprocessor, TextStats, FeatureNamePipeline
from sklearn import naive_bayes
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Binarizer


def train_clf(X_vec, y, classifier=None, binary=False, verbose=False):
    """build and train classifier on pre-vectorized data"""

    if verbose:
        print("Training classifier...")

    if not classifier:
        clf = naive_bayes.MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
    elif isinstance(classifier, type):
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


def build_pipeline(X, y, classifier=None, outpath=None, verbose=False,
                   words=True, wordgram_range=(1, 3),
                   chars=True, chargram_range=(3, 6),
                   binary=False):
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
        ('features', feature_vectorizer(words=words, wordgram_range=wordgram_range,
                                        chars=chars, chargram_range=chargram_range,
                                        binary=binary)),
        ('classifier', clf)
    ])

    model.fit(X, y)

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)
            print("Model written out to", outpath)

    return model


def feature_vectorizer(words=True, wordgram_range=(1, 3), chars=True, chargram_range=(3, 6),
                       binary=False, rules=True, lemmatize=True):
    vectorizer = FeatureUnion(
        transformer_list=[
            ("wordgrams", None if not words else
            FeatureNamePipeline([
                ("preprocessor", TokenizePreprocessor(rules=rules, lemmatize=lemmatize)),
                ("word_tfidf", TfidfVectorizer(
                    analyzer='word',
                    # min_df=5, # TODO find reasonable value (5 <= n << 50)
                    tokenizer=identity, preprocessor=None, lowercase=False,
                    ngram_range=wordgram_range,
                    binary=binary, norm='l2' if not binary else None, use_idf=not binary))
            ])),
            ("chargrams", None if not chars else
            FeatureNamePipeline([
                ("char_tfidf", TfidfVectorizer(
                    analyzer='char',
                    # min_df=5,
                    preprocessor=None, lowercase=False,
                    ngram_range=chargram_range,
                    binary=binary, norm='l2' if not binary else None, use_idf=not binary))
            ])),
            ("stats", None if binary else
            FeatureNamePipeline([
                ("stats", TextStats()),
                ("vect", DictVectorizer())
            ]))
        ])
    return vectorizer


def show_most_informative_features(model: object, text: object = None, n: object = 40) -> object:
    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.
    Note that this function will only work on linear models with coefs_
    """
    # Extract the vectorizer and the classifier from the pipeline
    features = model.named_steps['features']
    classifier = model.named_steps['classifier']

    # Check to make sure that we can perform this computation
    if not hasattr(classifier, 'coef_'):
        raise TypeError(
            "Cannot compute most informative features on {} model.".format(
                classifier.__class__.__name__
            )
        )

    if text is not None:
        # Compute the coefficients for the text
        tvec = model.transform([text]).toarray()
    else:
        # Otherwise simply use the coefficients
        tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], features.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    topn = zip(coefs[:n], coefs[:-(n + 1):-1])

    # Create the output string to return
    output = []

    # If text, add the predicted value to the output.
    if text is not None:
        output.append("\"{}\"".format(text))
        output.append("Classified as: {}".format(model.predict([text])))
        output.append("")

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
        )

    return "\n".join(output)

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
