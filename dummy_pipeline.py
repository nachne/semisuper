from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn import svm, naive_bayes
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer, VectorizerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split as tts
from operator import itemgetter
from helpers import identity

from preprocessors import BasicPreprocessor
import pickle


def build_and_evaluate(X, y,
                       classifier=naive_bayes.MultinomialNB(alpha=1.0,
                                                            class_prior=None,
                                                            fit_prior=True),
                       outpath=None, verbose=True):
    def build(classifier, X, y=None):
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
                 ),
                ("stats", FeatureNamePipeline([
                    ("stats", TextStats()),
                    ("vect", DictVectorizer())
                ]))
            ]
            )),
            ('classifier', classifier)
        ])

        model.fit(X, y)
        return model

    # Label encode the targets
    labels = LabelEncoder()
    y = labels.fit_transform(y)

    # Begin evaluation
    if verbose:
        print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    model = build(classifier, X_train, y_train)

    if verbose:
        print("Classification Report:\n")

    y_pred = model.predict(X_test)
    print(clsr(y_test, y_pred, target_names=labels.classes_))

    if verbose:
        print("Building complete model and saving ...")
    model = build(classifier, X, y)
    model.labels_ = labels

    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)

        print("Model written out to {}".format(outpath))

    return model


def show_most_informative_features(model: object, text: object = None, n: object = 40) -> object:
    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.
    Note that this function will only work on linear models with coefs_
    """
    # Extract the vectorizer and the classifier from the pipeline
    vectorizer = model.named_steps['vectorizer']
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
            zip(tvec[0], vectorizer.get_feature_names()),
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


class FeatureNamePipeline(Pipeline):
    def get_feature_names(self):
        return self._final_estimator.get_feature_names()


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from tokenized document for DictVectorizer

    inverse_length: 1/(number of tokens)
    """

    key_dict = {'inverse_length': 'inverse_length'}

    def fit(self, X=None, y=None):
        return self

    def transform(self, token_lists):
        for tl in token_lists:
            yield {'inverse_length': (1.0 / len(tl) if tl else 1.0)}

    def get_feature_names(self):
        return list(self.key_dict.keys())
