from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import svm
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer, VectorizerMixin
from sklearn.feature_extraction import DictVectorizer
from semisuper.helpers import identity
from semisuper.transformers import BasicPreprocessor, TextStats, FeatureNamePipeline
import pickle


# TODO find params that give less horrible results
def one_class_svm(X, X_test=None, y_test=None, outpath=None, verbose=True,
                  kernel="rbf", degree=3, shrinking=True):
    def build(X):
        """One-class SVM as PU baseline"""

        classifier = svm.OneClassSVM(kernel=kernel, degree=degree, verbose=verbose, shrinking=shrinking)

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
        model.fit(X)
        return model

    # Begin evaluation
    if verbose:
        print("Building for evaluation")
    model = build(X)
    if X_test is not None and y_test is not None:
        if verbose:
            print("Classification Report:\n")
        y_pred = model.predict(X_test)
        print(clsr(y_test, y_pred))
        if verbose:
            print("Building complete model and saving ...")
    if outpath:
        with open(outpath, 'wb') as f:
            pickle.dump(model, f)
        print("Model written out to {}".format(outpath))
    return model
