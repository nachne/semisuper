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
from helpers import identity
from transformers import BasicPreprocessor, TextStats, FeatureNamePipeline
import pickle


# TODO: generate proper results (compute meaningful threshold)
# TODO (refactor): get rid of redundancies with a nice interface
def ranking_cos_sim(X, X_test=None, y_test=None, threshold=0.1, X_thresh=True):
    """fits mean training vector and predicts whether cosine similarity is above threshold (default: 0.0)

    predict_proba returns similarity scores.
    if X_thresh is true, uses the training vectors' similarities to compute a threshold.
    """

    def build(X, threshold, min_sim_thresh):
        model = Pipeline([
            ('preprocessor', BasicPreprocessor()),
            ('vectorizer', TfidfVectorizer(
                    tokenizer=identity, preprocessor=None, lowercase=False, ngram_range=(1, 3))
             ),
            ('normalizer', Normalizer()),
            ('classifier', SimRanker(threshold, min_sim_thresh))])
        model.fit(X)
        return model

    model = build(X, threshold, X_thresh)
    return model


class SimRanker(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold, min_sim_thresh):
        self.threshold = threshold
        self.min_sim_thresh = min_sim_thresh
        self.mean_X = None
        return

    # TODO dimension mismatch error!
    def fit(self, X, y=None):
        self.mean_X = normalize(X.mean(axis=0))

        if self.min_sim_thresh:
            cos_sim = cosine_similarity(self.mean_X, X)
            self.threshold = (cos_sim.mean() + cos_sim.min()) / 2
            print("threshold (mean(mean_sim(X, X_mean), min_sim(X, X_mean)):", self.threshold)

        return self

    def predict(self, X):
        return [1 if cosine_similarity(self.mean_X, x) > [self.threshold] else 0 for x in X]

    def predict_proba(self, X):
        return cosine_similarity(self.mean_X, X)
