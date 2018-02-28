from __future__ import absolute_import, division, print_function

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
print('importing stuff')
print('from semisuper import transformers, helpers, loaders')
from semisuper import helpers, transformers, loaders

class KeySentencePredictor(BaseEstimator, TransformerMixin):
    """predicts positions and scores of relevant sentences for list of {"pmid" : <pmid>, "abstract" : <txt>} dicts"""

    def __init__(self, batch_size=100):
        """load pretrained classifier and maximum score in silver standard corpus for normalization"""

        print('INIT')
        self.max_batch_size = batch_size
        self.pipeline = loaders.load_pipeline("semisuper/pickles/semi_pipeline.pickle")

        if not hasattr(self.pipeline, "predict_proba"):
            self.max_score = loaders.max_score_from_csv(loaders.load_silver_standard())
        else:
            self.max_score = 1.0

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        """return dict with all pmids in X as keys and lists of key sentence <start, end, score> tuples as values

        same as transform"""

        return self.transform(X)

    def transform(self, X):
        """return dict with all pmids in X as keys and lists of key sentence <start, end, score> tuples as values"""

        return helpers.merge_dicts(map(self.transform_batch, helpers.partition(X, self.max_batch_size)))

    def transform_batch(self, X):
        """predicts positions and scores of relevant sentences for list of {pmid, abstract} dictionaries"""
        print('transofrm_batch')
        sentences, pmids, positions = self.sentences_pmids_positions(X)

        scores = self.sentence_scores(sentences)

        return self.hit_dict_list(pmids, scores, positions)

    def hit_dict_list(self, pmids, scores, positions):
        """build up result (dict of pmids and relevant sentences) from intermediate lists"""

        result_dict = {pmid: [] for pmid in pmids}

        for pmid, (start, end), score in zip(pmids, positions, scores):
            if score > 0:
                result_dict[pmid].append((start, end, score))

        return result_dict

    def sentences_pmids_positions(self, X):
        """turn list of dicts into lists of individual sentences, corresponding pmids, and positions

        dicts must have keys pmid and abstract"""
        print("sentences_pmids_positions")
        sentence_lists = [transformers.sentence_tokenize(x["abstract"])
                          for x in X]

        sentences = helpers.flatten(sentence_lists)
        positions = helpers.flatten(map(self.get_positions, sentence_lists))

        pmids = []
        for i in range(len(X)):
            pmids += [X[i]["pmid"]] * len(sentence_lists[i])

        return sentences, pmids, positions

    def sentence_scores(self, sentences):
        """return normalized scores for list of sentences independently from classifier type"""

        if hasattr(self.pipeline, 'predict_proba'):
            scores = self.normalized_probas(sentences)
        elif hasattr(self.pipeline, 'decision_function'):
            scores = self.normalized_dec_fns(sentences)
        else:
            scores = self.pipeline.predict(sentences)
        return scores

    def normalized_dec_fns(self, sentences):
        """map decision function values to relevance score in [-1,1] using maximum score in the corpus and a cutoff"""

        scores = self.pipeline.decision_function(sentences)
        return np.clip(np.array(scores) * 1.0 / self.max_score, -1.0, 1.0)

    def normalized_probas(self, sentences):
        """map probabilities to relevance score in [0,1]"""

        probas = self.pipeline.decision_function(sentences)
        return (np.abs(probas[:, 1]) - 0.5) * 2

    @staticmethod
    def get_positions(sentences):
        """return start and end position for each element in sentences"""

        end = -1
        positions = []

        for i in range(len(sentences)):
            start = end + 1
            end = start + len(sentences[i])
            positions.append((start, end))

        return positions
