import os
import pickle
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from semisuper import transformers, loaders, helpers
import build_corpus_and_ss_classifier
from build_classifier_from_ss_corpus import load_silver_standard, max_score_from_csv


# ----------------------------------------------------------------


# ----------------------------------------------------------------


def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)


# ----------------------------------------------------------------





# ----------------------------------------------------------------


def interactive():
    # TODO decide: input(), cmd() or sys.stdin?

    max_score = max_score_from_csv(load_silver_standard())

    def normalize_score(score):
        return min(score / max_score, 1.0)

    def get_positions(sentences):
        """return start and end position for each element in sentences"""

        end = -1
        positions = []

        for i in range(len(sentences)):
            start = end + 1
            end = start + len(sentences[i])
            positions.append((start, end))

        return positions

    pipeline = build_corpus_and_ss_classifier.train_pipeline(from_scratch=False, ratio=1.0)

    # predict test string to make sure pipeline is ready
    text = "test"
    pipeline.predict([text])

    while text:
        text = input()
        if not text:
            print()
            return
        sentences = transformers.sentence_tokenize(text)
        positions = get_positions(sentences)

        if hasattr(pipeline, 'predict_proba'):
            scores = (np.abs(pipeline.predict_proba(sentences)[:, 1]) - 0.5) * 2
        elif hasattr(pipeline, 'decision_function'):
            scores = np.clip(np.array(pipeline.decision_function(sentences)) / max_score, -1.0, 1.0)
        else:
            scores = pipeline.predict(sentences)

        for position, score, sentence in zip(positions, scores, sentences):
            if score > 0:
                score = normalize_score(score)  # divide by max score in corpus and cut off at 1.0
                print(position[0], position[1], score, "\n{}".format(sentence))
        print()
    return


if __name__ == "__main__":
    interactive()
