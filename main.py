import os
import pickle
import numpy as np

import build_corpus_and_ss_classifier
from semisuper import transformers

# ----------------------------------------------------------------

def file_path(file_relative):
    """return the correct file path given the file's path relative to calling script"""
    return os.path.join(os.path.dirname(__file__), file_relative)

def get_positions(sentences):
    end = -1
    positions = []

    for i in range(len(sentences)):
        start = end + 1
        end = start + len(sentences[i])
        positions.append((start, end))

    return positions

# ----------------------------------------------------------------


def main():
    # TODO decide: input(), cmd() or sys.stdin?

    pipeline = build_corpus_and_ss_classifier.train_pipeline(from_scratch=False, ratio=1.0)

    text = " "
    pipeline.predict([text])

    while text:
        text = input()
        sentences = transformers.sentence_tokenize(text)
        positions = get_positions(sentences)

        if hasattr(pipeline, 'decision_function'):
            scores = pipeline.decision_function(sentences)
        elif hasattr(pipeline, 'predict_proba'):
            scores = np.abs(pipeline.predict_proba(sentences)[:, 1]) - 0.5

        for sentence, position, score in zip(sentences, positions, scores):
            if score > 0:
                print(sentence, position, score)
        print()
    return

if __name__ == "__main__":
    main()