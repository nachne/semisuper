import time
import matplotlib.pyplot as plt
from semisuper.transformers import TokenizePreprocessor, TextStats
from semisuper.helpers import identity
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import random
from sklearn.manifold import MDS

from semisuper import loaders, pu_two_step, pu_biased_svm
from semisuper.helpers import num_rows

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

words, wordgram_range, chars, chargram_range, binary = [True, (1, 1), False, (1, 3), False]

pp = Pipeline([
    ('features', FeatureUnion(
        n_jobs=3,
        transformer_list=[
            ("wordgrams", None if not words else
            Pipeline([
                ("preprocessor", TokenizePreprocessor(rules=False, lemmatize=False)),
                ("word_tfidf", TfidfVectorizer(
                    analyzer='word',
                    # min_df=5, # TODO find reasonable value (5 <= n << 50)
                    tokenizer=identity, preprocessor=None, lowercase=False,
                    ngram_range=wordgram_range,
                    binary=binary, norm='l2' if not binary else None, use_idf=not binary))
            ])),
            ("chargrams", None if not chars else
            Pipeline([
                ("char_tfidf", TfidfVectorizer(
                    analyzer='char',
                    # min_df=5,
                    preprocessor=None, lowercase=False,
                    ngram_range=chargram_range,
                    binary=binary, norm='l2' if not binary else None, use_idf=not binary))
            ])),
            ("stats", None if binary else
            Pipeline([
                ("stats", TextStats()),
                ("vect", DictVectorizer())
            ]))
        ]))
])

sample = pp.transform(random.sample(civic, 100) + random.sample(abstracts, 100) +
                      random.sample(hocpos, 100) + random.sample(hocneg, 100) +
                      random.sample(piboso_outcome, 100) + random.sample(piboso_outcome, 100))

mds = MDS()
coords = mds.fit_transform(sample).embedding_

for x, y in coords[:(len(coords) / 6)]:
    plt.scatter(x, y, 'r')

for x, y in coords[(len(coords) / 6): (2 * len(coords) / 6)]:
    plt.scatter(x, y, 'm')

for x, y in coords[(2 * len(coords) / 6): (3 * len(coords) / 6)]:
    plt.scatter(x, y, 'b')

for x, y in coords[(3 * len(coords) / 6): (4 * len(coords) / 6)]:
    plt.scatter(x, y, 'c')

for x, y in coords[(4 * len(coords) / 6): (5 * len(coords) / 6)]:
    plt.scatter(x, y, 'g')

for x, y in coords[(5 * len(coords) / 6):]:
    plt.scatter(x, y, 'y')

plt.show()
