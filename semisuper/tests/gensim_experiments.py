from __future__ import absolute_import, division, print_function

import string
from collections import Counter
from operator import itemgetter

from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from semisuper import loaders, transformers, cleanup_corpora, helpers

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()
piboso_other = loaders.sentences_piboso_other()
piboso_outcome = loaders.sentences_piboso_outcome()

print("CIViC sentences:", len(civic))
print("Abstract sentences:", len(abstracts))
print("HoC positive sentences:", len(hocpos))
print("HoC negative sentences:", len(hocneg))
print("PIBOSO outcome sentences:", len(piboso_outcome))
print("PIBOSO other sentences:", len(piboso_other))

# make smaller
# civic = random.sample(civic, 100)
# abstracts = random.sample(abstracts, 100)

c, c_test = train_test_split(civic, test_size=0.2)
a, a_test = train_test_split(abstracts, test_size=0.2)
hp, hp_test = train_test_split(hocpos, test_size=0.2)
hn, hn_test = train_test_split(hocneg, test_size=0.2)
pp, pp_test = train_test_split(piboso_outcome, test_size=0.2)
pn, pn_test = train_test_split(piboso_other, test_size=0.2)

# tolerant
# hn = cleanup_corpora.remove_P_from_U(c, hn)
# hp = cleanup_corpora.remove_P_from_U(hn, hp)

# # strict
hn = cleanup_corpora.remove_P_from_U(c, hn)
hp = cleanup_corpora.remove_P_from_U(c, hp, inverse=True)


print("TRAINING SET FROM",
      "CIVIC",
      ", ABSTRACTS",
      ", HOC POS",
      ", HOC NEG"
      )
corpus_ = helpers.concatenate((
    c,
    a,
    hp,
    hn,
    # pp,
    # pn,
))

corpus_test = helpers.concatenate((
    c_test,
    a_test,
    hp_test,
    hn_test,
    # pp_test,
    # pn_test,
))

prepro = transformers.TokenizePreprocessor(rules=False, genia_opts=None)

corpus_vec = prepro.fit_transform(corpus_)
corpus_vec_test = prepro.transform(corpus_test)

# --------------------------------
# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(corpus_vec)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus_vec]

# Creating the object for LDA model using gensim library
# Running and Trainign LDA model on the document term matrix.
lda = LdaMulticore(doc_term_matrix, num_topics=6, id2word=dictionary, passes=50)

[print(x)
 for x in lda.print_topics(num_topics=-1, num_words=30)]


# --------------------------------


def topics2label(topics):
    return max(topics, key=itemgetter(1))[0]


def label_dist(corpus, dictionary, ldamodel):
    labels = [topics2label(ldamodel[x]) for x in
              [dictionary.doc2bow(doc) for doc in corpus]]

    freqs = Counter(labels)

    return [(lbl, freqs[lbl], freqs[lbl] / len(corpus)) for lbl in sorted(freqs)]


print("\n--------------------------------"
      "TEST SETS"
      "----------------------------------")

print("\nCIViC:")
[print(x, end="\t") for x in (label_dist(prepro.transform(c_test), dictionary, lda))]

print("\nAbstracts:")
[print(x, end="\t") for x in (label_dist(prepro.transform(a_test), dictionary, lda))]

print("\nHoC pos:")
[print(x, end="\t") for x in (label_dist(prepro.transform(hp_test), dictionary, lda))]

print("\nHoC neg:")
[print(x, end="\t") for x in (label_dist(prepro.transform(hn_test), dictionary, lda))]

print("\nPIBOSO outcome:")
[print(x, end="\t") for x in (label_dist(prepro.transform(pp_test), dictionary, lda))]

print("\nPIBOSO other:")
[print(x, end="\t") for x in (label_dist(prepro.transform(pn_test), dictionary, lda))]

print("\n--------------------------------"
      "FULL CORPORA"
      "----------------------------------")

print("\nCIViC:")
[print(x, end="\t") for x in (label_dist(prepro.transform(civic), dictionary, lda))]

print("\nAbstracts:")
[print(x, end="\t") for x in (label_dist(prepro.transform(abstracts), dictionary, lda))]

print("\nHoC pos:")
[print(x, end="\t") for x in (label_dist(prepro.transform(hocpos), dictionary, lda))]

print("\nHoC neg:")
[print(x, end="\t") for x in (label_dist(prepro.transform(hocneg), dictionary, lda))]

print("\nPIBOSO outcome:")
[print(x, end="\t") for x in (label_dist(prepro.transform(piboso_outcome), dictionary, lda))]

print("\nPIBOSO other:")
[print(x, end="\t") for x in (label_dist(prepro.transform(piboso_other), dictionary, lda))]

# --------------------------------
# obsolete tutorial stuff

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized
