import string
from collections import Counter
from operator import itemgetter

import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from semisuper import loaders, transformers

civic, abstracts = loaders.sentences_civic_abstracts()
hocpos, hocneg = loaders.sentences_HoC()

# make smaller
# civic = random.sample(civic, 100)
# abstracts = random.sample(abstracts, 100)

corpus = (
    []
    + civic
    + abstracts
    + hocpos
    + hocneg
)

pp = transformers.TokenizePreprocessor(rules=False, lemmatize=False)
corp_tokenized = pp.transform(corpus)

# --------------------------------
# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(corp_tokenized)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in corp_tokenized]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=2, id2word=dictionary, passes=50)

[print(x) for x in ldamodel.print_topics(num_topics=-1, num_words=20)]


# --------------------------------

def topics2label(topics):
    return max(topics, key=itemgetter(1))[0]


# y_true = [1] * len(civic) + [0] * len(abstracts)
# y_pred = [topics2label(ldamodel[dictionary.doc2bow(x)])
#           for x in corp_tokenized]
#
# print(clsr(y_true, y_pred))

def label_dist(corpus, dict, ldamodel):
    labels = [topics2label(ldamodel[x]) for x in
              [dict.doc2bow(doc) for doc in corpus]]

    freqs = Counter(labels)

    return [(lbl, freqs[lbl], freqs[lbl] / len(corpus)) for lbl in sorted(freqs)]


print("CIViC:")
[print(x, end="\t") for x in (label_dist(pp.transform(civic), dictionary, ldamodel))]
print()

print("Abstracts:")
[print(x, end="\t") for x in (label_dist(pp.transform(abstracts), dictionary, ldamodel))]
print()

print("HoC pos:")
[print(x, end="\t") for x in (label_dist(pp.transform(hocpos), dictionary, ldamodel))]
print()

print("HoC neg:")
[print(x, end="\t") for x in (label_dist(pp.transform(hocneg), dictionary, ldamodel))]
print()

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
